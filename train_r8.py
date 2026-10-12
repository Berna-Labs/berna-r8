"""
Berna R8 — Fine-tune R7 v3 with full cell system + bounded forgetting.

Usage:
  python train_r8.py --dry-run --steps 5 --device cpu
  python train_r8.py --steps 2000 --device cuda
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "cells"))

from modeling_berna_v2 import BernaConfig, BernaForCausalLM
from cell_integration import CellSystem
from forgetting.bounded import (
    install_freeze_hooks, evaluate_loss, measure_forgetting,
    save_forgetting_report,
)
from safetensors.torch import load_file


def load_token_data(data_dir, files, seq_len):
    """Return dict {domain: np.array of int tokens}."""
    out = {}
    for domain, fname in files.items():
        p = Path(data_dir) / fname
        if not p.exists():
            print(f"  [WARN] {p} not found")
            continue
        out[domain] = np.load(p, mmap_mode="r")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="hf_model/checkpoints/step_00010200/model")
    ap.add_argument("--data-dir", default="data/tokenized")
    ap.add_argument("--out", default="checkpoints_r8")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--measure-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Config
    with open(ckpt_dir / "config.json") as f:
        cfg_dict = json.load(f)
    cfg = BernaConfig(**cfg_dict)
    print(f"Config: {cfg.num_hidden_layers}L x {cfg.hidden_size}d")

    # 2. Model
    from cell_integration import build_cells_and_dnas; _cbl, _dn = build_cells_and_dnas(cfg_dict); model = BernaForCausalLM(cfg, cells_by_layer=_cbl, dnas=_dn)
    sd = load_file(str(ckpt_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded R7 v3. missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(args.device)

    # 3. CellSystem
    cellsys = CellSystem(
        model_uuid="mother-001",
        base_dir=str(out_dir / "cells_runtime"),
        measure_every=args.measure_every,
        verbose=True,
    )
    cellsys.attach_to_model(model, cfg_dict)
    loaded = cellsys.load_state()  # ← R8: load 896 trained cells
    print(f"[CellSystem] load_state: {loaded}")
    cellsys.sync_into_model(model)
    cellsys.cell_domains = {}
    n_cells = len(cellsys.manager.cells) if cellsys.manager else 0
    n_active = sum(1 for c in cellsys.manager.cells if c.state == "active") if cellsys.manager else 0
    print(f"[CellSystem] Cells: {n_cells} total, {n_active} active")

    # 4. Freeze hooks
    handles = install_freeze_hooks(model, cellsys.cells_by_layer)
    print(f"Installed {len(handles)} freeze hooks")

    # 5. Data
    files = {"eng": "eng.npy", "math": "math.npy", "code": "code.npy"}
    data = load_token_data(args.data_dir, files, args.seq_len)
    if not data:
        print("No data found. Aborting.")
        return
    domains = list(data.keys())
    print(f"Domains: {domains}")

    # 6. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.95), weight_decay=0.1)

    # 7. Baseline losses
    baseline = {}
    holdout = {}
    for dom, arr in data.items():
        n = min(100_000, len(arr) - 1)
        holdout[dom] = np.asarray(arr[-n:], dtype=np.int64)
        baseline[dom] = evaluate_loss(model, holdout[dom],
                                       seq_len=args.seq_len, max_batches=3,
                                       device=args.device)
        print(f"Baseline {dom}: loss={baseline[dom]:.4f}")

    # 8. Train
    model.train()
    step = 0
    accum = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    seq_len = args.seq_len

    while step < args.steps:
        # sample a batch
        dom = np.random.choice(domains)
        arr = data[dom]
        max_i = (len(arr) - seq_len - 1)
        i = np.random.randint(0, max_i)
        buf = arr[i:i + seq_len + 1]
        x = torch.from_numpy(buf[:-1].astype(np.int64)).unsqueeze(0).to(args.device)
        y = torch.from_numpy(buf[1:].astype(np.int64)).unsqueeze(0).to(args.device)

        out = model(input_ids=x, labels=y)
        loss = out.loss / args.grad_accum
        loss.backward()
        accum += 1

        if accum >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            step += 1

            # record domain per active cell
            for layer in model.model.layers:
                for c in layer.mlp.cells:
                    if c.state == "active":
                        cellsys.cell_domains.setdefault(c.cell_id, set()).add(dom)

            if step % 10 == 0:
                dt = time.time() - t0
                print(f"step {step:>5} | loss {out.loss.item():.4f} | {dt:.0f}s")

            # CellSystem step
            cellsys.maybe_step(model, step, dataset=None)

            # Save
            if step % args.save_every == 0:
                cellsys.manager.save()
                torch.save(model.state_dict(), out_dir / f"step_{step:06d}.pt")
                print(f"[SAVE] step {step}")

            if args.dry_run and step >= args.steps:
                break

    # 9. Forgetting measurement
    print("\n=== Forgetting measurement ===")
    results = measure_forgetting(model, holdout, baseline, device=args.device)
    for task, r in results.items():
        print(f"  {task}: L_before={r['L_before']:.4f} L_after={r['L_after']:.4f} "
              f"F_j={r['F_j']:.6e} bounded={r['bounded']}")
    save_forgetting_report(results, out_dir / "forgetting_report.json")

    # 10. K5/K6 variance check
    print("\n=== K5/K6 variance check ===")
    import statistics as st
    k5s = [c.K[4] for c in cellsys.manager.cells if c.state == "active"]
    k6s = [c.K[5] for c in cellsys.manager.cells if c.state == "active"]
    k2s = [c.K[1] for c in cellsys.manager.cells if c.state == "active"]
    if k5s:
        print(f"  K5: mean={st.mean(k5s):.4f} stdev={st.stdev(k5s) if len(k5s)>1 else 0:.6f}")
        print(f"  K6: mean={st.mean(k6s):.4f} stdev={st.stdev(k6s) if len(k6s)>1 else 0:.6f}")
        print(f"  K2: mean={st.mean(k2s):.4f} stdev={st.stdev(k2s) if len(k2s)>1 else 0:.6f}")

    # 11. Final save
    cellsys.manager.save()
    torch.save(model.state_dict(), out_dir / "final.pt")
    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
