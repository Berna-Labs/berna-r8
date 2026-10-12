"""
Bounded Forgetting (R8) — full implementation.

Definitions:
  P_j(t)  = performance on task j at time t (we use 1/loss).
  F_j     = max(0, L_after - L_before)
  Goal:   F_j <= EPS_NEG = 1e-6

Mechanism:
  1. Freeze saturated cells (S >= S* for T consecutive steps).
  2. Frozen cells: gradient hooks zero out their parameter ranges.
  3. New domain → new cells born (disabled in stub), old cells untouched.
"""
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch


EPS_NEG = 1e-6
S_STAR = 0.85
T_FREEZE = 3


class CellFreezer:
    def __init__(self, S_star: float = S_STAR, T: int = T_FREEZE):
        self.S_star = S_star
        self.T = T
        self.frozen_count = 0

    def maybe_freeze(self, cell, current_step: int) -> bool:
        if getattr(cell, "frozen", False):
            return False
        recent = cell.S_history[-self.T:] if hasattr(cell, "S_history") else []
        if len(recent) >= self.T and all(s >= self.S_star for s in recent):
            cell.frozen = True
            cell.freeze_step = current_step
            self.frozen_count += 1
            return True
        return False

    def freeze_all_saturated(self, cells, current_step: int) -> int:
        n = 0
        for c in cells:
            if c.state == "active" and self.maybe_freeze(c, current_step):
                n += 1
        return n


def install_freeze_hooks(model, cells_by_layer):
    """
    Register gradient hooks on CellAwareMLP gate/up projections
    that zero gradients for frozen cells' neuron ranges.
    """
    handles = []

    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        cells = cells_by_layer.get(layer_idx, [])
        if not cells:
            continue

        def make_hook(cells_ref):
            def hook(grad):
                g = grad.clone()
                for c in cells_ref:
                    if getattr(c, "frozen", False):
                        g[..., c.neuron_start:c.neuron_end] = 0.0
                return g
            return hook

        try:
            h1 = mlp.gate_proj.weight.register_hook(make_hook(cells))
            handles.append(h1)
        except AttributeError:
            pass
        try:
            h2 = mlp.up_proj.weight.register_hook(make_hook(cells))
            handles.append(h2)
        except AttributeError:
            pass

    return handles


@torch.no_grad()
def evaluate_loss(model, tokens, seq_len=2048, max_batches=10, device="cpu"):
    """Compute average loss on a token stream."""
    was_training = model.training
    model.eval()
    tokens = np.asarray(tokens, dtype=np.int64)
    if len(tokens) < seq_len + 1:
        if was_training:
            model.train()
        return float("nan")
    n = min(len(tokens) // (seq_len + 1), max_batches)
    if n == 0:
        if was_training:
            model.train()
        return float("nan")
    total = 0.0
    for i in range(n):
        s = i * seq_len
        x = torch.from_numpy(tokens[s:s + seq_len]).unsqueeze(0).to(device)
        y = torch.from_numpy(tokens[s + 1:s + seq_len + 1]).unsqueeze(0).to(device)
        out = model(input_ids=x, labels=y)
        total += float(out.loss.item())
    if was_training:
        model.train()
    return total / n


def measure_forgetting(model, task_tokens, baseline_losses, device="cpu"):
    """
    F_j = max(0, L_after - L_before).
    Bounded if all F_j <= EPS_NEG.
    """
    results = {}
    for task, tokens in task_tokens.items():
        l_before = baseline_losses[task]
        l_after = evaluate_loss(model, tokens, device=device)
        F_j = max(0.0, l_after - l_before)
        results[task] = {
            "L_before": l_before,
            "L_after": l_after,
            "F_j": F_j,
            "bounded": F_j <= EPS_NEG,
        }
    return results


def save_forgetting_report(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "max_F": max((r["F_j"] for r in results.values()), default=0.0),
        "all_bounded": all(r["bounded"] for r in results.values()),
        "per_task": results,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
