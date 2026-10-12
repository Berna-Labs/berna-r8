"""
Cell system integration for Berna R7 (R5-functional).

Bridges:
  - modeling_berna_v2.CellAwareMLP (gates driven by DNA)
  - cells/ package (CellManager, saturation, dna_kernel, plexus)
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from cells.cell_manager import CellManager
from cells.cell_birth import birth_cells_for_layer
from cells.dna_kernel import DNA, random_dna, signal_route
from cells.saturation import Cell, N_CELLS_PER_LAYER
from cognitive_log import CognitiveLog, pattern_signature, hamming_similarity
from dynamic_birth import BirthRegistry, birth_from_pattern, find_birth_candidate


# ====================== Build =========================

def _identity_dna(cell_id, seed=42):
    """
    DNA with a-gene = 1.0 across all 100 chromosomes.
    signal_route() returns mean of a-gene = 1.0 → gate = 1.0 → identity.
    Checkpoint 2600 stays compatible (no abrupt behavior change).
    """
    dna = random_dna(cell_id, seed=seed)
    for i in range(len(dna.genes)):
        dna.genes[i][0] = 1.0
    dna.signature = dna._compute_signature()
    return dna


def build_cells_and_dnas(config_dict, model_uuid="mother-001",
                          n_cells_per_layer=N_CELLS_PER_LAYER):
    """
    Build cells for all MLP layers + identity DNA for each.
    Returns (cells_by_layer, dnas).
    """
    num_layers = int(config_dict["num_hidden_layers"])
    interm = int(config_dict["intermediate_size"])

    cells_by_layer = {}
    all_cells = []
    for layer_idx in range(num_layers):
        layer_cells = birth_cells_for_layer(
            layer_idx=layer_idx,
            intermediate_size=interm,
            n_cells=n_cells_per_layer,
            model_uuid=model_uuid,
        )
        cells_by_layer[layer_idx] = layer_cells
        all_cells.extend(layer_cells)

    dnas = {}
    for i, c in enumerate(all_cells):
        dnas[c.cell_id] = _identity_dna(c.cell_id, seed=42 + i)
    return cells_by_layer, dnas


# ====================== Runtime System =========================

class CellSystem:
    """
    Owns cells_by_layer + dnas. Runs CellManager.step every N train steps.
    """

    def __init__(self, model_uuid="mother-001",
                 base_dir="/data/berna-r7/core/cells_runtime",
                 measure_every=50,
                 verbose=True):
        self.model_uuid = model_uuid
        self.base_dir = Path(base_dir)
        self.measure_every = measure_every
        self.verbose = verbose

        self.cells_by_layer: Dict[int, List[Cell]] = {}
        self.dnas: Dict[str, DNA] = {}
        self.manager: Optional[CellManager] = None
        self._last_step = 0
        self._n_split_events = 0
        self.cognitive_log = CognitiveLog(
            Path(base_dir) / model_uuid / "cognitive",
            model_uuid=model_uuid,
        )
        self.birth_registry = BirthRegistry()
        self._n_birth_events = 0

    # ---------- attach / load ----------

    def attach_to_model(self, model, config_dict):
        """Build fresh cells + DNA, push into model's CellAwareMLP layers."""
        cells_by_layer, dnas = build_cells_and_dnas(
            config_dict, model_uuid=self.model_uuid
        )
        self.cells_by_layer = cells_by_layer
        self.dnas = dnas

        for layer_idx, layer in enumerate(model.model.layers):
            layer.mlp.cells = cells_by_layer[layer_idx]
            layer.mlp.dnas = dnas
            layer.mlp._gate_cache = None
            layer.mlp._cache_version = -1

        self.manager = CellManager(
            model_uuid=self.model_uuid,
            base_dir=self.base_dir / self.model_uuid,
        )
        all_cells = []
        for cs in cells_by_layer.values():
            all_cells.extend(cs)
        self.manager.cells = all_cells
        self.manager.dnas = dnas

        self._assign_slots(model)

        if self.verbose:
            print(f"[CellSystem] attached: {len(all_cells)} cells, "
                  f"{len(dnas)} DNA records")

    def load_state(self):
        """Load previous state from disk. Overrides attached cells."""
        if self.manager is None:
            return False
        ok = self.manager.load()
        if ok:
            self.cells_by_layer = {}
            for c in self.manager.cells:
                self.cells_by_layer.setdefault(c.layer_idx, []).append(c)
            self.dnas = self.manager.dnas
            if self.verbose:
                print(f"[CellSystem] loaded: {len(self.manager.cells)} cells")
        return ok

    def _assign_slots(self, model):
        """Assign each cell a unique slot index per layer (0..255)."""
        for layer_idx, layer in enumerate(model.model.layers):
            mapping = {}
            for i, c in enumerate(layer.mlp.cells):
                mapping[c.cell_id] = i % 256
            layer.mlp.assign_slots(mapping)

    def sync_into_model(self, model):
        """Push loaded cells + dnas into model's layers."""
        for layer_idx, layer in enumerate(model.model.layers):
            layer.mlp.cells = self.cells_by_layer.get(layer_idx, [])
            layer.mlp.dnas = self.dnas
            layer.mlp._gate_cache = None
            layer.mlp._cache_version = -1
        self._assign_slots(model)


    # ---------- measurement ----------

    def _enable_measure(self, model, on=True):
        """Toggle _measure_now on all CellAwareMLP layers."""
        for layer in model.model.layers:
            layer.mlp._measure_now = on

    def _collect_activations(self, model):
        """Collect detached activations from last forward."""
        acts = {}
        for layer_idx, layer in enumerate(model.model.layers):
            inter = layer.mlp._last_inter_detached
            if inter is None:
                continue
            inter0 = inter[0].float()
            for cell in layer.mlp.cells:
                if cell.state != "active":
                    continue
                a = inter0[:, cell.neuron_start:cell.neuron_end]
                acts[cell.cell_id] = a.detach().cpu().numpy()
        return acts

    def _measure_forward(self, model, dataset, seq_len=512):
        """Run one small forward pass with _measure_now=True."""
        import torch as _torch
        self._enable_measure(model, True)
        try:
            idx = _torch.randint(0, len(dataset), (1,)).item()
            item = dataset[idx]
            x = item[0] if isinstance(item, (tuple, list)) else item
            x = x[:seq_len].unsqueeze(0)
            device = next(model.parameters()).device
            x = x.to(device)
            with _torch.no_grad():
                _ = model(input_ids=x)
        finally:
            self._enable_measure(model, False)

    def _compute_coactivations(self, acts, top_k=20, thresh=0.5):
        """
        A_uv = cosine similarity of mean activations.
        Only for pairs where both mean magnitudes > thresh.
        Keep top_k strongest pairs.
        """
        means = {cid: float(np.mean(np.abs(a))) for cid, a in acts.items()}
        active = [cid for cid, m in means.items() if m > thresh]
        pairs = []
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                u, v = active[i], active[j]
                au = acts[u].reshape(-1)
                av = acts[v].reshape(-1)
                n = min(len(au), len(av))
                if n < 2:
                    continue
                au = au[:n]; av = av[:n]
                nu = float(np.linalg.norm(au)) + 1e-9
                nv = float(np.linalg.norm(av)) + 1e-9
                cos = float(np.dot(au, av) / (nu * nv))
                pairs.append((cos, (u, v)))
        pairs.sort(reverse=True)
        return {p: c for c, p in pairs[:top_k]}

    def maybe_step(self, model, global_step, dataset=None):
        """
        If global_step is a multiple of measure_every → run one CellManager step.
        Returns summary dict or None.
        """
        if global_step % self.measure_every != 0:
            return None

        t0 = time.time()
        # Run a small measure-forward so _last_inter_detached is populated
        if dataset is not None:
            try:
                self._measure_forward(model, dataset)
            except Exception as e:
                if self.verbose:
                    print(f"[CellSystem] measure forward failed: {e}")
        acts = self._collect_activations(model)
        if not acts:
            if self.verbose:
                print(f"[CellSystem] step {global_step}: no activations collected")
            return None
        coacts = self._compute_coactivations(acts)

        # === Pattern fingerprint per cell ===
        for layer in model.model.layers:
            for cell in layer.mlp.cells:
                if cell.state != "active" or cell.cell_id not in acts:
                    continue
                fp = pattern_signature(acts[cell.cell_id], top_k=50)
                if not getattr(cell, "fp_history", None):
                    cell.fp_history = []
                cell.fp_history.append(fp)
                if len(cell.fp_history) > 100:
                    cell.fp_history = cell.fp_history[-100:]
                if not getattr(cell, "pattern_sig", ""):
                    cell.pattern_sig = fp
                elif hamming_similarity(fp, cell.pattern_sig) < 0.5:
                    cell.pattern_sig = fp

        # === Refresh all K6 centralities ===
        try:
            self.manager.encompassment.refresh_all(hamming_fn=hamming_similarity)
        except Exception as e:
            if self.verbose:
                print(f"[CellSystem] refresh_all failed: {e}")

        # === Dynamic births ===
        for layer_idx, layer in enumerate(model.model.layers):
            for cell in list(layer.mlp.cells):
                if cell.state != "active" or cell.cell_id not in acts:
                    continue
                fp = pattern_signature(acts[cell.cell_id], top_k=50)
                # Register with EncompassmentEngine (K6 CEO)
                # Register with EncompassmentEngine V2 (K6 CEO)
                self.manager.encompassment.register(cell.cell_id, fp, layer_idx=cell.layer_idx)
                if hamming_similarity(fp, cell.pattern_sig) < 0.3:
                    parent_dna = self.dnas.get(cell.cell_id)
                    if parent_dna is None:
                        continue
                    new_cell, new_dna = birth_from_pattern(
                        parent_cell=cell,
                        parent_dna=parent_dna,
                        new_pattern_sig=fp,
                        step=global_step,
                        model_uuid=self.model_uuid,
                        registry=self.birth_registry,
                    )
                    if new_cell is not None:
                        self.manager.cells.append(new_cell)
                        self.manager.dnas[new_cell.cell_id] = new_dna
                        self.dnas[new_cell.cell_id] = new_dna
                        self._n_birth_events += 1
                        self.cognitive_log.record_birth(
                            cell_id=new_cell.cell_id,
                            parent_id=new_cell.parent_id,
                            pattern_sig=fp,
                            step=global_step,
                            layer_idx=layer_idx,
                        )
                        if self.verbose:
                            print(f"[CellSystem] BIRTH: {new_cell.cell_id} "
                                  f"(parent={new_cell.parent_id}, step={global_step})")

        summary = self.manager.step(
            activations_by_cell=acts,
            coactivations=coacts,
            facets_by_cell=None,
            training_step=global_step,
        )

        # === Log splits ===
        for sp in summary.get("splits", []):
            children = sp.get("children", ["?", "?"])
            self.cognitive_log.record_split(
                parent_id=sp.get("parent_id", "?"),
                child_a=children[0],
                child_b=children[1] if len(children) > 1 else "?",
                S=sp.get("S", 0.0),
                step=global_step,
                layer_idx=-1,
            )
        self.manager.save()

        # Re-sync model after potential splits
        self.cells_by_layer = {}
        for c in self.manager.cells:
            self.cells_by_layer.setdefault(c.layer_idx, []).append(c)
        self.dnas = self.manager.dnas
        self.sync_into_model(model)

        self._last_step = global_step
        n_splits = len(summary.get("splits", []))
        self._n_split_events += n_splits

        if self.verbose:
            dt = time.time() - t0
            print(f"[CellSystem] step {global_step}: "
                  f"active={summary['n_cells_active']}/{summary['n_cells_total']} "
                  f"splits_now={n_splits} splits_total={self._n_split_events} "
                  f"({dt:.1f}s)")

        return summary

    def self_narrative(self):
        return self.cognitive_log.sample_for_training(n=3)

    def save_narrative(self):
        return self.cognitive_log.write_narrative()

    def stats(self):
        n_total = len(self.manager.cells) if self.manager else 0
        n_active = sum(1 for c in self.manager.cells if c.state == "active") \
                   if self.manager else 0
        return {
            "n_cells_total": n_total,
            "n_cells_active": n_active,
            "n_split_events": self._n_split_events,
        }