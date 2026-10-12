#!/usr/bin/env python3
"""
R5-compliant Cell Manager.

Orchestrates the full cell lifecycle:
  1. Load cells + DNA + plexus from disk.
  2. Receive activations for each cell (per step).
  3. Compute K (6D) and S = ||K||_omega.
  4. Update S_history; trigger splits when S >= S* for T_cons steps.
  5. Update plexus edges from co-activations.
  6. Save updated state.

This is the top-level object for a single model (Mother, Child, Grandchild).
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from .saturation import (
    Cell, save_cells, load_cells, compute_S,
    compute_K1_sequential_coverage, compute_K3_abstraction,
    compute_K5_temporal_freshness, compute_K6_encompassment_real,
    load_facets, S_STAR, T_CONSECUTIVE, DECAY_LAMBDA,
)
from .dna_kernel import DNA, save_dna, load_dna, random_dna, decide
import sys
sys.path.insert(0, '/data/berna-r7/core/code')
from encompassment_engine_v2 import EncompassmentEngineV2
from .cell_split import apply_split, is_ready_to_split
from .plexus import (
    Plexus, save_plexus, load_plexus,
    update_from_coactivations, compute_K4,
)


class CellManager:
    """Manages all cells of one model."""

    def __init__(self, model_uuid: str, base_dir: Optional[Path] = None):
        self.model_uuid = model_uuid
        self.base_dir = Path(base_dir) if base_dir else Path(f"/data/berna-r7/{model_uuid}")
        self.cells: List[Cell] = []
        self.dnas: Dict[str, DNA] = {}
        self.plexus = Plexus()
        self.current_step = 0
        self.facets_map = load_facets()
        self.history: List[dict] = []
        self.encompassment = EncompassmentEngineV2()

    # --- paths ---
    def _cells_path(self) -> Path:
        return self.base_dir / "cells.json"

    def _dna_dir(self) -> Path:
        return self.base_dir / "dna"

    def _plexus_path(self) -> Path:
        return self.base_dir / "plexus.json"

    # --- persistence ---
    def load(self) -> bool:
        """Load cells, DNA, plexus from disk. Returns True if successful."""
        if not self._cells_path().exists():
            return False
        self.cells = load_cells(self._cells_path())

        dna_dir = self._dna_dir()
        if dna_dir.exists():
            for f in dna_dir.glob("*.dna.json"):
                d = load_dna(f)
                self.dnas[d.cell_id] = d

        if self._plexus_path().exists():
            self.plexus = load_plexus(self._plexus_path())

        return True

    def save(self):
        """Save all state."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        save_cells(self.cells, self._cells_path())

        dna_dir = self._dna_dir()
        dna_dir.mkdir(parents=True, exist_ok=True)
        for cid, dna in self.dnas.items():
            save_dna(dna, dna_dir / f"{cid}.dna.json")

        save_plexus(self.plexus, self._plexus_path())

    # --- measurement ---
    def measure_cell(self, cell: Cell, activations: np.ndarray,
                     activated_facets: Optional[set] = None,
                     domain: Optional[str] = None) -> dict:
        """
        Compute K, S for a single cell from activations.
        activations: shape [seq_len, num_neurons]
        """
        # K1 (L) - sequential coverage
        K1 = compute_K1_sequential_coverage(activations)

        # K2 (W) - domain breadth (stub, deferred)
        K2 = 0.0

        # K3 (H) - abstraction
        K3 = compute_K3_abstraction(activations)

        # K4 (D) - reasoning depth via plexus
        K4 = compute_K4(self.plexus, cell.cell_id)

        # K5 (T) - temporal freshness
        K5 = compute_K5_temporal_freshness(
            cell.last_updated_step, self.current_step, DECAY_LAMBDA
        )

        # K6 (E) - encompassment (if activated_facets + domain provided)
        # K6 from EncompassmentEngine (CEO dimension)
        K6 = self.encompassment.K6(cell.cell_id)

        K = [K1, K2, K3, K4, K5, K6]
        S = compute_S(K, cell.omega)

        cell.K = K
        cell.S = S
        cell.S_history.append(S)
        cell.last_updated_step = self.current_step

        return {"cell_id": cell.cell_id, "K": K, "S": S}

    # --- main step ---
    def step(self, activations_by_cell: Dict[str, np.ndarray],
             coactivations: Optional[Dict[tuple, float]] = None,
             facets_by_cell: Optional[Dict[str, dict]] = None,
             training_step: Optional[int] = None) -> dict:
        """
        One full iteration.

        activations_by_cell: {cell_id: np.array [seq_len, num_neurons]}
        coactivations: {(cell_a, cell_b): A_uv} for plexus update
        facets_by_cell: {cell_id: {"activated": set, "domain": str}}
        training_step: actual training step (for split cooldown & K5)
        """
        if training_step is not None:
            self.current_step = training_step
        else:
            self.current_step += 1
        measured = []
        splits_triggered = []

        # 1) Measure all cells
        for cell in self.cells:
            if cell.state != "active":
                continue
            if cell.cell_id not in activations_by_cell:
                continue
            a = activations_by_cell[cell.cell_id]
            facets = facets_by_cell.get(cell.cell_id) if facets_by_cell else None
            domain = facets.get("domain") if facets else None
            activated = facets.get("activated") if facets else None
            info = self.measure_cell(cell, a, activated_facets=activated, domain=domain)
            measured.append(info)

        # 2) Update plexus
        if coactivations:
            update_from_coactivations(self.plexus, coactivations)

        # 3) Check splits
        for cell in list(self.cells):
            if cell.state != "active":
                continue
            dna = self.dnas.get(cell.cell_id)
            if dna is None:
                continue
            if is_ready_to_split(cell, current_step=self.current_step):
                result = apply_split(cell, dna, registry_path=None)
                if result["split"]:
                    # Mark parent's last_split_step for cooldown
                    cell.last_split_step = self.current_step
                    splits_triggered.append(result)
                    # Register children
                    from .cell_split import split_cell
                    ca, cb, da, db = split_cell(cell, dna, split_id=cell.cell_id)
                    # Children inherit cooldown from current step
                    ca.last_split_step = self.current_step
                    cb.last_split_step = self.current_step
                    self.cells.extend([ca, cb])
                    self.dnas[ca.cell_id] = da
                    self.dnas[cb.cell_id] = db

        summary = {
            "step": self.current_step,
            "measured": len(measured),
            "splits": splits_triggered,
            "n_cells_active": sum(1 for c in self.cells if c.state == "active"),
            "n_cells_total": len(self.cells),
        }
        self.history.append(summary)
        return summary


if __name__ == "__main__":
    import random
    rng = random.Random(42)

    # Build a small test model (fewer cells for speed)
    from .cell_birth import initialize_model_cells, make_dna_for_cells

    test_cfg = {"num_hidden_layers": 2, "intermediate_size": 64}
    cells = initialize_model_cells(test_cfg, model_uuid="test-model", n_cells_per_layer=4)
    dnas = make_dna_for_cells(cells)

    mgr = CellManager("test-model", base_dir=Path("/tmp/berna_r7_test"))
    mgr.cells = cells
    mgr.dnas = dnas

    print(f"Initial cells: {len(mgr.cells)}")
    print(f"DNA records:   {len(mgr.dnas)}")
    print()

    # Run 5 steps with random activations
    for t in range(5):
        activations = {}
        for c in mgr.cells:
            if c.state == "active":
                n = c.num_neurons()
                activations[c.cell_id] = np.abs(np.random.randn(16, n)).astype(np.float32)

        summary = mgr.step(
            activations_by_cell=activations,
            coactivations={(mgr.cells[0].cell_id, mgr.cells[1].cell_id): 0.7},
            facets_by_cell=None,
        )
        print(f"Step {summary['step']}: measured={summary['measured']} "
              f"splits={len(summary['splits'])} "
              f"active={summary['n_cells_active']}/{summary['n_cells_total']}")

    print()
    print(f"Final active cells: {sum(1 for c in mgr.cells if c.state == 'active')}")
    print(f"Final total cells:  {len(mgr.cells)}")
    print(f"Splits happened:    {len([c for c in mgr.cells if c.state == 'split'])}")
