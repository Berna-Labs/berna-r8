"""
K6 — Encompassment per-cell (R8 fix).

R7 bug: K6 = 0.6138 uniform across all cells
Cause: encompassment_engine_v2 uses a global pattern index.

R8 fix: K6[cell] = fraction of layer-distinct fingerprints
       that this cell has seen (via fp_history).
"""
import numpy as np


def _hamming_sim(a, b):
    """Similarity = 1 - normalized hamming distance on hex strings."""
    if not a or not b or len(a) != len(b):
        return 0.0
    try:
        x = int(a, 16) ^ int(b, 16)
    except ValueError:
        return 0.0
    n_bits = len(a) * 4
    dist = bin(x).count("1") / n_bits
    return 1.0 - dist


def build_layer_fingerprints(cells, history_window=100):
    """Group distinct fingerprints by layer."""
    out = {}
    for c in cells:
        h = getattr(c, "fp_history", [])
        if not h:
            continue
        out.setdefault(c.layer_idx, set()).update(h[-history_window:])
    return out


def compute_k6_percell(cells, layer_fps, sim_threshold=0.7,
                       history_window=100):
    """
    For each cell:
      K6 = fraction of layer's distinct fingerprints
           that the cell has 'seen' (similarity >= sim_threshold).

    Returns {cell_id: K6 in [0, 1]}.
    """
    out = {}
    for c in cells:
        lfps = layer_fps.get(c.layer_idx, set())
        if not lfps:
            out[c.cell_id] = 0.0
            continue
        history = getattr(c, "fp_history", [])
        if not history:
            out[c.cell_id] = 0.0
            continue

        recent = history[-history_window:]
        matched = 0
        for fp in lfps:
            for h in recent:
                if _hamming_sim(fp, h) >= sim_threshold:
                    matched += 1
                    break

        out[c.cell_id] = matched / max(1, len(lfps))
    return out