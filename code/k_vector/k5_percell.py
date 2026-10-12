"""
K5 — Freshness per-cell (R8 fix v3).
"""
import math
import numpy as np

DECAY_LAMBDA = 0.001
ABS_THRESHOLD = 0.5  # below this magnitude = not "significant"


def compute_k5_percell(cells, current_step):
    out = {}
    for c in cells:
        last = getattr(c, "last_significant_step", None)
        if last is None:
            last = 0
        age = min(500, max(0, current_step - last))
        out[c.cell_id] = math.exp(-DECAY_LAMBDA * age)
    return out


def update_significance(cell, activations, current_step,
                        abs_threshold=ABS_THRESHOLD):
    """
    Cell is 'significant' if its mean |activation| > abs_threshold.
    Simple, interpretable, produces real variance across cells.
    """
    a = np.abs(activations)
    if a.size == 0:
        return False
    m = float(a.mean())
    cell._last_mag = m  # for debugging
    
    if m > abs_threshold:
        cell.last_significant_step = current_step
        return True
    return False