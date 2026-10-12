"""
K2 — Domain breadth per-cell (R8 fix).
K2[cell] = distinct domains seen / total domains seen.
"""
from collections import defaultdict


def compute_k2_percell(cells, cell_domains):
    """
    Args:
        cells: iterable of Cell
        cell_domains: {cell_id: set(domain_names)}
    Returns:
        {cell_id: K2 in [0, 1]}
    """
    all_domains = set()
    for doms in cell_domains.values():
        all_domains.update(doms)
    n_total = max(1, len(all_domains))

    out = {}
    for c in cells:
        doms = cell_domains.get(c.cell_id, set())
        out[c.cell_id] = len(doms) / n_total
    return out


def update_cell_domains(cell_domains, cells, active_ids, domain_name):
    """Add domain_name to each active cell's domain set."""
    for cid in active_ids:
        cell_domains.setdefault(cid, set()).add(domain_name)