"""Save DNA for ALL cells (R8 fix)."""
from pathlib import Path
from .dna_kernel import save_dna

def save_all_dna(cells, dnas, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    missing = 0
    for cell in cells:
        dna = dnas.get(cell.cell_id)
        if dna is None:
            missing += 1
            continue
        dna.cell_id = cell.cell_id
        save_dna(dna, out_dir / f"{cell.cell_id}.dna.json")
        saved += 1
    print(f"[save_all_dna] saved={saved}, missing={missing}")
    return saved