---
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: text-generation
tags: [language-model, berna, cells, research]
---

# Berna R8

**Bounded Forgetting for Cell-Based Language Models**

Built on Berna R7 v3.

## Fixes over R7 v3

1. K5 per-cell (freshness)
2. K6 per-cell (encompassment)
3. K2 per-cell (breadth)
4. Plexus threshold 0.5 to 0.1
5. save_all_dna for all cells
6. CellFreezer + freeze hooks

## H2 Experiment — Preliminary Results

Setup:
- Base: R7 v3 (step_1400)
- Fine-tune: 2000 steps on German
- Metric: F_eng = ENG_after - ENG_before

Results:

| Metric | Value |
|--------|-------|
| ENG_BEFORE | 4.0781 |
| ENG_AFTER  | 4.0703 |
| **F_eng**  | **0.000000** |
| DE_BEFORE  | 4.7406 |
| DE_AFTER   | 4.0797 |
| **Bounded** | **True** |

## Model Files

- `model/model.safetensors` — R8 weights
- `model/config.json`
- `model/generation_config.json`

## Quick Usage

    from transformers import AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        "BernaLabs/berna-r8/model",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

## Files

    code/k_vector/k5_percell.py
    code/k_vector/k6_percell.py
    code/k_vector/k2_breadth.py
    code/cells/save_all_dna.py
    code/cells/cell_manager.py
    code/cell_integration.py
    code/forgetting/bounded.py
    train_r8.py

## License

CC BY-NC 4.0 — research use only.

## Citation

@misc{berna_r8_2026,
  author = {Kamil, Muhamed},
  title  = {Berna R8},
  year   = {2026},
  note   = {ORCID: 0009-0002-5866-8815}
}

## Zenodo DOI

**R8 preprint:** [10.5281/zenodo.23310767](https://doi.org/10.5281/zenodo.23310767)
