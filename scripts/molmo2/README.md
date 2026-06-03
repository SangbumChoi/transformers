# Molmo2 port verification & demo

`verify_molmo2.py` checks that the in-library **Molmo2** port reproduces the original
`allenai/Molmo2-8B` checkpoint, and runs a grounding demo with point overlays.

The original checkpoint ships as *remote code* with the **pre-refactor** parameter names
(`model.transformer.blocks`, `image_vit.transformer.resblocks`, `attention.wq`,
`image_projector.w1` …). The port refactors them (`model.language_model.blocks`,
`image_vit.encoder.layers`, `self_attn.q_proj`, `image_projector.gate_proj` …). The script
loads the **same weights** into both implementations (remapping the original state dict onto the
port) so any output difference is purely an implementation difference, not a weight mismatch.

## Subcommands

| command     | what it does                                                                                  |
|-------------|-----------------------------------------------------------------------------------------------|
| `selftest`  | string-only check that the rename map covers all 706 params (no torch, no weights, no GPU)    |
| `parity`    | identical inputs → both models; per-module max-abs diffs + logits diff + greedy-token match; saves `molmo2_parity.png` |
| `demo`      | runs the port, parses `<point .../>` grounding output, saves overlay PNG                       |
| `reference` | dumps the original model's logits/generation to JSON (cross-version fallback)                 |

## Quick local check (no model download)

```bash
python scripts/molmo2/verify_molmo2.py selftest
```

## Full parity / demo on Hugging Face Jobs

The model is 8B, so run on a GPU. The script is a self-contained
[uv script](https://docs.astral.sh/uv/guides/scripts/) (PEP 723) that installs `transformers`
from this branch automatically:

```bash
# parity: layer-by-layer + logits diff (float32 + eager = exact comparison)
hf jobs uv run --flavor a100-large --secrets HF_TOKEN \
    https://raw.githubusercontent.com/SangbumChoi/transformers/molmo2/scripts/molmo2/verify_molmo2.py \
    -- parity --max-new-tokens 20

# grounding demo with overlay
hf jobs uv run --flavor a100-large --secrets HF_TOKEN \
    https://raw.githubusercontent.com/SangbumChoi/transformers/molmo2/scripts/molmo2/verify_molmo2.py \
    -- demo --prompt "Point to the cat."
```

Add `--detach` to background a job and follow it with `hf jobs logs <job-id>`. Artifacts
(`molmo2_parity.png`, `molmo2_image_points.png`) are written under `--out-dir` (default
`molmo2_out/`); on Jobs, push them somewhere persistent or print/inspect via logs.

> **Note on the original's remote code.** `parity` imports the original via
> `trust_remote_code=True`. That code was written against an older transformers release; if it
> fails to import under this branch, use `reference` (pinned to a compatible transformers in its
> own job) to dump the original outputs, then compare the port against that JSON.

## Local GPU run

```bash
pip install -e .            # this branch
python scripts/molmo2/verify_molmo2.py parity --device cuda --dtype float32 --attn eager
```

`float32` + `eager` gives the tightest comparison (~1e-4 logits diff expected). Use
`--dtype bfloat16` if memory-constrained (looser tolerance).
