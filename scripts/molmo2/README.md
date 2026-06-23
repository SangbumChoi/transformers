# Molmo2 port verification & demo

Tools to check that the in-library **Molmo2** port reproduces the original `allenai/Molmo2-8B`
checkpoint, plus a grounding demo with point overlays.

- `verify_molmo2.py` — runs the **port** (from this branch): grounding demo, and `parity` that
  compares the port against a saved reference of the original.
- `reference_molmo2.py` — runs the **original** under a pinned, compatible transformers and dumps a
  reference (per-layer hidden states, vision-stage tensors, RoPE cos/sin, logits, generated tokens,
  and the exact input tensors), uploading it to a HF dataset repo.

## Why two scripts (cross-version comparison)

The original ships as *remote code* written for an older transformers release (`transformers
4.57.1`). It does **not** import/run under this branch (renamed internals: `ROPE_INIT_FUNCTIONS`,
`create_causal_mask` kwargs, …). So we capture the original's ground-truth in its native version
(`reference_molmo2.py`, pinned via PEP 723) and compare the port against it on this branch. The
reference stores the **exact input tensors**, so the comparison isolates the model, not preprocessing.

## Workflow (Hugging Face Jobs; the model is 8B, needs a GPU)

```bash
# 1) Dump the original's reference (pinned transformers==4.57.1) and upload to a dataset repo.
hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN \
    scripts/molmo2/reference_molmo2.py \
    -- --dtype float32 --attn sdpa --repo <your-username>/molmo2-parity-ref

# 2) Compare the port (this branch) against that reference, with identical inputs.
hf jobs uv run --flavor l40sx1 --secrets HF_TOKEN \
    scripts/molmo2/verify_molmo2.py \
    -- parity --reference <your-username>/molmo2-parity-ref --dtype float32 --attn sdpa

# Grounding demo with overlay
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
    scripts/molmo2/verify_molmo2.py -- demo --prompt "Point to the cat."
```

`-d`/`--detach` backgrounds a job; follow it with `hf jobs logs <job-id>`. `float32 + sdpa` is the
canonical path and gives a bit-exact comparison. Use `bfloat16` if memory-constrained (looser, and
note bf16 amplifies the model's massive-activation channels on a few dims).

## What `parity --reference` reports

Given identical inputs it loads the port, feeds the saved tensors, and prints:

- per-text-layer hidden-state diffs (max = outlier channels, mean = bulk),
- **vision pipeline stage split** (`ViT→pool` input → `pooled` → projected) to localize vision diffs,
- **layer-0 text-decoder submodule split** (`attn_norm` / `q_norm` / `k_norm` / `self_attn` /
  `ff_norm` / `mlp`) and a RoPE cos/sin check to localize text diffs,
- `inputs_embeds` diff split by image vs text token positions,
- final logits diff, next-token argmax, and greedy-token agreement; saves `molmo2_parity.png`.

## Subcommands

| command     | what it does                                                                                       |
|-------------|----------------------------------------------------------------------------------------------------|
| `selftest`  | string-only check that the rename map covers all params (no torch, no weights, no GPU)              |
| `parity`    | `--reference <repo|path>`: compare the port against a saved original reference; stage/submodule splits + plot |
| `demo`      | runs the port, parses `<point .../>` grounding output, saves an overlay PNG                         |

```bash
python scripts/molmo2/verify_molmo2.py selftest   # quick, no download
```

## Note on inputs: `mm_token_type_ids`

The original's processor emits `token_type_ids`; the port's forward expects **`mm_token_type_ids`**
(same tensor, renamed). `parity --reference` aliases it automatically. If you build inputs yourself,
pass `mm_token_type_ids` — otherwise the port silently builds a plain-causal mask instead of the
image↔image bidirectional mask, which looks like a large text-decoder divergence but is an input bug.

## Findings (this branch)

- In **float32 + SDPA** with inputs fed correctly, the port reproduces the original **bit-exactly**
  across the whole model: vision backbone, pooling, projection, all text-decoder layers, and final
  logits all show `0.000` difference; greedy generation matches.
- The only model-side nit: the pooling attention's **eager** path adds a boolean mask to the logits
  instead of excluding invalid patches with `-inf`, so eager differs from SDPA for partial pooling
  groups at image edges. SDPA (the default) is correct, and the original has the same eager/SDPA
  inconsistency, so this is low priority.
