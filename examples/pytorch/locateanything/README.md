# LocateAnything native LoRA training smoke test

This example verifies the standard autoregressive supervised fine-tuning path of
`LocateAnythingForConditionalGeneration` in Transformers. It uses assistant-only labels and trains LoRA adapters on
the language model while leaving the base model, vision tower, and multimodal projector frozen.

The default example is the industrial warehouse case from the LocateAnything integration test. It contains a
multi-category prompt and supervision for a forklift, pallets, stacked supply boxes, and the warehouse aisle.

## Scope

This is a one-example plumbing test, not a production training recipe or a reproduction of LocateAnything training.
The Transformers integration implements standard next-token prediction (NTP). NVIDIA's complete continual-SFT stack
adds streaming packing, DeepSpeed, MagiAttention, and Parallel Box Decoding (PBD/MTP); use the
[official Eagle training guide](https://github.com/NVlabs/Eagle/blob/main/Embodied/document/TRAINING.md) for that
workflow.

The released weights use the NVIDIA non-commercial license. Confirm the model license and the rights for every image
and annotation before fine-tuning or redistributing an adapter.

## Colab or GPU VM

Use an A10G, L4, A100, H100, or newer bfloat16-capable GPU. A T4 is not supported by this smoke test and 16 GB is
generally insufficient for the unquantized model plus training activations.

```python
%pip install -q \
  "git+https://github.com/SangbumChoi/transformers.git@38b9bd8d87cd52ffca787490de353004a71de1c1" \
  "peft>=0.17.0" requests pillow

!wget -q \
  https://raw.githubusercontent.com/SangbumChoi/transformers/codex/locateanything-training-demo/examples/pytorch/locateanything/locateanything_lora_train.py
```

Validate preprocessing and label masking without loading the 3B model:

```python
!python locateanything_lora_train.py --dry-run
```

Run one optimizer step and save the LoRA adapter:

```python
!python locateanything_lora_train.py \
  --steps 1 \
  --output-dir locateanything-warehouse-lora
```

The demo sets `--in-token-limit 8192`, which downscales the high-resolution warehouse image to roughly 2K merged
vision tokens. This keeps the SDPA training sequence practical without changing normalized `[0, 1000]` coordinates.

For real adaptation, replace `--image-url`, `--question`, and `--answer` with licensed examples and use a dataset
loader rather than repeating this single sample. Coordinates in the answer are quantized integers in `[0, 1000]`.
The structured target should use the form:

```text
<ref>label</ref><box><x1><y1><x2><y2></box>
```
