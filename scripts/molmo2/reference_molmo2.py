# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers==4.57.1",
#     "torch",
#     "torchvision",
#     "accelerate",
#     "pillow",
#     "requests",
#     "num2words",
#     "einops",
#     "huggingface_hub",
# ]
# ///
"""Dump original ``allenai/Molmo2-8B`` reference outputs under a transformers version compatible
with the repo's remote code, then upload them for the branch-side ``verify_molmo2.py parity
--reference`` to consume.

The original ships as remote code written for transformers 4.57.1; it does NOT run under the
branch transformers (renamed internals: ROPE_INIT_FUNCTIONS["default"], create_causal_mask's
``input_embeds`` kwarg, ...). So we capture the ground-truth here (pinned 4.57.1) and compare the
port against it on the branch. We save the EXACT input tensors so the comparison isolates the model.
"""

from __future__ import annotations

import argparse
import re

import torch


MODEL_ID = "allenai/Molmo2-8B"
DEFAULT_IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
BLOCK_RE = re.compile(r"model\.transformer\.blocks\.(\d+)$")  # original text-decoder block names


def load_image(src: str):
    import requests
    from PIL import Image

    if src.startswith(("http://", "https://")):
        return Image.open(requests.get(src, stream=True, timeout=60).raw).convert("RGB")
    return Image.open(src).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--prompt", default="Point to the cat.")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--repo", default="danelcsb/molmo2-parity-ref", help="dataset repo to upload to")
    ap.add_argument("--out", default="/tmp/molmo2_reference.pt")
    args = ap.parse_args()

    torch_dtype = getattr(torch, args.dtype)
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[load] original (remote code) {args.model_id} dtype={args.dtype} attn={args.attn}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, trust_remote_code=True, dtype=torch_dtype, attn_implementation=args.attn
    ).to(args.device).eval()

    image = load_image(args.image)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": args.prompt}, {"type": "image", "image": image},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(args.device)

    # Capture each text-decoder block's output (residual stream after layer i).
    blocks: dict[int, torch.Tensor] = {}
    handles = []
    for name, module in model.named_modules():
        m = BLOCK_RE.search(name)
        if m:
            idx = int(m.group(1))

            def make_hook(i):
                def hook(_m, _inp, out):
                    t = out[0] if isinstance(out, (tuple, list)) else out
                    if torch.is_tensor(t):
                        blocks[i] = t.detach().float().cpu()
                return hook

            handles.append(module.register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    for h in handles:
        h.remove()

    num_layers = max(blocks) + 1
    block_hidden = torch.stack([blocks[i][0] for i in range(num_layers)])  # [L, seq, hidden]
    logits = out.logits.float().cpu()

    n = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    tokens = gen[:, n:].cpu()

    ref = {
        "model_id": args.model_id, "prompt": args.prompt, "image": args.image,
        "dtype": args.dtype, "attn": args.attn,
        "inputs": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in inputs.items()},
        "block_hidden": block_hidden,
        "logits_last": logits[0, -1],
        "logits_argmax": logits[0].argmax(-1),
        "generated_ids": tokens[0],
    }
    torch.save(ref, args.out)
    print(f"[saved] {args.out} | layers={num_layers} seq={block_hidden.shape[1]} "
          f"hidden={block_hidden.shape[2]} gen_tokens={tokens.shape[1]}")
    print("generated:", processor.batch_decode(tokens, skip_special_tokens=True)[0])

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=True)
    api.upload_file(path_or_fileobj=args.out, path_in_repo="reference.pt",
                    repo_id=args.repo, repo_type="dataset")
    print(f"[uploaded] {args.repo}/reference.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
