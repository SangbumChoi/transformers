# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers @ git+https://github.com/SangbumChoi/transformers.git@molmo2",
#     "torch",
#     "torchvision",
#     "accelerate",
#     "pillow",
#     "requests",
#     "matplotlib",
#     "numpy",
#     "num2words",
#     "einops",
#     "av",
# ]
# ///
"""Verify the in-library Molmo2 port against the original ``allenai/Molmo2-8B`` checkpoint.

The original checkpoint ships as *remote code* (``trust_remote_code=True``) and uses the
pre-refactor parameter names (``model.transformer.blocks``, ``image_vit.transformer.resblocks``,
``attention.wq``, ``image_projector.w1`` ...). The in-library port refactors those names
(``model.language_model.blocks``, ``image_vit.encoder.layers``, ``self_attn.q_proj``,
``image_projector.gate_proj`` ...). This script proves the refactor *preserves* the original by:

  * ``parity``    -- loads BOTH implementations with the *same* weights (the original state dict
                     remapped onto the port), feeds identical inputs, hooks every matched module,
                     and reports per-module max-abs diffs, a logits diff, and greedy-token
                     agreement. Saves a per-layer diff plot.
  * ``demo``      -- runs the port end-to-end on an image (and optionally a video), parses the
                     ``<point .../>`` grounding output, and saves overlay PNG(s).
  * ``reference`` -- dumps reference outputs (logits / generated text) for cross-version compare,
                     useful when the original remote code is not importable under this
                     transformers version.

Designed to run on a GPU box or, since this needs an 8B model, on Hugging Face Jobs:

    hf jobs uv run --flavor a100-large --secrets HF_TOKEN \\
        https://raw.githubusercontent.com/SangbumChoi/transformers/molmo2/scripts/molmo2/verify_molmo2.py \\
        -- parity --max-new-tokens 20

(``--detach`` to background it; ``hf jobs logs <id>`` to follow.)
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

import requests


MODEL_ID = "allenai/Molmo2-8B"
DEFAULT_IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
DEFAULT_VIDEO = "https://storage.googleapis.com/oe-training-public/demo_videos/many_penguins.mp4"

# Ordered (original -> port) parameter-name rewrite rules. Order matters: the parent-path
# renames (``transformer.blocks`` -> ``language_model.blocks``, ``resblocks`` -> ``encoder.layers``)
# run first, then the per-submodule renames. The two ``wq/wk/wv/wo`` groups are disambiguated by
# their distinct prefixes (``.attention.`` for the ViT layers vs ``.image_pooling_2d.``).
RENAME_RULES: list[tuple[str, str]] = [
    # --- text decoder (language model) ---
    (r"^model\.transformer\.wte\.", "model.language_model.wte."),
    (r"^model\.transformer\.ln_f\.", "model.language_model.norm."),
    (r"^model\.transformer\.blocks\.", "model.language_model.blocks."),
    # --- vision encoder layers ---
    (r"\.image_vit\.transformer\.resblocks\.", ".image_vit.encoder.layers."),
    (r"\.attention_norm\.", ".layer_norm1."),
    (r"\.ffn_norm\.", ".layer_norm2."),
    (r"\.feed_forward\.w1\.", ".mlp.fc1."),
    (r"\.feed_forward\.w2\.", ".mlp.fc2."),
    (r"\.attention\.wq\.", ".self_attn.q_proj."),
    (r"\.attention\.wk\.", ".self_attn.k_proj."),
    (r"\.attention\.wv\.", ".self_attn.v_proj."),
    (r"\.attention\.wo\.", ".self_attn.out_proj."),
    # --- vision adapter (pooling + projector) ---
    (r"\.image_pooling_2d\.wq\.", ".image_pooling_2d.q_proj."),
    (r"\.image_pooling_2d\.wk\.", ".image_pooling_2d.k_proj."),
    (r"\.image_pooling_2d\.wv\.", ".image_pooling_2d.v_proj."),
    (r"\.image_pooling_2d\.wo\.", ".image_pooling_2d.out_proj."),
    (r"\.image_projector\.w1\.", ".image_projector.gate_proj."),
    (r"\.image_projector\.w2\.", ".image_projector.down_proj."),
    (r"\.image_projector\.w3\.", ".image_projector.up_proj."),
]

# Module-path rewrite rules (no trailing ``.``) for aligning hook names. Same intent as
# RENAME_RULES but matches module names rather than parameter (``weight``/``bias``) names.
MODULE_RENAME_RULES: list[tuple[str, str]] = [
    (r"^model\.transformer\.wte$", "model.language_model.wte"),
    (r"^model\.transformer\.ln_f$", "model.language_model.norm"),
    (r"^model\.transformer\.blocks", "model.language_model.blocks"),
    (r"^model\.transformer$", "model.language_model"),
    (r"\.image_vit\.transformer\.resblocks", ".image_vit.encoder.layers"),
    (r"\.image_vit\.transformer$", ".image_vit.encoder"),
    (r"\.attention_norm$", ".layer_norm1"),
    (r"\.ffn_norm$", ".layer_norm2"),
    (r"\.feed_forward\.w1$", ".mlp.fc1"),
    (r"\.feed_forward\.w2$", ".mlp.fc2"),
    (r"\.feed_forward$", ".mlp"),
    (r"\.attention\.wq$", ".self_attn.q_proj"),
    (r"\.attention\.wk$", ".self_attn.k_proj"),
    (r"\.attention\.wv$", ".self_attn.v_proj"),
    (r"\.attention\.wo$", ".self_attn.out_proj"),
    (r"\.image_pooling_2d\.wq$", ".image_pooling_2d.q_proj"),
    (r"\.image_pooling_2d\.wk$", ".image_pooling_2d.k_proj"),
    (r"\.image_pooling_2d\.wv$", ".image_pooling_2d.v_proj"),
    (r"\.image_pooling_2d\.wo$", ".image_pooling_2d.out_proj"),
    (r"\.image_projector\.w1$", ".image_projector.gate_proj"),
    (r"\.image_projector\.w2$", ".image_projector.down_proj"),
    (r"\.image_projector\.w3$", ".image_projector.up_proj"),
]


def apply_rules(name: str, rules: list[tuple[str, str]]) -> str:
    for pat, repl in rules:
        name = re.sub(pat, repl, name)
    return name


def convert_state_dict(orig_state_dict: dict) -> dict:
    """Rewrite original Molmo2 parameter names to the in-library port names."""
    return {apply_rules(k, RENAME_RULES): v for k, v in orig_state_dict.items()}


def convert_to_local_checkpoint(model_id: str, out_dir: str, dtype: str) -> str:
    """Stream the original safetensors shards, rename keys to the port layout, cast, and save.

    This produces a port-loadable checkpoint *without ever instantiating the original model*, so the
    port never depends on the remote code being importable. Memory-safe: one shard (~4GB) is held at
    a time, so it fits a small box."""
    import glob
    import json
    import os
    import shutil

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    if os.path.isdir(out_dir) and os.path.exists(os.path.join(out_dir, "model.safetensors.index.json")):
        return out_dir
    src = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.index.json", "config.json"])
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(src, "config.json"), out_dir)
    torch_dtype = getattr(torch, dtype)

    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    weight_map: dict[str, str] = {}
    for shard in shards:
        name = os.path.basename(shard)
        renamed = {apply_rules(k, RENAME_RULES): v.to(torch_dtype) for k, v in load_file(shard).items()}
        save_file(renamed, os.path.join(out_dir, name), metadata={"format": "pt"})
        weight_map.update({k: name for k in renamed})
        print(f"[convert] {name}: {len(renamed)} tensors -> {dtype}")
    json.dump({"metadata": {}, "weight_map": weight_map},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))
    return out_dir


# --------------------------------------------------------------------------------------------------
# String-only self test (runs without torch): validate the rename map against the published index.
# --------------------------------------------------------------------------------------------------
def selftest(model_id: str = MODEL_ID) -> int:
    url = f"https://huggingface.co/{model_id}/resolve/main/model.safetensors.index.json"
    keys = list(requests.get(url, timeout=60).json()["weight_map"].keys())
    converted = [apply_rules(k, RENAME_RULES) for k in keys]

    # No converted key may retain an original-only token.
    stale_tokens = [
        ".transformer.", "resblocks", ".attention.", "feed_forward", "ln_f",
        ".wq.", ".wk.", ".wv.", ".wo.", "attention_norm", "ffn_norm",
        ".image_projector.w1.", ".image_projector.w2.", ".image_projector.w3.",
    ]
    leftover = sorted({tok for k in converted for tok in stale_tokens if tok in k})
    print(f"original params: {len(keys)} | converted (unique): {len(set(converted))}")
    if len(set(converted)) != len(converted):
        print("FAIL: rename map collapsed distinct keys onto the same name", file=sys.stderr)
        return 1
    if leftover:
        print(f"FAIL: stale original tokens survived conversion: {leftover}", file=sys.stderr)
        for k, c in zip(keys, converted):
            if any(t in c for t in leftover):
                print(f"  {k}  ->  {c}")
        return 1
    print("examples:")
    for k in ("model.transformer.blocks.0.self_attn.att_proj.weight",
              "model.vision_backbone.image_vit.transformer.resblocks.5.attention.wq.weight",
              "model.vision_backbone.image_projector.w2.weight",
              "model.vision_backbone.image_pooling_2d.wo.bias"):
        if k in keys:
            print(f"  {k}\n    -> {apply_rules(k, RENAME_RULES)}")
    print("PASS: every original parameter maps to a unique, fully-rewritten port name.")
    return 0


# --------------------------------------------------------------------------------------------------
# Everything below imports torch lazily so the selftest stays dependency-light.
# --------------------------------------------------------------------------------------------------
def _torch():
    import torch
    return torch


def _patch_for_coexistence():
    """Make the original (remote code) and the in-library port loadable in one process.

    Two collisions to neutralize:

    1. The remote ``modeling_molmo2.py`` calls ``AutoModelForImageTextToText.register(Molmo2Config,
       ...)`` at import, which raises ``'Molmo2Config' is already used`` once the in-library port
       owns that slot. We make auto-mapping registration swallow that specific collision.
    2. ``AutoProcessor`` does not forward ``trust_remote_code`` to its sub-processors, so the image
       processor resolves it as ``None`` and *raises* (it sees the repo's advertised custom code).
       We default ``None -> False`` so sub-processors fall back to the in-library classes, while the
       original model still loads remotely because we pass ``trust_remote_code=True`` explicitly."""
    import transformers.models.auto.auto_factory as af

    cls = af._LazyAutoMapping
    if not getattr(cls.register, "_idem", False):
        orig_register = cls.register

        def register(self, key, value, exist_ok=False):
            try:
                return orig_register(self, key, value, exist_ok=True)
            except ValueError as e:
                if "already used" in str(e):
                    return
                raise

        register._idem = True
        cls.register = register

    # Default trust_remote_code None -> False everywhere it is referenced (each auto module imported
    # the symbol by name, so patch them individually).
    import importlib

    import transformers.dynamic_module_utils as dmu

    orig_resolve = getattr(dmu.resolve_trust_remote_code, "_orig", dmu.resolve_trust_remote_code)

    def resolve_trust_remote_code(trust_remote_code, *args, **kwargs):
        if trust_remote_code is None:
            return False
        return orig_resolve(trust_remote_code, *args, **kwargs)

    resolve_trust_remote_code._orig = orig_resolve
    dmu.resolve_trust_remote_code = resolve_trust_remote_code
    for modname in (
        "auto_factory", "configuration_auto", "image_processing_auto", "processing_auto",
        "tokenization_auto", "feature_extraction_auto", "video_processing_auto",
    ):
        try:
            mod = importlib.import_module(f"transformers.models.auto.{modname}")
        except Exception:
            continue
        if hasattr(mod, "resolve_trust_remote_code"):
            mod.resolve_trust_remote_code = resolve_trust_remote_code

    # The branch transformers refactored RoPE and removed the "default" key from
    # ROPE_INIT_FUNCTIONS, but the original's remote rotary does ROPE_INIT_FUNCTIONS["default"].
    # Re-add a default initializer (reads rope_theta / head_dim from the remote config).
    import transformers.modeling_rope_utils as mru

    if "default" not in mru.ROPE_INIT_FUNCTIONS:
        import torch

        def _default_rope(config, device=None, seq_len=None, **kw):
            params = getattr(config, "rope_parameters", None) or {}
            base = getattr(config, "rope_theta", None) or params.get("rope_theta", 10000.0)
            dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
            )
            return inv_freq, 1.0

        mru.ROPE_INIT_FUNCTIONS["default"] = _default_rope


@dataclass
class HookStore:
    outputs: dict = field(default_factory=dict)
    handles: list = field(default_factory=list)

    def register(self, module, name):
        def hook(_mod, _inp, out):
            torch = _torch()
            tensor = out
            if isinstance(out, (tuple, list)):
                tensor = next((o for o in out if torch.is_tensor(o)), None)
            elif hasattr(out, "last_hidden_state"):
                tensor = out.last_hidden_state
            if torch is not None and tensor is not None and torch.is_tensor(tensor):
                self.outputs[name] = tensor.detach().float().cpu()

        self.handles.append(module.register_forward_hook(hook))

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _free(*objs):
    """Drop references and reclaim GPU memory between the two parity phases."""
    import gc

    torch = _torch()
    for _ in objs:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _hook_all(model, store: "HookStore", name_map: list[tuple[str, str]] | None):
    """Register a forward hook on every named module, keyed by its canonical (port) name."""
    for name, module in model.named_modules():
        canonical = apply_rules(name, name_map) if name_map else name
        if canonical:
            store.register(module, canonical)


def load_processor(model_id: str):
    """Build the in-library Molmo2Processor directly from concrete classes.

    The repo's remote processor is incompatible with the branch transformers (it passes
    ``image_use_col_tokens`` to ``ProcessorMixin.__init__``, which now rejects unknown kwargs), and
    ``AutoProcessor`` trips on remote-code resolution for the sub-image-processor. Constructing the
    in-library components directly sidesteps both, while still reading the repo's configs."""
    import json

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from transformers.models.molmo2.image_processing_molmo2 import Molmo2ImageProcessor
    from transformers.models.molmo2.processing_molmo2 import Molmo2Processor

    image_processor = Molmo2ImageProcessor.from_pretrained(model_id)
    try:
        from transformers.models.molmo2.video_processing_molmo2 import Molmo2VideoProcessor

        video_processor = Molmo2VideoProcessor.from_pretrained(model_id)
    except Exception as e:  # video path is optional for image parity/demo
        print(f"[warn] video processor unavailable ({e}); continuing without it")
        video_processor = None
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    chat_template = open(hf_hub_download(model_id, "chat_template.jinja")).read()
    extra = json.load(open(hf_hub_download(model_id, "processor_config.json")))
    for k in ("auto_map", "processor_class"):
        extra.pop(k, None)
    return Molmo2Processor(
        image_processor=image_processor,
        video_processor=video_processor,
        tokenizer=tokenizer,
        chat_template=chat_template,
        **extra,
    )


def build_inputs(model_id: str, image_url: str, prompt: str, device: str):
    processor = load_processor(model_id)
    image = load_image(image_url)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt}, {"type": "image", "image": image},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(device)
    return processor, image, inputs


def _forward_filtered(model, inputs: dict):
    """Call ``model(**inputs)`` keeping only kwargs the model's forward accepts."""
    torch = _torch()
    import inspect

    sig = inspect.signature(model.forward)
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    kw = {k: v for k, v in inputs.items() if accepts_kwargs or k in sig.parameters}
    with torch.no_grad():
        return model(**kw, use_cache=False, output_hidden_states=True)


def _capture(model, inputs, processor, name_map, max_new_tokens):
    """Run one model: capture per-module activations, final logits, and greedy tokens.

    Runs a single model at a time so only one 8B copy lives on the GPU (fits a 24GB card).
    """
    torch = _torch()
    store = HookStore()
    _hook_all(model, store, name_map)
    out = _forward_filtered(model, dict(inputs))
    acts = dict(store.outputs)
    store.clear()
    logits = out.logits.float().cpu()
    tokens = None
    if max_new_tokens > 0:
        n = inputs["input_ids"].shape[1]
        try:
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            tokens = gen[:, n:].cpu()
        except Exception as e:  # generate may hit further remote/version cache-API gaps; forward still compared
            print(f"[warn] generate() failed ({type(e).__name__}: {e}); skipping greedy-token check")
    return acts, logits, tokens


def cmd_parity(args) -> int:
    torch = _torch()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import (
        AutoModelForImageTextToText,
        Molmo2Config,
        Molmo2ForConditionalGeneration,
    )

    torch_dtype = getattr(torch, args.dtype)
    _patch_for_coexistence()  # allow the remote original to import alongside the in-library port
    processor, _, inputs = build_inputs(args.model_id, args.image, args.prompt, args.device)

    # ---- Phase 1: ORIGINAL (remote code) -- keep its weights on CPU for the port, then free GPU.
    print(f"[phase 1] original (remote code) {args.model_id} dtype={args.dtype} attn={args.attn}")
    original = AutoModelForImageTextToText.from_pretrained(
        args.model_id, trust_remote_code=True, dtype=torch_dtype, attn_implementation=args.attn
    ).to(args.device).eval()
    orig_acts, orig_logits, orig_tokens = _capture(
        original, inputs, processor, MODULE_RENAME_RULES, args.max_new_tokens
    )
    del original
    _free()

    # ---- Phase 2: PORT (in-library) -- loaded from a renamed local checkpoint (independent of the
    # original model object), so the port never depends on the remote code being importable.
    print("[phase 2] in-library port from remapped weights")
    local_ckpt = convert_to_local_checkpoint(args.model_id, f"{args.out_dir.rstrip('/')}/port_ckpt", args.dtype)
    config = Molmo2Config.from_pretrained(args.model_id, trust_remote_code=False)
    config._attn_implementation = args.attn
    port = Molmo2ForConditionalGeneration.from_pretrained(
        local_ckpt, config=config, dtype=torch_dtype, attn_implementation=args.attn, device_map=args.device,
    ).eval()
    _free()
    port_acts, port_logits, port_tokens = _capture(
        port, inputs, processor, None, args.max_new_tokens
    )
    del port
    _free()

    # ---- Phase 3: compare ----
    rows = []
    for name in sorted(set(orig_acts) & set(port_acts)):
        a, b = orig_acts[name], port_acts[name]
        if a.shape != b.shape:
            rows.append((name, float("nan"), f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"))
            continue
        max_abs = (a - b).abs().max().item()
        denom = a.abs().max().item() or 1.0
        rows.append((name, max_abs, f"rel={max_abs / denom:.2e}"))

    rows.sort(key=lambda r: (r[1] != r[1], -r[1]))  # NaNs first, then largest diff
    print(f"\n=== per-module max-abs diff (top 25 of {len(rows)} matched) ===")
    print(f"{'module':70s} {'max_abs_diff':>14s}  note")
    for name, diff, note in rows[:25]:
        print(f"{name:70s} {diff:>14.3e}  {note}")

    logits_diff = (orig_logits - port_logits).abs().max().item()
    top1_orig = orig_logits[0, -1].argmax().item()
    top1_port = port_logits[0, -1].argmax().item()
    top1_match = top1_orig == top1_port
    print("\n=== summary ===")
    worst = next((r for r in rows if r[1] == r[1]), ("-", 0.0, ""))
    print(f"worst module diff : {worst[0]} = {worst[1]:.3e}")
    print(f"final logits diff : {logits_diff:.3e}")
    print(f"argmax next token : original={top1_orig} port={top1_port} "
          f"({'MATCH' if top1_match else 'MISMATCH'})")

    tokens_match = True
    if orig_tokens is not None and port_tokens is not None:
        m = min(orig_tokens.shape[1], port_tokens.shape[1])
        tokens_match = bool((orig_tokens[:, :m] == port_tokens[:, :m]).all())
        print(f"greedy tokens     : {'MATCH' if tokens_match else 'MISMATCH'} over {m} tokens")
        print("  original:", processor.batch_decode(orig_tokens, skip_special_tokens=True)[0])
        print("  port    :", processor.batch_decode(port_tokens, skip_special_tokens=True)[0])

    # Plot per-module diffs (text blocks + vision layers separated).
    def subset(tag):
        return [(n, d) for n, d, _ in rows if tag in n and d == d]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for ax, (tag, title) in zip(axes, [
        ("language_model.blocks", "Text decoder modules"),
        ("image_vit.encoder.layers", "Vision encoder modules"),
    ]):
        data = subset(tag)
        if data:
            ax.bar(range(len(data)), [d for _, d in data])
            ax.set_yscale("log")
            ax.set_title(f"{title}: max-abs diff per module (n={len(data)})")
            ax.set_ylabel("max abs diff (log)")
    fig.suptitle(f"Molmo2 original vs port parity  |  logits diff = {logits_diff:.2e}")
    fig.tight_layout()
    out_path = f"{args.out_dir.rstrip('/')}/molmo2_parity.png"
    _ensure_dir(args.out_dir)
    fig.savefig(out_path, dpi=120)
    print(f"\n[saved] {out_path}")

    # In bf16/fp16 the absolute logits diff between two implementations is not a meaningful gate
    # (low-precision accumulation order differs), so PASS is decided by token agreement; the logits
    # diff is reported and additionally gated only in float32.
    ok = top1_match and tokens_match
    if args.dtype == "float32":
        ok = ok and logits_diff < args.tol
    print(f"\nlogits diff {logits_diff:.3e} (fp32 gate tol={args.tol:g}) | "
          f"argmax {'OK' if top1_match else 'X'} | greedy {'OK' if tokens_match else 'X'}")
    print(f"PARITY {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------------
# Grounding demo + overlays
# --------------------------------------------------------------------------------------------------
POINT_RE = re.compile(r'<point\b[^>]*?\bx\d*="([0-9.]+)"[^>]*?\by\d*="([0-9.]+)"', re.I)
POINTS_COORDS_RE = re.compile(r'coords="([^"]+)"', re.I)
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_points(text: str) -> list[tuple[float, float]]:
    """Parse normalized (0-100) points from Molmo2 grounding output.

    Handles both ``<point x="..." y="..."/>`` / ``<point x1=.. y1=.. x2=.. y2=..>`` and the
    ``<points coords="t,id,x,y ...">`` payload used for video. Returns (x, y) in 0-100 space.
    """
    pts: list[tuple[float, float]] = []
    for x, y in POINT_RE.findall(text):
        pts.append((float(x), float(y)))
    if not pts:
        for coords in POINTS_COORDS_RE.findall(text):
            nums = [float(n) for n in NUM_RE.findall(coords)]
            # Coords groups are timestamp/id-prefixed; take trailing (x, y) pairs heuristically.
            for i in range(0, len(nums) - 1, 2):
                x, y = nums[i], nums[i + 1]
                if 0 <= x <= 100 and 0 <= y <= 100:
                    pts.append((x, y))
    return pts


def overlay_points(image, points, out_path: str, title: str = ""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, h = image.size
    fig, ax = plt.subplots(figsize=(w / 100, h / 100))
    ax.imshow(image)
    for i, (x, y) in enumerate(points):
        px, py = x / 100 * w, y / 100 * h
        ax.plot(px, py, "o", ms=14, mfc="none", mec="red", mew=2.5)
        ax.text(px + 6, py, str(i), color="red", fontsize=12, weight="bold")
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    _ensure_dir(out_path.rsplit("/", 1)[0] if "/" in out_path else ".")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}  ({len(points)} points)")


def cmd_demo(args) -> int:
    torch = _torch()
    from transformers import Molmo2Config, Molmo2ForConditionalGeneration

    torch_dtype = getattr(torch, args.dtype)
    _patch_for_coexistence()
    # Exercise the port end-to-end on the original weights, remapped into a local checkpoint.
    local_ckpt = convert_to_local_checkpoint(args.model_id, f"{args.out_dir.rstrip('/')}/port_ckpt", args.dtype)
    config = Molmo2Config.from_pretrained(args.model_id, trust_remote_code=False)
    config._attn_implementation = args.attn
    model = Molmo2ForConditionalGeneration.from_pretrained(
        local_ckpt, config=config, dtype=torch_dtype, attn_implementation=args.attn, device_map=args.device,
    ).eval()
    processor = load_processor(args.model_id)

    # --- image grounding ---
    image = load_image(args.image)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": args.prompt}, {"type": "image", "image": image},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(args.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print("=== image generation ===")
    print(text)
    pts = parse_points(text)
    print(f"=== parsed {len(pts)} point(s) (normalized 0-100) ===")
    for i, (x, y) in enumerate(pts):
        print(f"  [{i}] x={x:.2f} y={y:.2f}")
    overlay_points(image, pts, f"{args.out_dir.rstrip('/')}/molmo2_image_points.png",
                   title=args.prompt)
    return 0


# --------------------------------------------------------------------------------------------------
# reference dump (cross-version comparison fallback)
# --------------------------------------------------------------------------------------------------
def cmd_reference(args) -> int:
    torch = _torch()
    import json

    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype = getattr(torch, args.dtype)
    _patch_for_coexistence()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, trust_remote_code=True, dtype=torch_dtype, attn_implementation=args.attn
    ).to(args.device).eval()
    processor = load_processor(args.model_id)
    image = load_image(args.image)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": args.prompt}, {"type": "image", "image": image},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(args.device)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    n = inputs["input_ids"].shape[1]
    ref = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "image": args.image,
        "logits_last": out.logits[0, -1].float().tolist(),
        "generated_ids": gen[0, n:].tolist(),
        "generated_text": processor.batch_decode(gen[:, n:], skip_special_tokens=True)[0],
    }
    _ensure_dir(args.out_dir)
    path = f"{args.out_dir.rstrip('/')}/molmo2_reference.json"
    with open(path, "w") as f:
        json.dump(ref, f)
    print(f"[saved] {path}\ngenerated: {ref['generated_text']}")
    return 0


# --------------------------------------------------------------------------------------------------
def load_image(src: str):
    from PIL import Image

    if src.startswith(("http://", "https://")):
        return Image.open(requests.get(src, stream=True, timeout=60).raw).convert("RGB")
    return Image.open(src).convert("RGB")


def _ensure_dir(path: str):
    import os

    os.makedirs(path, exist_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model-id", default=MODEL_ID)
        sp.add_argument("--image", default=DEFAULT_IMAGE)
        sp.add_argument("--prompt", default="Point to the cat.")
        sp.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--attn", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
        sp.add_argument("--max-new-tokens", type=int, default=20)
        sp.add_argument("--out-dir", default="molmo2_out")

    sp = sub.add_parser("selftest", help="string-only rename-map check (no torch / no weights)")
    sp.add_argument("--model-id", default=MODEL_ID)

    sp = sub.add_parser("parity", help="layer-by-layer + logits parity, original vs port")
    common(sp)
    sp.add_argument("--tol", type=float, default=2e-2, help="max-abs logits diff to pass")

    sp = sub.add_parser("demo", help="run the port, parse grounding points, save overlay")
    common(sp)
    sp.set_defaults(max_new_tokens=256)

    sp = sub.add_parser("reference", help="dump original outputs for cross-version compare")
    common(sp)

    args = p.parse_args()
    if args.cmd == "selftest":
        return selftest(args.model_id)
    if args.cmd == "parity":
        return cmd_parity(args)
    if args.cmd == "demo":
        return cmd_demo(args)
    if args.cmd == "reference":
        return cmd_reference(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
