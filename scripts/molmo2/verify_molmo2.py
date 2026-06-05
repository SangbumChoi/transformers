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

    # The branch renamed the masking helpers' ``input_embeds`` kwarg to ``inputs_embeds``; the remote
    # code still passes ``input_embeds``. Shim the module-level names *before* the remote module's
    # ``from ...masking_utils import create_causal_mask`` binds them, translating the old kwarg. The
    # in-library port already imported the real functions at startup, so it is unaffected.
    import inspect

    import transformers.masking_utils as mu

    for _fname in ("create_causal_mask", "create_masks_for_generate", "create_sliding_window_causal_mask"):
        _orig = getattr(mu, _fname, None)
        if _orig is None or getattr(_orig, "_kwshim", False):
            continue

        def _make_shim(orig):
            sig = inspect.signature(orig)
            accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            allowed = set(sig.parameters)

            def shim(*a, **k):
                # Old remote code used input_embeds + cache_position; the new signature renamed the
                # former to inputs_embeds and derives positions internally. Translate, then drop any
                # kwarg the current signature no longer accepts.
                if "input_embeds" in k and "inputs_embeds" not in k:
                    k["inputs_embeds"] = k.pop("input_embeds")
                if not accepts_var_kw:
                    k = {key: v for key, v in k.items() if key in allowed}
                return orig(*a, **k)

            shim._kwshim = True
            return shim

        setattr(mu, _fname, _make_shim(_orig))


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


BLOCK_INDEX_RE = re.compile(r"model\.language_model\.blocks\.(\d+)$")  # port text-decoder block names


def _load_reference(ref: str):
    """Load a reference dump produced by reference_molmo2.py (local path or HF dataset repo id)."""
    torch = _torch()
    import os

    if os.path.exists(ref):
        path = ref
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=ref, filename="reference.pt", repo_type="dataset")
    return torch.load(path, map_location="cpu", weights_only=False)


def _capture_block_hidden(model, inputs):
    """Forward the port and capture each text-decoder block output (stacked to [L, seq, hidden])
    plus the input to block 0 (== inputs_embeds, after image features are merged in)."""
    torch = _torch()

    blocks: dict[int, "torch.Tensor"] = {}
    embeds: dict[str, "torch.Tensor"] = {}
    handles = []
    for name, module in model.named_modules():
        m = BLOCK_INDEX_RE.search(name)
        if m:
            idx = int(m.group(1))

            def make_hook(i):
                def hook(_m, _inp, out):
                    t = out[0] if isinstance(out, (tuple, list)) else out
                    if torch.is_tensor(t):
                        blocks[i] = t.detach().float().cpu()
                return hook

            handles.append(module.register_forward_hook(make_hook(idx)))
            if idx == 0:
                def pre_hook(_m, args, kwargs):
                    t = args[0] if args else kwargs.get("hidden_states")
                    if torch.is_tensor(t):
                        embeds["x"] = t.detach().float().cpu()
                handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))

    # Vision-pipeline stages, mirroring the reference dump (to_pool -> pooled -> projected).
    vis: dict[str, "torch.Tensor"] = {}

    def cap_pre(key):
        def h(_m, args, kwargs):
            t = args[1] if len(args) > 1 else kwargs.get("keys", kwargs.get("to_pool"))
            if torch.is_tensor(t):
                vis[key] = t.detach().float().cpu()
        return h

    def cap_out(key):
        def h(_m, _i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if torch.is_tensor(t):
                vis[key] = t.detach().float().cpu()
        return h

    text_probes = {
        "L0.attn_norm": ".blocks.0.attn_norm",
        "L0.q_norm": ".blocks.0.self_attn.q_norm",
        "L0.k_norm": ".blocks.0.self_attn.k_norm",
        "L0.attn_out": ".blocks.0.self_attn",
        "L0.ff_norm": ".blocks.0.ff_norm",
        "L0.mlp": ".blocks.0.mlp",
    }

    def cap_txt(key):
        def h(_m, _i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if torch.is_tensor(t):
                vis[key] = t.detach().float().cpu()
        return h

    for name, module in model.named_modules():
        if name.endswith(".image_pooling_2d"):
            handles.append(module.register_forward_pre_hook(cap_pre("vit_to_pool"), with_kwargs=True))
            handles.append(module.register_forward_hook(cap_out("pooled")))
        elif name.endswith(".image_projector"):
            handles.append(module.register_forward_hook(cap_out("proj_out")))
        for key, suffix in text_probes.items():
            if name.endswith(suffix):
                handles.append(module.register_forward_hook(cap_txt(key)))

    def cap_rope(_m, _i, o):
        if isinstance(o, (tuple, list)) and len(o) >= 2:
            vis["rope_cos"] = o[0].detach().float().cpu()
            vis["rope_sin"] = o[1].detach().float().cpu()

    for name, module in model.named_modules():
        if name.endswith("rotary_embs.default") or name.endswith(".rotary_emb"):
            handles.append(module.register_forward_hook(cap_rope))

    out = _forward_filtered(model, dict(inputs))
    for h in handles:
        h.remove()
    num_layers = max(blocks) + 1
    block_hidden = torch.stack([blocks[i][0] for i in range(num_layers)])
    inputs_embeds = embeds["x"][0] if "x" in embeds else None
    return block_hidden, out.logits.float().cpu(), inputs_embeds, vis


def cmd_parity_reference(args) -> int:
    """Compare the in-library port against a saved reference dump of the original (cross-version)."""
    torch = _torch()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import Molmo2Config, Molmo2ForConditionalGeneration

    print(f"[ref] loading reference: {args.reference}")
    ref = _load_reference(args.reference)
    inputs = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in ref["inputs"].items()}
    # The original's processor emits `token_type_ids`; the port's forward expects `mm_token_type_ids`
    # (same tensor, renamed). Without this alias the port silently builds a plain-causal mask while
    # the original used image<->image bidirectional attention -> spurious text-decoder divergence.
    if "token_type_ids" in inputs and "mm_token_type_ids" not in inputs:
        inputs["mm_token_type_ids"] = inputs["token_type_ids"]
        print("[harness] aliased token_type_ids -> mm_token_type_ids for the port")
    print(f"[ref] original dtype={ref.get('dtype')} attn={ref.get('attn')} prompt={ref.get('prompt')!r} "
          f"layers={ref['block_hidden'].shape[0]} seq={ref['block_hidden'].shape[1]}")

    torch_dtype = getattr(torch, args.dtype)
    _patch_for_coexistence()
    local_ckpt = convert_to_local_checkpoint(args.model_id, "/tmp/molmo2_port_ckpt", args.dtype)
    config = Molmo2Config.from_pretrained(args.model_id, trust_remote_code=False)
    config._attn_implementation = args.attn
    port = Molmo2ForConditionalGeneration.from_pretrained(
        local_ckpt, config=config, dtype=torch_dtype, attn_implementation=args.attn, device_map=args.device,
    ).eval()

    port_hidden, port_logits, port_embeds, port_vis = _capture_block_hidden(port, inputs)
    ref_hidden = ref["block_hidden"]
    n_layers = min(ref_hidden.shape[0], port_hidden.shape[0])

    # ---- Vision-pipeline stage split: ViT features -> pooled -> projected. Pinpoints which stage
    # first introduces the image-feature divergence (pixel_values are identical across both runs).
    print("\n=== vision pipeline stage diffs (original ref vs port) ===")
    print(f"{'stage':>12s} {'shape':>22s} {'max_abs':>12s} {'mean_abs':>12s} {'mean_rel':>10s}")
    for key, label in [("vit_to_pool", "ViT->pool in"), ("pooled", "pooled"), ("proj_out", "projected")]:
        a, b = ref.get(key), port_vis.get(key)
        if a is None or b is None:
            print(f"{label:>12s} {'(missing)':>22s}")
            continue
        if tuple(a.shape) != tuple(b.shape):
            print(f"{label:>12s} {f'ref{tuple(a.shape)} port{tuple(b.shape)}':>22s}  SHAPE MISMATCH")
            continue
        d = (a - b).abs()
        mean_rel = d.mean().item() / (a.abs().mean().item() or 1.0)
        print(f"{label:>12s} {str(tuple(a.shape)):>22s} {d.max().item():>12.3e} "
              f"{d.mean().item():>12.3e} {mean_rel:>10.2e}")

    # ---- RoPE cos/sin check: with Q/K/V bit-exact, a cos/sin mismatch is the attention divergence.
    rc, rs = ref.get("rope_cos"), ref.get("rope_sin")
    pc, ps = port_vis.get("rope_cos"), port_vis.get("rope_sin")
    if rc is not None and pc is not None:
        print("\n=== RoPE cos/sin (default rope) original ref vs port ===")
        if tuple(rc.shape) != tuple(pc.shape):
            print(f"  SHAPE MISMATCH: ref cos {tuple(rc.shape)} vs port {tuple(pc.shape)}")
        else:
            dc = (rc - pc).abs()
            ds = (rs - ps).abs()
            print(f"  cos: shape={tuple(rc.shape)} max_abs={dc.max().item():.3e} mean_abs={dc.mean().item():.3e}")
            print(f"  sin: shape={tuple(rs.shape)} max_abs={ds.max().item():.3e} mean_abs={ds.mean().item():.3e}")
            # ratio at a mid-position channel reveals a constant scale (attention_scaling) difference
            flat_r = rc.flatten().abs(); flat_p = pc.flatten().abs()
            nz = flat_r > 1e-3
            if nz.any():
                ratio = (flat_p[nz] / flat_r[nz])
                print(f"  port/ref cos ratio: mean={ratio.mean().item():.4f} "
                      f"min={ratio.min().item():.4f} max={ratio.max().item():.4f}")

    # ---- Layer-0 text-decoder submodule split: the first probe that diverges (given bit-exact block
    # input) is where the text-decoder discrepancy enters (q_norm/k_norm => qk_norm layout, etc.).
    ref_txt = ref.get("text_probes") or {}
    if ref_txt:
        print("\n=== layer-0 text-decoder submodule diffs (original ref vs port) ===")
        print(f"{'probe':>14s} {'shape':>22s} {'max_abs':>12s} {'mean_abs':>12s} {'mean_rel':>10s}")
        for key in ["L0.attn_norm", "L0.q_norm", "L0.k_norm", "L0.attn_out", "L0.ff_norm", "L0.mlp"]:
            a, b = ref_txt.get(key), port_vis.get(key)
            if a is None or b is None:
                print(f"{key:>14s} {'(missing)':>22s}")
                continue
            if tuple(a.shape) != tuple(b.shape):
                print(f"{key:>14s} {f'ref{tuple(a.shape)} port{tuple(b.shape)}':>22s}  SHAPE MISMATCH")
                continue
            d = (a - b).abs()
            mean_rel = d.mean().item() / (a.abs().mean().item() or 1.0)
            print(f"{key:>14s} {str(tuple(a.shape)):>22s} {d.max().item():>12.3e} "
                  f"{d.mean().item():>12.3e} {mean_rel:>10.2e}")

        # Discriminate RoPE/position_ids (hits all positions) from the multimodal mask (concentrated
        # in the bidirectional image region) using the attention output's per-position diff.
        a, b = ref_txt.get("L0.attn_out"), port_vis.get("L0.attn_out")
        if a is not None and b is not None and tuple(a.shape) == tuple(b.shape) and a.dim() == 3:
            ids = inputs["input_ids"][0].cpu()
            image_token_id = getattr(config, "image_token_id", None)
            per_pos = (a[0] - b[0]).abs().amax(dim=-1)  # [seq]
            print("\n  L0.attn_out per-position diff (RoPE vs mask discriminator):")
            if image_token_id is not None:
                img = ids == image_token_id
                txt_m = ~img
                if int(img.sum()):
                    print(f"    image positions: max={per_pos[img].max().item():.3e} "
                          f"mean={per_pos[img].mean().item():.3e}")
                if int(txt_m.sum()):
                    print(f"    text  positions: max={per_pos[txt_m].max().item():.3e} "
                          f"mean={per_pos[txt_m].mean().item():.3e}")
            half = per_pos.shape[0] // 2
            print(f"    first-half mean={per_pos[:half].mean().item():.3e}  "
                  f"second-half mean={per_pos[half:].mean().item():.3e} (growth-with-position => RoPE)")

    # ---- Vision-path isolation: split inputs_embeds diff by image vs text positions.
    # Image positions merge in image features additively, so the text embedding cancels and the diff
    # there == the image-feature diff; text positions should be ~0 (identical ids + embedding table).
    ref_embeds = ref.get("inputs_embeds")
    if ref_embeds is not None and port_embeds is not None:
        ids = inputs["input_ids"][0].cpu()
        image_token_id = getattr(config, "image_token_id", None)
        d = (ref_embeds - port_embeds).abs()  # [seq, hidden]
        per_pos = d.amax(dim=-1)
        print("\n=== inputs_embeds (text-layer 0 input) -- vision-path isolation ===")
        print(f"overall   : max={d.max().item():.3e}  mean={d.mean().item():.3e}")
        if image_token_id is not None:
            img_mask = ids == image_token_id
            txt_mask = ~img_mask
            n_img = int(img_mask.sum())
            print(f"image_token_id={image_token_id}  image_positions={n_img}  text_positions={int(txt_mask.sum())}")
            if n_img:
                di = d[img_mask]
                print(f"image pos : max={di.max().item():.3e}  mean={di.mean().item():.3e}")
            if int(txt_mask.sum()):
                dt = d[txt_mask]
                print(f"text  pos : max={dt.max().item():.3e}  mean={dt.mean().item():.3e}")
        order = per_pos.argsort(descending=True)[:8]
        print("top-8 divergent positions (pos, input_id, max_abs_diff):")
        for p in order.tolist():
            tag = "IMG" if (image_token_id is not None and int(ids[p]) == image_token_id) else "txt"
            print(f"  pos={p:>4d} id={int(ids[p]):>6d} [{tag}] diff={per_pos[p].item():.3e}")

    print(f"\n=== per-layer hidden-state diff (text decoder, {n_layers} layers) ===")
    print(f"{'layer':>6s} {'max_abs':>12s} {'max_rel':>10s} {'mean_abs':>12s} {'mean_rel':>10s}")
    per_layer = []
    for i in range(n_layers):
        a, b = ref_hidden[i], port_hidden[i]
        diff = (a - b).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        max_rel = max_abs / (a.abs().max().item() or 1.0)
        mean_rel = mean_abs / (a.abs().mean().item() or 1.0)  # bulk divergence vs outlier channels
        per_layer.append(max_abs)
        print(f"{i:>6d} {max_abs:>12.3e} {max_rel:>10.2e} {mean_abs:>12.3e} {mean_rel:>10.2e}")

    logits_diff = (ref["logits_last"] - port_logits[0, -1]).abs().max().item()
    top1_ref = int(ref["logits_last"].argmax())
    top1_port = int(port_logits[0, -1].argmax())
    top1_match = top1_ref == top1_port

    # Greedy-token agreement (drop input keys the port's generate path does not accept, e.g. the
    # remote processor's token_type_ids).
    tokens_match, n_tok = None, 0
    try:
        import inspect

        fwd = set(inspect.signature(port.forward).parameters)
        gen_inputs = {k: v for k, v in inputs.items() if k in fwd}
        n = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = port.generate(**gen_inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        port_tokens = gen[0, n:].cpu()
        ref_tokens = ref["generated_ids"]
        n_tok = min(len(ref_tokens), len(port_tokens))
        tokens_match = bool((ref_tokens[:n_tok] == port_tokens[:n_tok]).all())
    except Exception as e:
        print(f"[warn] port generate failed ({type(e).__name__}: {e})")

    print("\n=== summary ===")
    print(f"worst layer diff  : layer {int(max(range(n_layers), key=lambda i: per_layer[i]))} "
          f"= {max(per_layer):.3e}")
    print(f"final logits diff : {logits_diff:.3e}")
    print(f"argmax next token : ref={top1_ref} port={top1_port} ({'MATCH' if top1_match else 'MISMATCH'})")
    if tokens_match is not None:
        print(f"greedy tokens     : {'MATCH' if tokens_match else 'MISMATCH'} over {n_tok} tokens")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(range(n_layers), per_layer)
    ax.set_yscale("log")
    ax.set_xlabel("text decoder layer")
    ax.set_ylabel("max abs diff (log)")
    ax.set_title(f"Molmo2 original (ref) vs port -- per-layer hidden-state diff | logits diff {logits_diff:.2e}")
    fig.tight_layout()
    _ensure_dir(args.out_dir)
    out_path = f"{args.out_dir.rstrip('/')}/molmo2_parity.png"
    fig.savefig(out_path, dpi=120)
    print(f"[saved] {out_path}")

    ok = top1_match and (tokens_match in (None, True))
    print(f"\nPARITY {'PASS' if ok else 'FAIL'} (token-based; logits diff reported for reference)")
    return 0 if ok else 1


def cmd_parity(args) -> int:
    if getattr(args, "reference", None):
        return cmd_parity_reference(args)
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
    local_ckpt = convert_to_local_checkpoint(args.model_id, "/tmp/molmo2_port_ckpt", args.dtype)
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

    Molmo2 emits ``<points coords="<count> <id> <x> <y> [<id> <x> <y> ...]">label</points>`` where
    coordinates are on a 0-1000 grid (observed: ``coords="1 1 577 495"`` -> one point at id 1,
    x=577, y=495). Also handles the legacy ``<point x=".." y="..">`` attribute style. Returns points
    on a 0-100 scale; the scale is auto-detected (values >100 imply the 0-1000 grid)."""
    pts: list[tuple[float, float]] = []
    for x, y in POINT_RE.findall(text):
        pts.append((float(x), float(y)))

    for coords in POINTS_COORDS_RE.findall(text):
        nums = [float(n) for n in NUM_RE.findall(coords)]
        if not nums:
            continue
        count = int(nums[0]) if nums[0].is_integer() else None
        rest = nums[1:]
        if count is not None and count > 0 and len(rest) == count * 3:
            pts += [(rest[3 * i + 1], rest[3 * i + 2]) for i in range(count)]  # (id, x, y) triples
        elif count is not None and count > 0 and len(rest) == count * 2:
            pts += [(rest[2 * i], rest[2 * i + 1]) for i in range(count)]  # (x, y) pairs
        else:  # fallback: pair up all numbers
            pts += [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]

    if pts:
        scale = 1000.0 if max(max(x, y) for x, y in pts) > 100 else 100.0
        pts = [(x / scale * 100.0, y / scale * 100.0) for x, y in pts]
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
    local_ckpt = convert_to_local_checkpoint(args.model_id, "/tmp/molmo2_port_ckpt", args.dtype)
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
    png_path = f"{args.out_dir.rstrip('/')}/molmo2_image_points.png"
    overlay_points(image, pts, png_path, title=args.prompt)
    # Emit the overlay inline (base64) so it is retrievable from job logs without extra storage.
    import base64
    import os

    data = open(png_path, "rb").read()
    if len(data) < 3_000_000:
        print("BEGIN_PNG_B64")
        print(base64.b64encode(data).decode())
        print("END_PNG_B64")
    print(f"[done] overlay bytes={len(data)} at {os.path.abspath(png_path)}")
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


def cmd_video_parity(args) -> int:
    """Compare original (remote code) vs in-library port on a *video* pointing prompt.

    Same sequential, memory-safe scheme as ``parity``: load the original, run it, free it, then load
    the port and run it on the *identical* inputs. Reports final-logits diff, greedy-token agreement,
    and both decoded `<points coords=...>` payloads so we can see whether any missed/extra point is a
    port regression or simply how the original behaves."""
    torch = _torch()
    from transformers import (
        AutoModelForImageTextToText,
        Molmo2Config,
        Molmo2ForConditionalGeneration,
    )
    from transformers.video_utils import load_video

    torch_dtype = getattr(torch, args.dtype)
    _patch_for_coexistence()

    processor = load_processor(args.model_id)
    video, metadata = load_video(args.video)
    print(f"[video] {args.video}\n        frames={getattr(video, 'shape', None)} "
          f"sampled_to={args.num_frames}")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": args.prompt}, {"type": "video", "video": video},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        video_metadata=[metadata], do_sample_frames=True, num_frames=args.num_frames,
    ).to(args.device)
    n = inputs["input_ids"].shape[1]
    print(f"[inputs] input_ids={tuple(inputs['input_ids'].shape)} "
          f"pixel_values_videos={tuple(inputs['pixel_values_videos'].shape)}")

    def run(model, tag, rename=None):
        # The in-library processor emits ``mm_token_type_ids``; the original remote model wants the
        # same tensor under its native name ``token_type_ids`` (dropping it leaves its multimodal
        # merge indexing a None). Apply the per-model rename first, then auto-drop any *other* kwarg a
        # model reports as unused and retry — so each model runs the identical tensors it accepts.
        mk = dict(inputs)
        for src, dst in (rename or {}).items():
            if src in mk:
                mk[dst] = mk.pop(src)
                print(f"[{tag}] renamed {src} -> {dst}")
        for _ in range(5):
            try:
                with torch.no_grad():
                    logits = model(**mk, use_cache=False).logits[0, -1].float().cpu()
                    gen = model.generate(**mk, max_new_tokens=args.max_new_tokens, do_sample=False)
                break
            except (ValueError, TypeError) as e:
                m = re.search(r"not used by the model: \[([^\]]*)\]", str(e))
                if not m:
                    raise
                drop = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
                print(f"[{tag}] dropping unused kwargs: {drop}")
                for d in drop:
                    mk.pop(d, None)
        else:
            raise RuntimeError(f"[{tag}] could not satisfy model kwargs after dropping")
        tok = gen[0, n:].cpu()
        text = processor.batch_decode(gen[:, n:], skip_special_tokens=True)[0]
        print(f"[{tag}] kept_keys={sorted(mk)}")
        print(f"[{tag}] generated: {text}")
        return logits, tok, text

    # ---- Phase 1: ORIGINAL (remote code) ----
    print(f"[phase 1] original (remote code) {args.model_id} dtype={args.dtype} attn={args.attn}")
    original = AutoModelForImageTextToText.from_pretrained(
        args.model_id, trust_remote_code=True, dtype=torch_dtype, attn_implementation=args.attn
    ).to(args.device).eval()
    o_logits, o_tok, o_text = run(original, "original", rename={"mm_token_type_ids": "token_type_ids"})
    del original
    _free()

    # ---- Phase 2: PORT (in-library, remapped weights) ----
    print("[phase 2] in-library port from remapped weights")
    local_ckpt = convert_to_local_checkpoint(args.model_id, "/tmp/molmo2_port_ckpt", args.dtype)
    config = Molmo2Config.from_pretrained(args.model_id, trust_remote_code=False)
    config._attn_implementation = args.attn
    port = Molmo2ForConditionalGeneration.from_pretrained(
        local_ckpt, config=config, dtype=torch_dtype, attn_implementation=args.attn, device_map=args.device,
    ).eval()
    p_logits, p_tok, p_text = run(port, "port")
    del port
    _free()

    # ---- Compare ----
    logits_diff = (o_logits - p_logits).abs().max().item()
    top1_match = o_logits.argmax().item() == p_logits.argmax().item()
    m = min(len(o_tok), len(p_tok))
    tok_match = bool((o_tok[:m] == p_tok[:m]).all())
    first_div = next((i for i in range(m) if o_tok[i] != p_tok[i]), None)
    print("\n=== video parity summary ===")
    print(f"final logits diff : {logits_diff:.3e}")
    print(f"argmax next token : {'MATCH' if top1_match else 'MISMATCH'}")
    print(f"greedy tokens     : {'MATCH' if tok_match else 'MISMATCH'} over {m} tokens"
          + ("" if tok_match else f" (first divergence at token {first_div})"))
    print(f"text identical    : {o_text == p_text}")
    ok = top1_match and tok_match
    if args.dtype == "float32":
        ok = ok and logits_diff < args.tol
    print(f"VIDEO PARITY {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_upload(args) -> int:
    """Convert the original checkpoint to the in-library Molmo2 layout and push it to the Hub.

    Memory-safe: the conversion streams shards (no full model in RAM) and we upload the folder
    directly, adding the canonical port config, the port processor, the original generation config,
    and a model card that credits the original."""
    import json
    import os

    from huggingface_hub import HfApi, hf_hub_download
    from transformers import Molmo2Config

    out_dir = convert_to_local_checkpoint(args.model_id, "/tmp/molmo2_port_upload", args.dtype)

    # Canonical port config (overwrites the original config.json copied during conversion).
    config = Molmo2Config.from_pretrained(args.model_id, trust_remote_code=False)
    config.torch_dtype = args.dtype
    config.save_pretrained(out_dir)

    # Port processor (tokenizer + image/video processor + chat template). Also save the sub
    # processors explicitly so standalone preprocessor_config.json / video_preprocessor_config.json
    # exist (AutoImageProcessor needs them; the combined processor_config.json is not enough).
    processor = load_processor(args.model_id)
    processor.save_pretrained(out_dir)
    processor.image_processor.save_pretrained(out_dir)
    if getattr(processor, "video_processor", None) is not None:
        processor.video_processor.save_pretrained(out_dir)

    # Strip every `auto_map` (inherited from the original's remote-code repo) from all JSON configs,
    # so the repo loads via the in-library classes with NO trust_remote_code.
    def strip_auto_map(obj):
        if isinstance(obj, dict):
            obj.pop("auto_map", None)
            for v in obj.values():
                strip_auto_map(v)
        elif isinstance(obj, list):
            for v in obj:
                strip_auto_map(v)

    for fn in os.listdir(out_dir):
        if fn.endswith(".json"):
            path = os.path.join(out_dir, fn)
            try:
                data = json.load(open(path))
            except Exception:
                continue
            before = json.dumps(data)
            strip_auto_map(data)
            if json.dumps(data) != before:
                json.dump(data, open(path, "w"), indent=2)
                print(f"[upload] stripped auto_map from {fn}")

    # Carry over the original generation config if present.
    try:
        gen = hf_hub_download(args.model_id, "generation_config.json")
        import shutil

        shutil.copy(gen, os.path.join(out_dir, "generation_config.json"))
    except Exception as e:
        print(f"[warn] no generation_config.json carried over ({e})")

    card = f"""---
license: apache-2.0
base_model: {args.model_id}
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- molmo2
- multimodal
- pointing
---

# Molmo2-8B (transformers format)

This is [`{args.model_id}`]({"https://huggingface.co/" + args.model_id}) converted to the
in-library 🤗 Transformers `Molmo2` implementation (no `trust_remote_code` needed). Weights are the
original ones, only renamed to the in-library parameter layout (`dtype={args.dtype}`).

Verified bit-exact against the original in float32 + SDPA across the full model (vision backbone,
pooling, projection, all text-decoder layers, and final logits).

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained("{args.repo}", dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("{args.repo}")
```

All credit for the model goes to the original authors (Ai2). See the base model card for license,
training details, and intended use.
"""
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(card)

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=out_dir, repo_id=args.repo, repo_type="model",
                      commit_message=f"Add Molmo2-8B converted from {args.model_id} (transformers format)")
    vis = "private" if args.private else "public"
    print(f"[uploaded] https://huggingface.co/{args.repo} ({vis})")
    return 0


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
    sp.add_argument("--reference", default=None,
                    help="reference dump (local path or HF dataset repo id) from reference_molmo2.py; "
                         "compares the port against it instead of loading the original in-process")

    sp = sub.add_parser("demo", help="run the port, parse grounding points, save overlay")
    common(sp)
    sp.set_defaults(max_new_tokens=256)

    sp = sub.add_parser("reference", help="dump original outputs for cross-version compare")
    common(sp)

    sp = sub.add_parser("video-parity", help="original vs port greedy/logits compare on a video")
    common(sp)
    sp.set_defaults(prompt="Point to the penguins.", attn="sdpa", dtype="bfloat16", max_new_tokens=256)
    sp.add_argument("--video", default=DEFAULT_VIDEO)
    sp.add_argument("--num-frames", type=int, default=24)
    sp.add_argument("--tol", type=float, default=2e-2)

    sp = sub.add_parser("upload", help="convert original -> in-library layout and push to the Hub")
    sp.add_argument("--model-id", default=MODEL_ID)
    sp.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    sp.add_argument("--repo", required=True, help="target model repo id, e.g. user/Molmo2-8B-hf")
    sp.add_argument("--private", action="store_true", help="create the repo as private")

    args = p.parse_args()
    if args.cmd == "selftest":
        return selftest(args.model_id)
    if args.cmd == "parity":
        return cmd_parity(args)
    if args.cmd == "demo":
        return cmd_demo(args)
    if args.cmd == "reference":
        return cmd_reference(args)
    if args.cmd == "video-parity":
        return cmd_video_parity(args)
    if args.cmd == "upload":
        return cmd_upload(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
