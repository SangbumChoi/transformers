# Copyright 2026 NVIDIA CORPORATION and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...cache_utils import DynamicCache
from ...generation import GenerationMixin
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    is_flash_attn_2_available,
    is_peft_available,
    logging,
    torch_compilable_check,
)
from ..qwen2.modeling_qwen2 import Qwen2ForCausalLM
from ..qwen3.modeling_qwen3 import Qwen3ForCausalLM
from .configuration_locateanything import LocateAnythingConfig
from .generate_utils import (
    build_window_attention_mask,
    get_token_ids_from_config,
    handle_pattern,
    sample_tokens,
)
from .modeling_vit import MoonVitPretrainedModel


logger = logging.get_logger(__name__)

__all__ = ["LocateAnythingForConditionalGeneration", "LocateAnythingPreTrainedModel"]


LOCATEANYTHING_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`LocateAnythingConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LocateAnything Model outputting raw hidden-states without any specific head on top.",
    LOCATEANYTHING_START_DOCSTRING,
)
class LocateAnythingPreTrainedModel(PreTrainedModel):
    config_class = LocateAnythingConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_quantized_cache = True
    _supports_sdpa = True

    @classmethod
    def _autoset_attn_implementation(cls, config, *args, **kwargs):
        if getattr(config, "_attn_implementation", None) == "magi":
            return config
        return super()._autoset_attn_implementation(config, *args, **kwargs)

    def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False, *args, **kwargs):
        if attn_implementation == "magi":
            return "magi"
        return super()._check_and_adjust_attn_implementation(attn_implementation, is_init_check, *args, **kwargs)

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", None) or self.config.text_config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LocateAnythingForConditionalGeneration(LocateAnythingPreTrainedModel, GenerationMixin):
    config_class = LocateAnythingConfig

    def __init__(self, config: LocateAnythingConfig, vision_model=None, language_model=None):
        super().__init__(config)

        self.template = config.template
        self.mlp_checkpoint = config.mlp_checkpoint

        logger.debug("mlp_checkpoint: %s", self.mlp_checkpoint)
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.vision_config.model_type == "moonvit":
                vision_attn_impl = getattr(config.vision_config, "_attn_implementation", None) or "flash_attention_2"
                if vision_attn_impl == "flash_attention_2" and not is_flash_attn_2_available():
                    logger.warning_once("flash_attn is not available for MoonViT inference; falling back to sdpa.")
                    vision_attn_impl = "sdpa"
                config.vision_config._attn_implementation = vision_attn_impl
                self.vision_model = MoonVitPretrainedModel(config.vision_config)
            else:
                raise ValueError(
                    f"Unsupported vision model type: {config.vision_config.model_type}. Only moonvit is supported."
                )

        text_attn_impl = (
            getattr(config.text_config, "_attn_implementation", None)
            or getattr(config, "_attn_implementation", None)
            or "sdpa"
        )
        if text_attn_impl == "flash_attention_2" and not is_flash_attn_2_available():
            logger.warning_once("flash_attn is not available for LocateAnything text inference; falling back to sdpa.")
            text_attn_impl = "sdpa"
        config.text_config._attn_implementation = text_attn_impl
        config.text_config._attn_implementation_internal = text_attn_impl

        if language_model is not None:
            self.language_model = language_model
        else:
            if config.text_config.architectures[0] == "Qwen2ForCausalLM":
                self.language_model = Qwen2ForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == "Qwen3ForCausalLM":
                self.language_model = Qwen3ForCausalLM(config.text_config)
            else:
                raise ValueError(
                    f"Unsupported language model architecture: {config.text_config.architectures[0]}. Only "
                    "Qwen2ForCausalLM and Qwen3ForCausalLM are supported."
                )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        # MLP for moonvit (without pixel_shuffle_back, direct mapping)
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * 4),
            nn.Linear(vit_hidden_size * 4, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        self.image_token_index = config.image_token_index
        self.neftune_alpha = None

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        self.use_llm_lora = config.use_llm_lora
        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        self.token_ids = get_token_ids_from_config(config)

        # Set _no_split_modules dynamically based on the actual LLM architecture
        arch = (
            config.text_config.architectures[0]
            if hasattr(config.text_config, "architectures") and config.text_config.architectures
            else "Qwen2ForCausalLM"
        )
        if "Qwen3" in arch:
            self._no_split_modules = ["Qwen3DecoderLayer"]
        else:
            self._no_split_modules = ["Qwen2DecoderLayer"]

        # Initialize weights and set up tied-weight bookkeeping.
        self.post_init()

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        if not is_peft_available():
            raise ImportError("PEFT is required to enable LocateAnything vision backbone LoRA adapters.")
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=r,
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
                "mlp.fc1",
                "mlp.fc2",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        if not is_peft_available():
            raise ImportError("PEFT is required to enable LocateAnything language model LoRA adapters.")
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=r,
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.down_proj",
                "mlp.up_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.use_llm_lora = True

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_hws: torch.Tensor | None = None,
        image_flags: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        """
        Encodes image pixels with the vision tower and projects them into the language model hidden size.
        """
        image_features = self.extract_feature(pixel_values, image_grid_hws)

        if image_flags is not None and image_flags.sum() > 0:
            filtered_image_features = []
            feature_index = 0
            for flag in image_flags:
                num_images = flag.item()
                if num_images != 0:
                    filtered_image_features.extend(image_features[feature_index : feature_index + num_images])
                    feature_index += num_images
                else:
                    feature_index += 1
            image_features = filtered_image_features

        if not image_features:
            return torch.empty(
                0, self.config.text_config.hidden_size, device=pixel_values.device, dtype=pixel_values.dtype
            )

        image_features = torch.cat(image_features, dim=0)
        return self.mlp1(image_features)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ) -> torch.BoolTensor:
        """
        Obtains the image placeholder mask and checks that the number of placeholder tokens matches image features.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.image_token_index, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.image_token_index

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        torch_compilable_check(
            inputs_embeds[special_image_mask].numel() == image_features.numel(),
            f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
        )
        return special_image_mask

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_grid_hws: torch.Tensor | None = None,
        image_flags: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values, image_grid_hws=image_grid_hws, image_flags=image_flags
            )
            image_features = image_features.to(input_embeds.device, input_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=input_embeds, image_features=image_features
            )
            input_embeds = input_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def extract_feature(self, pixel_values, image_grid_hws):
        vit_embeds = self.vision_model(pixel_values=pixel_values, grid_hws=image_grid_hws)

        return vit_embeds

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        visual_features: torch.FloatTensor | None = None,
        image_grid_hws: torch.Tensor | None = None,
        tokenizer=None,
        n_future_tokens: int = 6,
        **generate_kwargs,
    ) -> str:
        r"""
        Parallel Box Decoding (PBD) generation. Three modes are supported:

        - `"slow"`: pure auto-regressive decoding.
        - `"fast"`: block-wise multi-token prediction (MTP) only.
        - `"hybrid"` (default): MTP first, falling back to AR on uncertain boxes.

        The language model is the standard library decoder; the MTP block-diffusion attention
        mask is built by [`~models.locateanything.generate_utils.build_window_attention_mask`]
        and passed in as a 4D mask, and the speculative window is rolled back from the
        [`~cache_utils.DynamicCache`] after each MTP step.
        """
        # `verbose` is accepted for backward compatibility but no longer drives any timing logic.
        generate_kwargs.pop("verbose", False)

        pixel_values = pixel_values.to(self.language_model.dtype)
        if isinstance(image_grid_hws, np.ndarray):
            image_grid_hws = torch.from_numpy(image_grid_hws).to(pixel_values.device, dtype=torch.int32)

        batch_size, seq_len = input_ids.shape
        if batch_size != 1:
            raise ValueError("LocateAnything generation only supports batch size 1 for now.")
        if not generate_kwargs.get("use_cache", False):
            raise ValueError("LocateAnything generation only supports `use_cache=True`.")

        generation_mode = generate_kwargs.get("generation_mode", "hybrid")
        if generation_mode not in ("fast", "slow", "hybrid"):
            raise ValueError(f"Unsupported generation_mode='{generation_mode}'. Use 'fast', 'slow', or 'hybrid'.")

        device = input_ids.device
        embed_tokens = self.language_model.get_input_embeddings()

        # Build the prompt embeddings once, scattering the projected image features into the
        # image placeholder positions (same merge used by `forward`).
        if visual_features is not None:
            vit_embeds = visual_features
        elif pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values, image_grid_hws)
            vit_embeds = torch.cat(vit_embeds, dim=0)
            vit_embeds = self.mlp1(vit_embeds)
        else:
            vit_embeds = None

        prompt_embeds = embed_tokens(input_ids)
        if vit_embeds is not None:
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=prompt_embeds, image_features=vit_embeds
            )
            prompt_embeds = prompt_embeds.masked_scatter(special_image_mask, vit_embeds.to(prompt_embeds.dtype))

        generated = input_ids.clone()
        total_gen_length = min(tokenizer.model_max_length, seq_len + generate_kwargs.get("max_new_tokens", 2048))
        past_key_values = None

        use_mtp = generation_mode in ("fast", "hybrid")
        block_size = n_future_tokens
        default_mask_token_id = self.token_ids["default_mask_token_id"]
        pre_mask_tokens = torch.full(
            (batch_size, n_future_tokens - 1), default_mask_token_id, dtype=generated.dtype, device=device
        )
        max_possible_len = total_gen_length + n_future_tokens
        full_position_ids = torch.arange(0, max_possible_len, device=device).unsqueeze(0)

        def _embed_new(sequence, past_len):
            """Embed the not-yet-cached suffix of `sequence`, reusing merged prompt embeds for the prefill."""
            if past_len == 0:
                # First forward: the prompt (with merged image features) plus any appended tokens.
                if sequence.size(1) > seq_len:
                    return torch.cat([prompt_embeds, embed_tokens(sequence[:, seq_len:])], dim=1)
                return prompt_embeds
            return embed_tokens(sequence[:, past_len:])

        while generated.size(1) < total_gen_length:
            commit_len = generated.size(1)
            past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

            if use_mtp:
                sequence = torch.cat((generated, generated[:, -1:], pre_mask_tokens), dim=1)
                position_ids = full_position_ids[:, past_len : sequence.size(1)].clone()
                position_ids[0, -n_future_tokens:] -= 1
                q_len = sequence.size(1) - past_len
                attn = build_window_attention_mask(
                    q_len, sequence.size(1), past_len, block_size, prompt_embeds.dtype, device
                )
            else:
                sequence = generated
                position_ids = full_position_ids[:, past_len : sequence.size(1)]
                attn = torch.ones((batch_size, sequence.size(1)), dtype=torch.long, device=device)

            inputs_embeds = _embed_new(sequence, past_len)
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                position_ids=position_ids,
                past_key_values=past_key_values if past_key_values is not None else DynamicCache(),
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            # Roll back the speculative window so only committed tokens stay cached.
            past_key_values.crop(commit_len)

            if use_mtp:
                next_token_logits = outputs.logits[:, -n_future_tokens:, :]
                _, _, x0, box_avg = sample_tokens(
                    next_token_logits, generated, self.token_ids, keep_k=5, **generate_kwargs
                )
                is_box_empty = (box_avg[0] == 0).all()
                new_tokens = x0[0] if is_box_empty else box_avg[0]
                out_pattern = handle_pattern(new_tokens, self.token_ids, generation_mode)
                out_type = out_pattern["type"]
                out_token = torch.tensor(out_pattern["tokens"], dtype=generated.dtype, device=device)
            else:
                next_token_logits = outputs.logits[:, -1:, :]
                _, _, x0, _ = sample_tokens(next_token_logits, generated, self.token_ids, **generate_kwargs)
                out_token = x0[0]
                out_type = self._classify_ar_token(out_token[0].item(), generation_mode)

            generated = torch.cat([generated, out_token.unsqueeze(0)], dim=1)

            if out_type == "im_end":
                break
            if generation_mode == "hybrid":
                if out_type == "error_box":
                    use_mtp = False
                elif out_type == "box_end_ar":
                    use_mtp = True

        response = tokenizer.batch_decode(generated[:, seq_len:], skip_special_tokens=False)
        return response[0]

    def _classify_ar_token(self, token_val: int, generation_mode: str) -> str:
        """Classify a single AR token to drive hybrid mode switching and termination."""
        if generation_mode == "hybrid":
            if token_val == self.token_ids["box_end_token_id"]:
                return "box_end_ar"
            if (
                self.token_ids["coord_start_token_id"] <= token_val <= self.token_ids["coord_end_token_id"]
                or token_val == self.token_ids["none_token_id"]
            ):
                return "coord_ar"
            return "im_end"
        return "im_end" if token_val == self.token_ids["im_end_token_id"] else "continue_ar"
