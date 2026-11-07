# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from the following projects:
#   - InternVL (https://github.com/OpenGVLab/InternVL), licensed under the MIT License.
#   - LLaVA-OneVision / LLaVA-NeXT in the Hugging Face Transformers project
#     (https://github.com/huggingface/transformers), licensed under the Apache License, Version 2.0.
#
# Modifications © 2026 NVIDIA CORPORATION & AFFILIATES, licensed under
# the Apache License, Version 2.0.
#
# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License
# --------------------------------------------------------
# Hugging Face Transformers (LLaVA-OneVision / LLaVA-NeXT)
# Copyright 2024 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0
# --------------------------------------------------------

import warnings
import inspect
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
import torch.utils.checkpoint as cp
from ..moon_vit.modeling_vit import MoonVitPretrainedModel
from peft import LoraConfig, get_peft_model
from transformers.generation import GenerationMixin
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from .configuration_eagle_embodied import EagleEmbodiedConfig
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from eaglevl.sp_utils import get_pg_manager, split_for_sequence_parallel
from eaglevl.train.liger_loss_weight_ops import LigerFusedLinearCrossEntropyLoss
from eaglevl.train.sequence_parallel import (
    graph_connected_fp32_zero,
    pad_packed_sequence_for_ulysses,
    shift_packed_labels,
)


logger = logging.get_logger(__name__)
_MIN_FUSED_CE_TOKENS = 64


# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L241C1-L280C1
EagleEmbodied_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`EagleEmbodiedConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

@add_start_docstrings(
    "The bare EagleEmbodied Model outputting raw hidden-states without any specific head on top.",
    EagleEmbodied_START_DOCSTRING,
)
class EagleEmbodiedPreTrainedModel(PreTrainedModel):
    config_class = EagleEmbodiedConfig
    base_model_prefix = "model"
    main_input_name = 'input_ids'
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer", "Qwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_flash_attn = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_quantized_cache = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

IGNORE_INDEX = -100


def _shift_packed_label_weights(label_weights, shifted_labels, sub_sample_lengths):
    """Align per-label loss weights with ``shift_packed_labels`` output.

    ``label_weights`` matches the unshifted labels, which Ulysses padding may
    have extended. Padded positions and positions whose shifted label is
    ignored get weight zero.
    """
    if label_weights is None:
        return None
    weights = label_weights.to(dtype=torch.float32)
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)
    padding = shifted_labels.shape[1] - weights.shape[1]
    if weights.shape[0] != shifted_labels.shape[0] or padding < 0:
        raise ValueError(
            f'label_weights must have shape {tuple(shifted_labels.shape)}, got {tuple(weights.shape)}'
        )
    shifted = shift_packed_labels(F.pad(weights, (0, padding)), sub_sample_lengths)
    return shifted.masked_fill(shifted_labels == IGNORE_INDEX, 0.0)


class EagleEmbodiedForConditionalGeneration(EagleEmbodiedPreTrainedModel, GenerationMixin):
    config_class = EagleEmbodiedConfig
    def __init__(self, config: EagleEmbodiedConfig, vision_model=None, language_model=None):
        super().__init__(config)

        self.template = config.template
        self.mlp_checkpoint = config.mlp_checkpoint

        logger.info(f'mlp_checkpoint: {self.mlp_checkpoint}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.vision_config.model_type == 'moonvit':
                config.vision_config._attn_implementation = 'flash_attention_2'
                self.vision_model = MoonVitPretrainedModel(config.vision_config, use_final_layernorm=False)
            else:
                raise ValueError(f'Unsupported vision model type: {config.vision_config.model_type}. Only moonvit is supported.')

        config.text_config._attn_implementation = 'flash_attention_2'

        if language_model is not None:
            self.language_model = language_model
        else:
            if config.text_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Qwen3ForCausalLM':
                self.language_model = Qwen3ForCausalLM(config.text_config)
            else:
                raise ValueError(f'Unsupported language model architecture: {config.text_config.architectures[0]}. Only Qwen2ForCausalLM and Qwen3ForCausalLM are supported.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        # MLP for moonvit (without pixel_shuffle_back, direct mapping)
        self.mlp1 = nn.Sequential(
                nn.LayerNorm(vit_hidden_size*4),
                nn.Linear(vit_hidden_size*4, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
        self.image_token_index = config.image_token_index
        self.neftune_alpha = None

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        self.use_llm_lora = config.use_llm_lora
        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        self.check_forward_kwargs()

    @property
    def supports_vflash_generation(self):
        """True for a Qwen3 language model, the only target a VFlash drafter can bind to."""
        text_config = self.config.text_config
        architectures = getattr(text_config, 'architectures', None) or []
        return (
            getattr(text_config, 'model_type', None) == 'qwen3'
            and bool(architectures)
            and architectures[0] == 'Qwen3ForCausalLM'
        )

    def check_forward_kwargs(self):
        # We intentionally avoid using **kwargs in forward because Hugging Face Transformers
        # has special handling for functions with **kwargs parameters that would affect
        # how our model is processed during training and inference.
        forward_params = inspect.signature(self.forward).parameters
        assert not any(k.kind == inspect.Parameter.VAR_KEYWORD for k in forward_params.values())


    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj',
                            'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                            'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()
        self.use_llm_lora = True

    def get_sub_sample_lengths(self, input_ids):
        # for compatibility with packing
        sub_sample_lengths = [torch.tensor([each.shape[0]], device=input_ids.device, dtype=torch.int32) for each in input_ids]
        return sub_sample_lengths

    def forward(
            self,
            pixel_values: List[torch.FloatTensor],
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_grid_hws: Optional[torch.Tensor] = None,
            image_flags: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            label_weights: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            sub_sample_lengths: Optional[List[torch.Tensor]] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if sub_sample_lengths is None:
            sub_sample_lengths = self.get_sub_sample_lengths(input_ids)

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        if input_embeds.dim() == 2:
            input_embeds = input_embeds.unsqueeze(0)

        has_images = image_flags is not None and image_flags.sum() > 0

        vit_embeds = self.extract_feature(pixel_values, image_grid_hws)

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if has_images:
            filtered_vit_embeds = []
            idx = 0
            for flag in image_flags:
                flag_val = flag.item()
                if flag_val != 0:
                    filtered_vit_embeds.extend(vit_embeds[idx:idx + flag_val])
                    idx += flag_val
                else:
                    idx += 1

            vit_embeds = filtered_vit_embeds
            vit_embeds = torch.cat(vit_embeds, dim=0)

            vit_embeds = self.mlp1(vit_embeds)
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.image_token_index)
            try:
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds
                ignore_flag = False
            except Exception as e:
                print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.sum()
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
                ignore_flag = True
        else:
            ignore_flag = False
            vit_embeds = torch.cat(vit_embeds, dim=0)
            vit_embeds = self.mlp1(vit_embeds)
            input_embeds[0] = vit_embeds.sum()

        input_embeds = input_embeds.reshape(B, N, C)


        if self.use_llm_lora:
            language_model_forward = self.language_model.model.model.forward
        else:
            language_model_forward = self.language_model.model.forward

        if position_ids is None:
            position_ids = self._generate_position_ids(input_embeds, sub_sample_lengths)

        pg_manager = get_pg_manager()
        if pg_manager is not None:
            sp_size = pg_manager.sequence_parallel_world_size
            input_embeds, attention_mask, position_ids, labels, sub_sample_lengths = (
                pad_packed_sequence_for_ulysses(
                    input_embeds,
                    attention_mask,
                    position_ids,
                    labels,
                    sub_sample_lengths,
                    sp_size,
                )
            )
            shifted_labels = shift_packed_labels(labels, sub_sample_lengths)
            shifted_label_weights = _shift_packed_label_weights(
                label_weights, shifted_labels, sub_sample_lengths
            )
            local_input_embeds = split_for_sequence_parallel(
                input_embeds,
                dim=1,
                sp_group=pg_manager.ulysses_sequence_parallel_group,
            )
            local_position_ids = split_for_sequence_parallel(
                position_ids,
                dim=1,
                sp_group=pg_manager.ulysses_sequence_parallel_group,
            )
            local_shifted_labels = split_for_sequence_parallel(
                shifted_labels,
                dim=1,
                sp_group=pg_manager.ulysses_sequence_parallel_group,
            )
            local_shifted_label_weights = (
                None
                if shifted_label_weights is None
                else split_for_sequence_parallel(
                    shifted_label_weights,
                    dim=1,
                    sp_group=pg_manager.ulysses_sequence_parallel_group,
                )
            )
            # Ulysses all-to-all reconstructs the full sequence inside each
            # attention layer, so every rank keeps the full packing mask.
            local_attention_mask = attention_mask
        else:
            shifted_labels = shift_packed_labels(labels, sub_sample_lengths)
            local_input_embeds = input_embeds
            local_position_ids = position_ids
            local_attention_mask = attention_mask
            local_shifted_labels = shifted_labels
            local_shifted_label_weights = _shift_packed_label_weights(
                label_weights, shifted_labels, sub_sample_lengths
            )

        outputs = language_model_forward(
            inputs_embeds=local_input_embeds,
            attention_mask=local_attention_mask,
            position_ids=local_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
            )

        # not every token needs to be computed by lm_head, we only compute the tokens that have valid labels
        hidden_states = outputs.last_hidden_state
        lm_head_weight = self.language_model.lm_head.weight

        hidden_dim = hidden_states.shape[-1]

        shift_hidden_states = hidden_states.contiguous().view(-1, hidden_dim)
        shift_labels = local_shifted_labels.contiguous().view(-1)

        # Prompt and media positions are ignored by the loss. Filtering them
        # before fused linear CE avoids unnecessary vocabulary projection work,
        # which is substantial for long multimodal contexts.
        valid_labels = shift_labels != IGNORE_INDEX
        valid_hidden_states = shift_hidden_states[valid_labels].contiguous()
        valid_shift_labels = shift_labels[valid_labels].contiguous()

        valid_loss_weights = None
        if local_shifted_label_weights is not None:
            shift_label_weights = local_shifted_label_weights.contiguous().view(-1)
            if shift_label_weights.numel() != shift_labels.numel():
                raise ValueError(
                    'label_weights must align with labels token-for-token, got '
                    f'{shift_label_weights.numel()} weights for {shift_labels.numel()} labels'
                )
            valid_loss_weights = shift_label_weights[valid_labels]

        def compute_ce(reduction):
            # The local SP shard can contain only one or two supervised tokens
            # when a long prompt ends near a shard boundary. The custom Triton
            # fused-sum kernel is numerically unreliable for that degenerate
            # shape; materializing these few rows is cheap and robust.
            if valid_shift_labels.numel() < _MIN_FUSED_CE_TOKENS:
                logits = F.linear(valid_hidden_states, lm_head_weight).float()
                if valid_loss_weights is not None:
                    token_loss = F.cross_entropy(logits, valid_shift_labels, reduction='none')
                    return (token_loss * valid_loss_weights.to(token_loss)).sum()
                return F.cross_entropy(logits, valid_shift_labels, reduction=reduction)
            liger_loss_fn = LigerFusedLinearCrossEntropyLoss(
                ignore_index=IGNORE_INDEX,
                reduction=reduction,
            )
            return liger_loss_fn(
                lm_head_weight,
                valid_hidden_states,
                valid_shift_labels,
                loss_weight=valid_loss_weights,
            )

        if pg_manager is None:
            if valid_shift_labels.numel() == 0:
                loss = graph_connected_fp32_zero(shift_hidden_states, lm_head_weight)
            elif valid_loss_weights is None:
                loss = compute_ce('mean')
            else:
                # Weights scale each label's term; the denominator stays the
                # number of supervised labels.
                loss = compute_ce('sum') / valid_shift_labels.numel()
        else:
            if valid_shift_labels.numel() == 0:
                # Every rank must enter the scalar all-reduce with the same
                # dtype. Fused/standard CE returns FP32; a BF16 graph-connected
                # zero on empty-label shards silently corrupts NCCL reduction.
                local_loss_sum = graph_connected_fp32_zero(
                    shift_hidden_states,
                    lm_head_weight,
                )
            else:
                local_loss_sum = compute_ce('sum')

            if not torch.isfinite(local_loss_sum.detach()) or local_loss_sum.detach() < -1e-6:
                raise FloatingPointError(
                    'Local cross-entropy sum must be finite and non-negative, got '
                    f'{local_loss_sum.detach().item()} for {valid_shift_labels.numel()} labels.'
                )

            local_valid_count = torch.tensor(
                valid_shift_labels.numel(),
                dtype=torch.long,
                device=hidden_states.device,
            )
            global_valid_count = local_valid_count.clone()
            dist.all_reduce(
                global_valid_count,
                op=dist.ReduceOp.SUM,
                group=pg_manager.sequence_parallel_group,
            )
            if global_valid_count.item() == 0:
                loss = local_loss_sum
            else:
                # DeepSpeed averages replicated parameter gradients across all
                # ranks. Compensate for the SP replicas so their disjoint token
                # gradients add up to the full packed-sequence mean.
                gradient_loss = (
                    local_loss_sum
                    * pg_manager.sequence_parallel_world_size
                    / global_valid_count.to(local_loss_sum.dtype)
                )
                global_loss_sum = local_loss_sum.detach().clone()
                dist.all_reduce(
                    global_loss_sum,
                    op=dist.ReduceOp.SUM,
                    group=pg_manager.sequence_parallel_group,
                )
                if not torch.isfinite(global_loss_sum) or global_loss_sum < -1e-6:
                    raise FloatingPointError(
                        'Reduced cross-entropy sum must be finite and non-negative, got '
                        f'{global_loss_sum.item()}.'
                    )
                global_mean = global_loss_sum / global_valid_count.to(global_loss_sum.dtype)
                loss = gradient_loss + (global_mean - gradient_loss.detach())
        logits = None

        if ignore_flag:
            loss = loss * 0.0

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

    @staticmethod
    def _select_flagged_vit_embeds(vit_embeds, image_flags):
        """Keep the per-image features of the images whose ``image_flags`` entry is non-zero.

        A flag ``n > 0`` keeps the next ``n`` features; a zero flag skips the one
        placeholder image of a text-only sample.
        """
        filtered_vit_embeds = []
        feature_index = 0
        for flag in image_flags:
            count = int(flag.item())
            if count:
                filtered_vit_embeds.extend(
                    vit_embeds[feature_index:feature_index + count]
                )
                feature_index += count
            else:
                feature_index += 1
        return filtered_vit_embeds

    def prepare_multimodal_inputs_embeds(
            self,
            input_ids,
            pixel_values=None,
            image_grid_hws=None,
            visual_features=None,
            image_flags=None,
    ):
        """Build the language-model prompt embeddings used for generation.

        Speculative decoders must condition on the language hidden states after
        visual features have replaced the ``<IMG_CONTEXT>`` embeddings, so raw
        vision features are not a valid substitute for this method's output.
        """
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        if isinstance(image_grid_hws, np.ndarray):
            reference = pixel_values if pixel_values is not None else input_embeds
            image_grid_hws = torch.from_numpy(image_grid_hws).to(
                reference.device, dtype=torch.int32
            )
        if visual_features is not None:
            vit_embeds = visual_features
        elif pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values, image_grid_hws)
        else:
            return input_embeds

        if image_flags is not None:
            if int(image_flags.sum()) == 0:
                dummy = self.mlp1(torch.cat(vit_embeds, dim=0))
                input_embeds[:, 0, :] = dummy.sum()
                return input_embeds
            vit_embeds = self._select_flagged_vit_embeds(vit_embeds, image_flags)

        batch_size, sequence_length, hidden_size = input_embeds.shape
        flat_embeds = input_embeds.reshape(batch_size * sequence_length, hidden_size)
        if image_grid_hws is not None:
            vit_embeds = self.mlp1(torch.cat(vit_embeds, dim=0))
            selected = input_ids.reshape(-1) == self.image_token_index
            if int(selected.sum()) != int(vit_embeds.shape[0]):
                raise ValueError(
                    "Image-feature/token mismatch while preparing prompt embeddings: "
                    f"{int(vit_embeds.shape[0])} features for {int(selected.sum())} tokens"
                )
            flat_embeds[selected] = vit_embeds
        return flat_embeds.reshape(batch_size, sequence_length, hidden_size)

    def _apply(self, fn, recurse=True):
        """Move an attached VFlash drafter, which is not a registered submodule, with this model."""
        draft = getattr(self, 'vflash_draft', None)
        result = super()._apply(fn, recurse=recurse)
        if draft is not None:
            draft._apply(fn, recurse=recurse)
        # Dtype casts change the weight fingerprint the drafter is bound to, so
        # drop the cached value and let the next VFlash call recompute it.
        if getattr(self, '_vflash_target_weights_sha256', None) is not None:
            object.__setattr__(self, '_vflash_target_weights_sha256', None)
        return result

    def load_vflash_draft(self, draft_model_path, **kwargs):
        """Load the VFlash drafter bound to this model and attach it for VFlash generation.

        Keyword arguments go to ``eaglevl.model.vflash.binding.load_vflash_draft``.
        The drafter is not registered as a submodule, so the state dict and the
        weight fingerprint stay unchanged.
        """
        from eaglevl.model.vflash.binding import load_vflash_draft

        draft = load_vflash_draft(self, draft_model_path, **kwargs)
        object.__setattr__(self, 'vflash_draft', draft)
        object.__setattr__(self, '_vflash_draft_path', str(draft_model_path))
        return draft

    def _generate_position_ids(self, input_embeds, sub_sample_lengths=None):
        """Generate position IDs for the input embeddings.

        Args:
            input_embeds: Input embeddings tensor
            sub_sample_lengths: Optional list of sub-sample lengths for each batch item

        Returns:
            position_ids: Position IDs tensor
        """
        if sub_sample_lengths is not None:
            bsz = len(sub_sample_lengths)
            position_ids = []
            for b in range(bsz):
                each_sum_sample_lengths = sub_sample_lengths[b]
                position_ids.append(torch.cat([torch.arange(each, dtype=torch.long, device=input_embeds.device) for each in each_sum_sample_lengths]))
            position_ids = torch.stack(position_ids)
            if position_ids.shape[1] != input_embeds.shape[1]:
                print(position_ids.shape[1], input_embeds.shape[1], sub_sample_lengths)
        else:
            position_ids = torch.arange(0, input_embeds.shape[1], dtype=torch.long, device=input_embeds.device).unsqueeze(0).expand(input_embeds.shape[0], -1)

        return position_ids

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            image_grid_hws: Optional[torch.Tensor] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        """AR generation, or VFlash speculative decoding with ``generation_mode='vflash'``.

        VFlash-only keyword arguments: ``vflash_model_path``, ``return_vflash_stats``
        (the result becomes ``(ids, None, stats)``), ``verify_atoms`` (only False is
        supported), ``vflash_text_block_size``, ``vflash_draft_temperature``,
        ``atomic_decoding_tokenizer``, ``initial_prefix_token_ids``, ``bbox_hybrid``
        (``eaglevl.model.vflash.bbox_hybrid.BBoxHybridConfig``) and
        ``vflash_publication_errors`` (``'raise'`` or ``'return'``).
        """
        generation_mode = str(generate_kwargs.pop('generation_mode', 'ar')).lower()
        if generation_mode not in {'ar', 'vflash'}:
            raise ValueError(f'Unsupported Eagle generation_mode: {generation_mode}')
        draft_model_path = generate_kwargs.pop('vflash_model_path', None)
        return_stats = bool(generate_kwargs.pop('return_vflash_stats', False))
        verify_atoms = bool(generate_kwargs.pop('verify_atoms', True))
        bbox_hybrid = generate_kwargs.pop('bbox_hybrid', None)
        initial_prefix_token_ids = generate_kwargs.pop('initial_prefix_token_ids', None)
        publication_errors = generate_kwargs.pop('vflash_publication_errors', 'raise')
        text_block_size = int(generate_kwargs.pop('vflash_text_block_size', 16))
        draft_temperature = float(generate_kwargs.pop('vflash_draft_temperature', 0.0))
        if generation_mode == 'vflash':
            input_embeds = self.prepare_multimodal_inputs_embeds(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_hws=image_grid_hws,
                visual_features=visual_features,
            )
            if draft_model_path is not None:
                self.load_vflash_draft(draft_model_path)
            draft_model = getattr(self, 'vflash_draft', None)
            if draft_model is None:
                raise ValueError(
                    'VFlash decoding requires a standalone drafter; call load_vflash_draft() '
                    'or pass vflash_model_path'
                )
            target_parameter = next(self.language_model.parameters())
            draft_parameter = next(draft_model.parameters())
            if (
                draft_parameter.device != target_parameter.device
                or draft_parameter.dtype != target_parameter.dtype
            ):
                raise ValueError(
                    'VFlash drafter device/dtype must match the Eagle language target'
                )
            from eaglevl.model.vflash.binding import validate_vflash_target_weights
            from eaglevl.model.vflash.generation import (
                dflash_generate,
                resolve_vflash_generation_kwargs,
            )

            validate_vflash_target_weights(self, draft_model.config)
            if output_hidden_states:
                raise NotImplementedError(
                    'VFlash generation returns token IDs/stats, not HF hidden-state outputs'
                )
            atomic_decoding_tokenizer = generate_kwargs.pop('atomic_decoding_tokenizer', None)
            vflash_settings = resolve_vflash_generation_kwargs(
                generate_kwargs,
                generation_config=generation_config,
                default_eos_token_id=self.language_model.config.eos_token_id,
            )
            physical_config = getattr(draft_model.config, 'physical_atom_config', None)
            if physical_config is None:
                physical_config = getattr(self.config, 'physical_atom_config', None)
            result = dflash_generate(
                draft_model=draft_model,
                target_language_model=self.language_model,
                input_ids=input_ids,
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                physical_atom_config=physical_config,
                text_block_size=text_block_size,
                verify_atoms=verify_atoms,
                return_stats=return_stats,
                atomic_decoding_tokenizer=atomic_decoding_tokenizer,
                draft_temperature=draft_temperature,
                bbox_hybrid=bbox_hybrid,
                initial_prefix_token_ids=initial_prefix_token_ids,
                publication_errors=publication_errors,
                **vflash_settings,
            )
            if return_stats:
                return result.output_ids, None, result
            return result
        if bbox_hybrid is not None or initial_prefix_token_ids is not None:
            raise ValueError('The bbox hybrid decoder requires generation_mode="vflash"')
        if not verify_atoms:
            raise ValueError("verify_atoms only applies to generation_mode='vflash'")
        if draft_temperature != 0.0:
            raise ValueError(
                "vflash_draft_temperature only applies to generation_mode='vflash'"
            )
        if publication_errors != 'raise':
            raise ValueError(
                "vflash_publication_errors only applies to generation_mode='vflash'"
            )

        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Convert numpy array to tensor if needed
        if isinstance(image_grid_hws, np.ndarray):
            image_grid_hws = torch.from_numpy(image_grid_hws).to(pixel_values.device, dtype=torch.int32)

        if visual_features is not None:
            vit_embeds = visual_features
        elif pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values, image_grid_hws)

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        if image_grid_hws is not None:
            vit_embeds = torch.cat(vit_embeds, dim=0)
            vit_embeds = self.mlp1(vit_embeds)
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.image_token_index)
            input_embeds[selected] = vit_embeds

        input_embeds = input_embeds.reshape(B, N, C)

        if 'use_cache' not in generate_kwargs:
            generate_kwargs['use_cache'] = True

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            **generate_kwargs,
        )

        return outputs

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_input_embeddings
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_input_embeddings
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_output_embeddings
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_decoder
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_decoder
    def get_decoder(self):
        return self.language_model.get_decoder()
