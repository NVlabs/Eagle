# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from .modeling_qwen2 import Qwen2ForCausalLM
# from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
import torch.utils.checkpoint as cp
from ..moon_vit.modeling_vit import MoonVitPretrainedModel
from peft import LoraConfig, get_peft_model
from transformers.generation import GenerationMixin
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from .configuration_locateanything import LocateAnythingConfig
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from eaglevl.sp_utils import  (get_pg_manager, ring_split_for_sequence_parallel)
from eaglevl.train.liger_loss_weight_ops import LigerFusedLinearCrossEntropyLoss


logger = logging.get_logger(__name__)


# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L241C1-L280C1
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
    main_input_name = 'input_ids'
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_quantized_cache = True
    _supports_sdpa = True
    
    def _init_weights(self, module):
        std = getattr(self.config, 'initializer_range', None) or self.config.text_config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

IGNORE_INDEX = -100
class LocateAnythingForConditionalGeneration(LocateAnythingPreTrainedModel, GenerationMixin):
    config_class = LocateAnythingConfig
    def __init__(self, config: LocateAnythingConfig, vision_model=None, language_model=None):
        super().__init__(config)

        self.template = config.template
        self.mlp_checkpoint = config.mlp_checkpoint

        logger.info(f'mlp_checkpoint: {self.mlp_checkpoint}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.vision_config.model_type == 'moonvit':
                config.vision_config._attn_implementation = 'flash_attention_2'
                self.vision_model = MoonVitPretrainedModel(config.vision_config)
            else:
                raise ValueError(f'Unsupported vision model type: {config.vision_config.model_type}. Only moonvit is supported.')

        text_attn_impl = (
            getattr(config.text_config, '_attn_implementation', None)
            or getattr(config, '_attn_implementation', None)
            or 'magi'
        )
        config.text_config._attn_implementation = text_attn_impl

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

        # Set _no_split_modules dynamically based on the actual LLM architecture
        arch = config.text_config.architectures[0] if hasattr(config.text_config, 'architectures') and config.text_config.architectures else 'Qwen2ForCausalLM'
        if 'Qwen3' in arch:
            self._no_split_modules = ["Qwen3DecoderLayer"]
        else:
            self._no_split_modules = ["Qwen2DecoderLayer"]

        
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
            loss_weight: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            sub_sample_lengths: Optional[List[torch.Tensor]] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        RING_ZIGZAG = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if sub_sample_lengths is None:
            sub_sample_lengths = self.get_sub_sample_lengths(input_ids)

        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        has_images = image_flags is not None and image_flags.sum() > 0
        
        vit_embeds = self.extract_feature(pixel_values, image_grid_hws)
            
        B, N, C = input_embeds.shape
        # LoRA's input-gradient hook can make embedding outputs leaf tensors.
        # Clone before indexed writes so the same path works with and without LoRA.
        input_embeds = input_embeds.reshape(B * N, C).clone()

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
            n_token = int(selected.sum().item())
            n_embed = vit_embeds.shape[0]
            if n_embed == n_token:
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds
                ignore_flag = False
            else:
                print(f'warning: image token/feature mismatch, input_embeds[selected].shape={input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                n_assign = min(n_token, n_embed)
                if n_assign > 0:
                    selected_indices = selected.nonzero(as_tuple=False).squeeze(1)[:n_assign]
                    input_embeds[selected_indices] = input_embeds[selected_indices] * 0.0 + vit_embeds[:n_assign]
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
        
        ssl_tensor = None
        if sub_sample_lengths is not None:
            ssl = sub_sample_lengths[0] if isinstance(sub_sample_lengths, list) else sub_sample_lengths
            total_packed_len = int(ssl.sum().item()) if isinstance(ssl, torch.Tensor) else int(sum(ssl))
            seq_len = int(input_ids.shape[-1])
            if total_packed_len != seq_len:
                raise ValueError(
                    f"Packed sequence length mismatch: seq_len={seq_len}, "
                    f"sum(sub_sample_lengths)={total_packed_len}, "
                    f"sub_sample_lengths={ssl.tolist() if isinstance(ssl, torch.Tensor) else ssl}"
                )
            if len(ssl) > 1:  # Multiple samples packed together
                ssl_tensor = ssl

        outputs = language_model_forward(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            sub_sample_lengths=ssl_tensor,  # Pass sub_sample_lengths for stream packing
            )  
        
        # not every token needs to be computed by lm_head, we only compute the tokens that have valid labels
        hidden_states = outputs.last_hidden_state
        lm_head_weight = self.language_model.lm_head.weight
        
        hidden_dim = hidden_states.shape[-1]

        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_hidden_states = shift_hidden_states.view(-1, hidden_dim)
        shift_labels = shift_labels.view(-1)
        valid_shift_labels = int(shift_labels.ne(IGNORE_INDEX).sum().item())
        if valid_shift_labels == 0:
            raise ValueError(
                f"No valid shifted labels in packed batch: labels_shape={tuple(labels.shape)}, "
                f"input_ids_shape={tuple(input_ids.shape)}, "
                f"sub_sample_lengths={ssl.tolist() if 'ssl' in locals() and isinstance(ssl, torch.Tensor) else ssl if 'ssl' in locals() else None}"
            )

        # Process loss_weight: shift it like labels and flatten
        shift_loss_weight = None
        if loss_weight is not None:
            shift_loss_weight = loss_weight[..., 1:].contiguous()
            shift_loss_weight = shift_loss_weight.view(-1)

        liger_loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='mean')
        loss = liger_loss_fn(lm_head_weight, shift_hidden_states, shift_labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss detected before backward: loss={loss.detach().float().item()}, "
                f"valid_shift_labels={valid_shift_labels}, "
                f"hidden_states_has_nan={bool(torch.isnan(shift_hidden_states).any().item())}, "
                f"hidden_states_has_inf={bool(torch.isinf(shift_hidden_states).any().item())}"
            )
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
