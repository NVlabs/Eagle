# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, is_flash_attn_2_available, logging
from peft import LoraConfig, get_peft_model

from .configuration_locateanything import LocateAnythingConfig
from .modeling_qwen2 import Qwen2ForCausalLM
from .modeling_vit import MoonVitPretrainedModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from .mask_sdpa_utils import *
from .mask_magi_utils import *
from .configuration_qwen2 import Qwen2Config

from .generate_utils import (
    sample_tokens,
    handle_pattern,
    get_token_ids_from_config,
)

logger = logging.get_logger(__name__)


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

    @classmethod
    def _autoset_attn_implementation(cls, config, *args, **kwargs):
        if getattr(config, '_attn_implementation', None) == 'magi':
            return config
        return super()._autoset_attn_implementation(config, *args, **kwargs)

    def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False):
        if attn_implementation == "magi":
            return "magi"
        return super()._check_and_adjust_attn_implementation(attn_implementation, is_init_check)
    
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
                vision_attn_impl = getattr(config.vision_config, '_attn_implementation', None) or 'flash_attention_2'
                if vision_attn_impl == 'flash_attention_2' and not is_flash_attn_2_available():
                    logger.warning_once(
                        "flash_attn is not available for MoonViT inference; falling back to sdpa."
                    )
                    vision_attn_impl = 'sdpa'
                config.vision_config._attn_implementation = vision_attn_impl
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

        self.token_ids = get_token_ids_from_config(config)

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
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:selected.sum()]
        else:
            if vit_embeds:
                vit_embeds = torch.cat(vit_embeds, dim=0)
                vit_embeds = self.mlp1(vit_embeds)
                input_ids = input_ids.reshape(B * N)
                selected = (input_ids == self.image_token_index)
                if selected.sum() > 0:
                    input_embeds[selected] = vit_embeds[:selected.sum()]

        input_embeds = input_embeds.reshape(B, N, C)

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
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        image_grid_hws: Optional[torch.Tensor] = None,
        tokenizer = None, 
        n_future_tokens: int = 6,
        **generate_kwargs,
    ) -> torch.LongTensor:

        verbose = generate_kwargs.pop('verbose', False)
        start_time = time.time()
        prefill_time = None

        pixel_values = pixel_values.to(self.language_model.dtype)
        # Convert numpy array to tensor if needed
        if isinstance(image_grid_hws, np.ndarray):
            image_grid_hws = torch.from_numpy(image_grid_hws).to(pixel_values.device, dtype=torch.int32)

        batch_size, seq_len = input_ids.shape
        assert batch_size == 1, 'only batch size = 1 is supported now'
        assert generate_kwargs.get('use_cache', False), "Only use_cache=True is supported."

        generated = input_ids.clone()
        total_gen_length = min(tokenizer.model_max_length, seq_len + generate_kwargs.get('max_new_tokens', 2048))
        iter_round = 0
        past_key_values = None

        # Extract visual features once before the loop
        if visual_features is not None:
            vit_embeds = visual_features
        elif pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values, image_grid_hws)
        else:
            vit_embeds = None
        
        if image_grid_hws is not None:
            vit_embeds = torch.cat(vit_embeds, dim=0)
            vit_embeds = self.mlp1(vit_embeds)

        # ==================== Generation Mode ====================
        # 'fast'   : MTP only, never fall back to AR
        # 'slow'   : AR only, pure auto-regressive decoding
        # 'hybrid' : MTP first, fall back to AR on error, switch back on box_end
        generation_mode = generate_kwargs.get('generation_mode', 'hybrid')
        assert generation_mode in ('fast', 'slow', 'hybrid'), \
            f"Unsupported generation_mode='{generation_mode}'. Use 'fast', 'slow', or 'hybrid'."

        sampling_history = []


        use_mtp = generation_mode in ('fast', 'hybrid')
        switch_to_ar_count = 0

        # Pre-allocate mask tokens and position ids
        default_mask_token_id = self.token_ids['default_mask_token_id']
        pre_mask_tokens = torch.full(
            (batch_size, n_future_tokens - 1),
            default_mask_token_id,
            dtype=generated.dtype,
            device=generated.device
        )
        max_possible_len = total_gen_length + n_future_tokens
        full_position_ids = torch.arange(0, max_possible_len, device=generated.device).unsqueeze(0)


        def _prepare_inputs_in_mtp(generated):
            generated_with_mask = torch.cat(
                (
                    generated, 
                    generated[:, -1].unsqueeze(1),
                    pre_mask_tokens
                ), 
                dim=1
            ) # [batch_size, seq_len + 1 +  n_future_tokens - 1]

            # Update pe for kvcache
            start_idx = past_key_values[0][0].size(2) if past_key_values is not None else 0
            position_ids = full_position_ids[:, start_idx : generated_with_mask.size(1)].clone()
            position_ids[0, -n_future_tokens:] -= 1

            prepare_inputs = self.language_model.prepare_inputs_for_generation(
                generated_with_mask,
                past_key_values,
                None,
                inputs_embeds=None,
                use_cache=True,
                position_ids=position_ids
            )
            return prepare_inputs


        def _prepare_input_in_ar(generated):
            start_idx = past_key_values[0][0].size(2) if past_key_values is not None else 0
            position_ids = full_position_ids[:, start_idx : generated.size(1)]
            prepare_inputs = self.language_model.prepare_inputs_for_generation(
                generated,
                past_key_values,
                None,
                inputs_embeds=None,
                use_cache=True,
                position_ids=position_ids
            )
            return prepare_inputs


        def _sample_token_in_mtp(generated, outputs):
            """Sample tokens using MTP (Multi-Token Prediction) mode."""
            next_token_logits = outputs.logits[:, -n_future_tokens:, :]
            probs, confidence, x0, box_avg = sample_tokens(
                next_token_logits, generated, self.token_ids, keep_k=5, **generate_kwargs
            )

            is_box_empty = (box_avg[0] == 0).all()
            new_tokens = x0[0] if is_box_empty else box_avg[0]

            out_pattern = handle_pattern(new_tokens, self.token_ids, generation_mode)
            out_type = out_pattern['type']
            out_token = torch.tensor(out_pattern['tokens'], dtype=x0.dtype, device=x0.device)

            return out_type, out_token


        def _sample_token_in_ar(generated, outputs):
            """Sample a single token using AR (Auto-Regressive) mode."""
            next_token_logits = outputs.logits[:, -1:, :]
            probs, confidence, x0, _ = sample_tokens(
                next_token_logits, generated, self.token_ids, **generate_kwargs
            )

            out_token = x0[0]
            out_type = 'continue_ar'
            token_val = out_token[0].item()

            box_end_token_id = self.token_ids['box_end_token_id']
            coord_start_token_id = self.token_ids['coord_start_token_id']
            coord_end_token_id = self.token_ids['coord_end_token_id']
            none_token_id = self.token_ids['none_token_id']
            im_end_token_id = self.token_ids['im_end_token_id']

            if generation_mode == 'hybrid':
                # Hybrid AR phase: detect box boundaries to switch back to MTP
                if token_val == box_end_token_id:
                    out_type = 'box_end_ar'
                elif coord_start_token_id <= token_val <= coord_end_token_id or token_val == none_token_id:
                    out_type = 'coord_ar'
                else:
                    out_type = 'im_end'
            else:
                # Slow mode: pure AR, only stop on im_end
                if token_val == im_end_token_id:
                    out_type = 'im_end'

            return out_type, out_token


        # Generate loop
        while generated.size(1) < total_gen_length:
            iter_round += 1

            # Step 1: Prepare inputs
            if use_mtp:
                prepare_inputs = _prepare_inputs_in_mtp(generated)
            else:
                prepare_inputs = _prepare_input_in_ar(generated)

            if iter_round == 1:
                prepare_inputs.update({
                    'visual_features': vit_embeds,
                    'image_token_index': self.config.image_token_index,
                })

            # Step 2: Model forward & update KV cache
            with torch.no_grad():
                outputs = self.language_model(**prepare_inputs)

            past_key_values = tuple(
                (kv[0][:, :, :generated.shape[1], :], kv[1][:, :, :generated.shape[1], :])
                for kv in outputs.past_key_values
            )

            # Step 3: Sample tokens
            if use_mtp:
                out_type, out_token = _sample_token_in_mtp(generated, outputs)
            else:
                out_type, out_token = _sample_token_in_ar(generated, outputs)

            if verbose:
                sampling_history.append(('ar' if 'ar' in out_type else 'mtp', tokenizer.decode(out_token, skip_special_tokens=False)))

            generated = torch.cat([generated, out_token.unsqueeze(0)], dim=1)

            # Step 4: Mode switching & termination
            if out_type == 'im_end':
                break

            if generation_mode == 'hybrid':
                if out_type == 'error_box':
                    use_mtp = False
                    switch_to_ar_count += 1
                elif out_type == 'box_end_ar':
                    use_mtp = True
            # fast mode: use_mtp stays True always
            # slow mode: use_mtp stays False always

            if prefill_time is None:
                prefill_time = time.time() - start_time

        # Decode and return
        generated_ids = generated[:, seq_len:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        if verbose:
            end_time = time.time()
            num_tokens = generated_ids.size(1)
            num_boxes = response[0].count("<box>")
            total_time = end_time - start_time

            out_info =  f"\nStatistic Info, num_tokens={num_tokens}; " + \
                    f"generate_time(s)={total_time:.4f}; " + \
                    f"tps={(num_tokens / total_time):.4f}; " + \
                    f"forward_step={iter_round}; " + \
                    f"num_boxes={num_boxes}; " + \
                    f"bps={(num_boxes / total_time):.4f}; " + \
                    f"prefill_time={(prefill_time):.4f}; " + \
                    f"switch_to_ar={switch_to_ar_count}\n"
            print(out_info)

            return response[0], sampling_history, out_info

        return response[0]
