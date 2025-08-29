
# EagleVL
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
import inspect
from typing import Any, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import torch.utils.checkpoint as cp
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from peft import LoraConfig, get_peft_model
from transformers.generation import GenerationMixin
from transformers import (AutoModel, GenerationConfig)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from eaglevl.model.eagle2_5.configuration_eagle2_5_vl import Eagle2_5_VLConfig
from eaglevl.model.c_radio.radio_model import RADIOModel, RADIOConfig
from eaglevl.sp_utils import  (get_pg_manager, split_for_sequence_parallel, ring_split_for_sequence_parallel,
                                gather_from_sequence_parallel, ring_gather_for_sequence_parallel,
                                LowMemLogitProjCrossEnt)

from eaglevl.train.liger_loss_weight_ops import LigerFusedLinearCrossEntropyLoss
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings

logger = logging.get_logger(__name__)

def find_sequences_torch(arr):
    mask = (arr != IGNORE_INDEX)
    
    mask_int = mask.int()

    diff = mask_int[1:] - mask_int[:-1]
    start_indices = (diff == 1).nonzero(as_tuple=False).flatten() + 1
    end_indices = (diff == -1).nonzero(as_tuple=False).flatten()
    if len(mask)==0: return []
    if mask[0]:
        start_indices = torch.cat((torch.tensor([0], device=arr.device), start_indices))
    if mask[-1]:
        end_indices = torch.cat((end_indices, torch.tensor([len(arr) - 1], device=arr.device)))
    sequences = list(zip(start_indices.tolist(), end_indices.tolist()))
    return sequences


def pre_calc_loss_weight(num_samples, acc_lengths, shift_labels, loss_version):    
    loss_weight = torch.ones(shift_labels.shape[0], device=shift_labels.device, dtype=torch.float32)
    num_valid_labels_list = []
    loss_weight[shift_labels==IGNORE_INDEX] = 0
    all_num_valid_labels = (shift_labels!=IGNORE_INDEX).sum()
    for sample_idx in range(num_samples):
        weight_this_sample = loss_weight[acc_lengths[sample_idx]: acc_lengths[sample_idx+1]]
        shift_labels_this_sample = shift_labels[acc_lengths[sample_idx]:acc_lengths[sample_idx+1]]
        if "multi_turn_scale" in loss_version:
            turn_start_end_list = find_sequences_torch(shift_labels_this_sample)
            for turn_start, turn_end in turn_start_end_list:
                num_valid_labels_this_turn = torch.tensor(turn_end-turn_start+1, device=weight_this_sample.device, dtype=weight_this_sample.dtype)
                weight_this_sample[turn_start:turn_end+1] *= num_valid_labels_this_turn.rsqrt()
                num_valid_labels_list.append(num_valid_labels_this_turn)
        else:
            num_valid_labels = (shift_labels_this_sample!=IGNORE_INDEX).sum(-1)
            if num_valid_labels > 0:
                weight_this_sample *= num_valid_labels.rsqrt()
                num_valid_labels_list.append(num_valid_labels)
        loss_weight[acc_lengths[sample_idx]: acc_lengths[sample_idx+1]] = weight_this_sample
    base_num = torch.stack(num_valid_labels_list).sqrt().sum()
    loss_weight = loss_weight / base_num
    return loss_weight


# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L241C1-L280C1
EAGLE2_5_VL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Eagle2_5_VLConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

@add_start_docstrings(
    "The bare Eagle2_5_VL Model outputting raw hidden-states without any specific head on top.",
    EAGLE2_5_VL_START_DOCSTRING,
)
class Eagle2_5_VLPreTrainedModel(PreTrainedModel):
    config_class = Eagle2_5_VLConfig
    base_model_prefix = "model"
    main_input_name = 'input_ids'
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer", "LlamaDecoderLayer" ,"Siglip2EncoderLayer", "SiglipEncoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
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
class Eagle2_5_VLForConditionalGeneration(Eagle2_5_VLPreTrainedModel, GenerationMixin):
    config_class = Eagle2_5_VLConfig
    def __init__(self, config: Eagle2_5_VLConfig, vision_model=None, language_model=None):
        super().__init__(config)

        image_size = config.force_image_size or config.vision_config.image_size

        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        
        if config.use_pixel_shuffle:
            self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        else:
            self.num_image_token = int((image_size // patch_size) ** 2)

        self.select_layer = config.select_layer
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio
        self.loss_version = config.loss_version
        self.mlp_checkpoint = config.mlp_checkpoint
        self.use_pixel_shuffle = config.use_pixel_shuffle
        self.mlp_connector_layers = config.mlp_connector_layers
        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'mlp_checkpoint: {self.mlp_checkpoint}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.vision_config.model_type == 'intern_vit_6b':
                self.vision_model = InternVisionModel(config.vision_config)
            elif config.vision_config.model_type == 'siglip_vision_model':
                config.vision_config._attn_implementation = 'flash_attention_2'
                self.vision_model = SiglipVisionModel(config.vision_config)
            elif config.vision_config.model_type == 'radio':
                self.vision_model = RADIOModel(config.vision_config)

        if language_model is not None:
            self.language_model = language_model
        else:
            if config.text_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Qwen2ForCausalLM':
                assert config.text_config._attn_implementation == 'flash_attention_2', f"Qwen2 must use flash_attention_2 but got {config.text_config._attn_implementation}"
                self.language_model = Qwen2ForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Qwen3ForCausalLM':
                assert config.text_config._attn_implementation == 'flash_attention_2', f"Qwen3 must use flash_attention_2 but got {config.text_config._attn_implementation}"
                self.language_model = Qwen3ForCausalLM(config.text_config)
            else:
                raise NotImplementedError(f'{config.text_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        if config.mlp_connector_layers == 2:
            self.mlp1 = nn.Sequential(
                nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
                nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
        elif config.mlp_connector_layers == 1 and config.use_pixel_shuffle:
            self.mlp1 = nn.Sequential(
                nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            )
        elif config.mlp_connector_layers == 1 and not config.use_pixel_shuffle:
            self.mlp1 = nn.Sequential(
                nn.Linear(vit_hidden_size, llm_hidden_size),
            )
        else:
            raise NotImplementedError(f'{config.mlp_connector_layers} is not implemented.')

        self.image_token_index = config.image_token_index
        self.neftune_alpha = None


        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        self.use_llm_lora = config.use_llm_lora 
        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)
            
        self.check_forward_kwargs()
        
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

    def get_sub_sample_lengths(self, input_ids):
        # for compatibility with packing
        sub_sample_lengths = [torch.tensor([each.shape[0]], device=input_ids.device, dtype=torch.int32) for each in input_ids]
        return sub_sample_lengths

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            sub_sample_lengths=None
            # **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        RING_ZIGZAG = False
        assert attention_mask is not None, f'attention_mask is None, input_ids.shape={input_ids.shape}, sub_sample_lengths={sub_sample_lengths}, image_flags={image_flags}'

        if sub_sample_lengths is None:
            sub_sample_lengths = self.get_sub_sample_lengths(input_ids)
            
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        num_images = pixel_values.shape[0]
        
        # scatter the pixel_values to the sequence parallel group
        if get_pg_manager() is not None:
            num_images = pixel_values.shape[0]
            pixel_values = split_for_sequence_parallel(pixel_values, dim=0, sp_group=get_pg_manager().sequence_parallel_group, fill_zeros=True)
        

        vit_embeds = self.extract_feature(pixel_values)

        # gather the vit_embeds from the sequence parallel group
        if get_pg_manager() is not None:
            vit_embeds = gather_from_sequence_parallel(vit_embeds, dim=0, sp_group=get_pg_manager().sequence_parallel_group)
            vit_embeds = vit_embeds[:num_images]

        if not isinstance(image_flags, list):
            image_flags = image_flags.squeeze(-1)
            vit_embeds = vit_embeds[image_flags == 1]
        
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.image_token_index)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            # vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)


        if position_ids is None:
            position_ids = self._generate_position_ids(input_embeds, sub_sample_lengths)

        if get_pg_manager() is not None:
            local_position_ids = ring_split_for_sequence_parallel(position_ids, ulysses_group=get_pg_manager().ulysses_sequence_parallel_group, 
                ring_group=get_pg_manager().ring_sequence_parallel_group, sub_sample_lengths=sub_sample_lengths, ring_zigzag=RING_ZIGZAG) 
            local_input_embeds = ring_split_for_sequence_parallel(input_embeds, ulysses_group=get_pg_manager().ulysses_sequence_parallel_group, 
                ring_group=get_pg_manager().ring_sequence_parallel_group, sub_sample_lengths=sub_sample_lengths, ring_zigzag=RING_ZIGZAG)  
            local_attention_mask = ring_split_for_sequence_parallel(attention_mask, ulysses_group=get_pg_manager().ulysses_sequence_parallel_group, 
                ring_group=get_pg_manager().ring_sequence_parallel_group, sub_sample_lengths=sub_sample_lengths, split_by_ulysses=False, ring_zigzag=RING_ZIGZAG)
        else:
            local_position_ids = position_ids
            local_input_embeds = input_embeds
            local_attention_mask = attention_mask

        if self.use_llm_lora:
            language_model_forward = self.language_model.model.model.forward
        else:
            language_model_forward = self.language_model.model.forward
        
        
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
        if get_pg_manager() is None:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_labels = torch.roll(labels, shifts=-1, dims=-1).contiguous()
            shift_labels[:, -1] = IGNORE_INDEX
            shift_hidden_states = hidden_states
        
        shift_hidden_states = shift_hidden_states.view(-1, hidden_dim)
        shift_labels = shift_labels.view(-1)
        lengths_list = torch.cat(sub_sample_lengths)
        acc_lengths = torch.cumsum(lengths_list, dim=0)
        acc_lengths = torch.cat([torch.tensor([0], device=acc_lengths.device, dtype=acc_lengths.dtype), acc_lengths] )
        num_samples = len(lengths_list)
        loss_weight = pre_calc_loss_weight(num_samples, acc_lengths, shift_labels, self.loss_version)
        if get_pg_manager() is not None:
            loss_weight = ring_split_for_sequence_parallel(loss_weight, ulysses_group=get_pg_manager().ulysses_sequence_parallel_group, 
                ring_group=get_pg_manager().ring_sequence_parallel_group, sub_sample_lengths=sub_sample_lengths, ring_zigzag=RING_ZIGZAG)
            shift_labels = ring_split_for_sequence_parallel(shift_labels, ulysses_group=get_pg_manager().ulysses_sequence_parallel_group, 
                ring_group=get_pg_manager().ring_sequence_parallel_group, sub_sample_lengths=sub_sample_lengths, ring_zigzag=RING_ZIGZAG)
        liger_loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='sum')
        loss = liger_loss_fn(lm_head_weight, shift_hidden_states, shift_labels, loss_weight=loss_weight)
        logits = None

        if get_pg_manager() is not None:
            loss_list = [torch.zeros_like(loss, device=loss.device, dtype=loss.dtype) for _ in range(dist.get_world_size(get_pg_manager().sequence_parallel_group))]
            dist.all_gather(loss_list, loss, group=get_pg_manager().sequence_parallel_group)
            loss_list[dist.get_rank(get_pg_manager().sequence_parallel_group)] = loss
            loss = torch.stack(loss_list).sum()

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

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))

        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def noised_embed(self, vit_embeds, noise_alpha=5):
        dims = torch.tensor(vit_embeds.size(1) * vit_embeds.size(2))
        mag_norm = noise_alpha / torch.sqrt(dims)
        noise = torch.zeros_like(vit_embeds).uniform_(-mag_norm, mag_norm)
        return vit_embeds + noise

    
    def feature_compression(self, vit_embeds):
        if self.use_pixel_shuffle:
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio) # torch.Size([B, 1024, 1024]) -> torch.Size([B, 16, 16, 4096])
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1]) # torch.Size([B, 16, 16, 4096]) -> torch.Size([B, 256, 4096])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True)
            if hasattr(vit_embeds, 'last_hidden_state'):
                vit_embeds = vit_embeds.last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        

        if self.training and self.neftune_alpha is not None:
            vit_embeds = self.noised_embed(vit_embeds, self.neftune_alpha)


        if self.mlp_checkpoint and vit_embeds.requires_grad:
            vit_embeds = cp.checkpoint(self.feature_compression, vit_embeds)
        else:
            vit_embeds = self.feature_compression(vit_embeds)


        return vit_embeds


    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_tiles_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_tiles_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_tiles_list` instead.')

        image_token_index = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.image_token_index = image_token_index

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_tiles in enumerate(num_tiles_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_tiles + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_tiles_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_tiles_list is None:
            num_tiles_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_tiles_list)

        image_token_index = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.image_token_index = image_token_index

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_tiles in num_tiles_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_tiles + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id

        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            image_sizes: Optional[List[Tuple[int, int]]] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.config.image_token_index)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
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

