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
# This file is adopted from the Hugging Face Transformers project
# (https://github.com/huggingface/transformers), licensed under the Apache License, Version 2.0.
#
# Modifications © 2026 NVIDIA CORPORATION & AFFILIATES, licensed under
# the Apache License, Version 2.0.
#
# --------------------------------------------------------
# Hugging Face Transformers
# Copyright 2024 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0
# --------------------------------------------------------

import inspect
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist

from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal, logging
from transformers.integrations.flash_attention import flash_attention_forward as original_flash_attention_forward
flash_241 = is_flash_attn_greater_or_equal("2.4.1")
deterministic_g = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"


logger = logging.get_logger(__name__)


try:
    from eaglevl.sp_utils import ( 
        pre_process_for_sequence_parallel_attn, 
        post_process_for_sequence_parallel_attn, 
        get_pg_manager, 
        split_for_sequence_parallel,
        ring_split_for_sequence_parallel
    )
    from eaglevl.sp_utils.ring import (
        zigzag_ring_flash_attn_varlen_func,
        zigzag_ring_flash_attn_func,
        ring_flash_attn_varlen_func,
        ring_flash_attn_func
    )
except:
    def get_pg_manager():
        return None


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
    ):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.
    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`int`, *optional*):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_sliding_windows (`bool`, *optional*):
            Whether to activate sliding window attention.
    """

    
    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
        causal = is_causal and query_length != 1
    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}
    if flash_241:
        if deterministic is None:
            deterministic = deterministic_g
        flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    enable_sequence_parallel = (
        dist.is_initialized() and get_pg_manager() is not None and get_pg_manager().sequence_parallel_world_size > 1
    )

    if target_dtype is not None:
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    if enable_sequence_parallel:
        query_states, key_states, value_states = pre_process_for_sequence_parallel_attn(query_states, key_states, value_states)
        query_length = query_states.shape[1]

    if attention_mask is None or attention_mask.ndim != 2:
        raise ValueError('Packed FlashAttention requires a 2D integer data-index mask.')
    if attention_mask.shape[-1] != query_length:
        raise ValueError(
            f'Packing mask length ({attention_mask.shape[-1]}) does not match the '
            f'post-all-to-all query length ({query_length}).'
        )
    if os.environ.get('EAGLE_VALIDATE_PACKING_MASK', '0') == '1':
        for row in attention_mask:
            if int(row[0].item()) != 0:
                raise ValueError('Packing mask IDs must start at zero.')
            differences = row[1:] - row[:-1]
            if not torch.all((differences == 0) | (differences == 1)):
                raise ValueError('Packing mask IDs must form contiguous, monotonically increasing segments.')

    # Segment IDs are contiguous and start at zero. ``bincount`` derives every
    # packed length in one device operation; the previous per-segment ``item``
    # loop introduced thousands of CPU/GPU synchronizations at 256K context.
    sub_sample_lengths = [
        torch.bincount(row.to(torch.long)).to(dtype=torch.int32)
        for row in attention_mask
    ]

    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, _indices_q, cu_seq_lens, max_seq_lens = _unpad_input_packing(
            query_states, key_states, value_states, attention_mask, query_length, sub_sample_lengths
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens


        # TODO: currently only support ulysse parallel, need to support ring parallel in the future
        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
            )

        # Packing keeps every token (including the independent dummy segment),
        # so indices_q is the identity and a reshape avoids another 256K-sized
        # gather/copy.
        attn_output = attn_output_unpad.reshape(
            batch_size,
            query_length,
            attn_output_unpad.shape[-2],
            attn_output_unpad.shape[-1],
        )
        if enable_sequence_parallel:
            attn_output = post_process_for_sequence_parallel_attn(attn_output)
        return attn_output
    else:
        raise NotImplementedError("Flash Attention is not available.")

def _get_unpad_data_packing(attention_mask, sub_sample_lengths):
    """
    In this function, we didn't skip padding tokens.
    """
    seqlens_in_batch = torch.cat(
        [lengths.reshape(-1) for lengths in sub_sample_lengths],
        dim=0,
    ).to(device=attention_mask.device, dtype=torch.int32)

    indices = torch.arange(attention_mask.flatten().shape[0], device=attention_mask.device, dtype=torch.int64)
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
    
def _unpad_input_packing(query_layer, key_layer, value_layer, attention_mask, query_length, sub_sample_lengths):
    batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
    _, _, num_heads_q, _ = query_layer.shape
    if kv_seq_len != attention_mask.shape[-1]:
        attention_mask_num_tokens = attention_mask.shape[-1]
        attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data_packing(attention_mask, sub_sample_lengths)
    key_layer = key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim)
    value_layer = value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim)

    if query_length == kv_seq_len:
        query_layer = query_layer.reshape(batch_size * kv_seq_len, num_heads_q, head_dim)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        assert False, "query_length is not equal to kv_seq_len, not considering this case yet."
    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )




from transformers.utils import is_flash_attn_greater_or_equal_2_10
_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


def flash_attention_forward_for_packing(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if attention_mask is None:
        return original_flash_attention_forward(
            module, query, key, value, attention_mask, dropout, scaling, sliding_window, softcap, **kwargs
        )
    
    # This is before the transpose
    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)
    assert attention_mask is not None, f'attention_mask is None, query.shape={query.shape}, key.shape={key.shape}, value.shape={value.shape}'

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=module.is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=target_dtype,
        **kwargs,
    )

    return attn_output, None


def patch_flash_attention_forward_for_lower_hf_version():
    def _flash_attention_forward_for_lower_hf_version(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False):
        window_size = (self.config.sliding_window, self.config.sliding_window) if use_sliding_windows else None
        return _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            is_causal=self.is_causal,
            query_length=query_length,
            dropout=dropout,
            softmax_scale=softmax_scale,
            window_size=window_size,
            use_top_left_mask = self._flash_attn_uses_top_left_mask,
        )
    from transformers.models.qwen2.modeling_qwen2 import Qwen2FlashAttention2
    Qwen2FlashAttention2._flash_attention_forward = _flash_attention_forward_for_lower_hf_version
        



def patch_packing_attention():
    import importlib
    import pkg_resources
    transformers_version = importlib.import_module("transformers").__version__
    if pkg_resources.parse_version(transformers_version) <= pkg_resources.parse_version("4.43.0"):
        patch_flash_attention_forward_for_lower_hf_version()
    else:
        from transformers.modeling_utils import AttentionInterface
        AttentionInterface._global_mapping['flash_attention_2'] = flash_attention_forward_for_packing
        setattr(AttentionInterface, 'flash_attention_2', flash_attention_forward_for_packing)

        # transformers >= 4.55: _preprocess_mask_arguments converts 2D masks to bool,
        # which destroys packing data-index masks ([0,0,1,1,2,2] -> [False,False,True,...]).
        # Patch it to skip the bool conversion for flash_attention_2/3, keeping the raw
        # integer mask that our flash_attention_forward_for_packing expects.
        try:
            import transformers.masking_utils as _masking_utils
            _orig_preprocess = _masking_utils._preprocess_mask_arguments

            def _patched_preprocess_mask_arguments(
                config, input_embeds, attention_mask, cache_position,
                past_key_values, position_ids, layer_idx,
            ):
                if (
                    attention_mask is not None
                    and isinstance(attention_mask, torch.Tensor)
                    and attention_mask.ndim == 2
                    and config._attn_implementation in ("flash_attention_2", "flash_attention_3")
                    and attention_mask.dtype != torch.bool
                ):
                    return True, attention_mask, None, None, None
                return _orig_preprocess(
                    config, input_embeds, attention_mask, cache_position,
                    past_key_values, position_ids, layer_idx,
                )

            _masking_utils._preprocess_mask_arguments = _patched_preprocess_mask_arguments
            logger.info("Patched _preprocess_mask_arguments for packing mask compatibility")
        except (ImportError, AttributeError):
            pass
