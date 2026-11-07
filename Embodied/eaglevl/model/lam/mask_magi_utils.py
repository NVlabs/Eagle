# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
MagiAttention Mask Utilities for MTP Training and Inference.

Provides range-based sparse attention plan construction for MagiAttention
(flex_flash_attn) used in both training (packing) and inference (decode).
"""
import torch
from typing import Dict, Tuple

from .mask_sdpa_utils import _find_sample_x0_len_packed

FULL, CAUSAL = 0, 1


def build_magi_ranges(kv_len: int, q_len: int, block_size: int, ar_decode: bool = False, device: str = "cpu"):
    """Build MagiAttention range plan for inference decode steps.

    Args:
        kv_len: Total key/value sequence length (including cache).
        q_len: Current query length.
        block_size: MTP block size.
        ar_decode: If True, use simple causal AR decoding.
        device: Target device.

    Returns:
        Dict with q_ranges, k_ranges, attn_type_map tensors.
    """
    assert 0 < q_len <= kv_len

    if ar_decode:
        if q_len == kv_len:
            return {
                "q_ranges": torch.tensor([[0, q_len]], dtype=torch.int32, device=device).contiguous(),
                "k_ranges": torch.tensor([[0, kv_len]], dtype=torch.int32, device=device).contiguous(),
                "attn_type_map": torch.tensor([CAUSAL], dtype=torch.int32, device=device).contiguous(),
            }

        prefix_len = kv_len - q_len
        q_ranges, k_ranges, types = [], [], []

        if prefix_len > 0:
            q_ranges.append([0, q_len])
            k_ranges.append([0, prefix_len])
            types.append(FULL)

        q_ranges.append([0, q_len])
        k_ranges.append([prefix_len, kv_len])
        types.append(CAUSAL)

        return {
            "q_ranges": torch.tensor(q_ranges, dtype=torch.int32, device=device).contiguous(),
            "k_ranges": torch.tensor(k_ranges, dtype=torch.int32, device=device).contiguous(),
            "attn_type_map": torch.tensor(types, dtype=torch.int32, device=device).contiguous(),
        }

    assert 0 < block_size <= q_len <= kv_len
    B = block_size
    r = q_len - B
    q_global_start = kv_len - q_len

    window_start_k = kv_len - B
    blocked_k = window_start_k - 1

    q_ranges, k_ranges, types = [], [], []

    if q_len == kv_len:
        prefix_len = window_start_k

        if prefix_len > 0:
            q_ranges += [[0, prefix_len]]
            k_ranges += [[0, prefix_len]]
            types += [CAUSAL]

        if prefix_len > 0 and blocked_k > 0:
            q_ranges += [[prefix_len, kv_len]]
            k_ranges += [[0, blocked_k]]
            types += [FULL]

        q_ranges += [[prefix_len, kv_len]]
        k_ranges += [[prefix_len, kv_len]]
        types += [FULL]

        return {
            "q_ranges": torch.tensor(q_ranges, dtype=torch.int32, device=device).contiguous(),
            "k_ranges": torch.tensor(k_ranges, dtype=torch.int32, device=device).contiguous(),
            "attn_type_map": torch.tensor(types, dtype=torch.int32, device=device).contiguous(),
        }

    for i in range(r):
        g = q_global_start + i
        q_ranges.append([i, i + 1])
        k_ranges.append([0, g + 1])
        types.append(FULL)

    q_win = [r, q_len]

    if blocked_k > 0:
        q_ranges.append(q_win)
        k_ranges.append([0, blocked_k])
        types.append(FULL)

    q_ranges.append(q_win)
    k_ranges.append([window_start_k, kv_len])
    types.append(FULL)

    return {
        "q_ranges": torch.tensor(q_ranges, dtype=torch.int32, device=device).contiguous(),
        "k_ranges": torch.tensor(k_ranges, dtype=torch.int32, device=device).contiguous(),
        "attn_type_map": torch.tensor(types, dtype=torch.int32, device=device).contiguous(),
    }


@torch.no_grad()
def convert_mtp_mask_to_magi_plan(
    block_size: int,
    position_ids: torch.Tensor,
    data_index: torch.Tensor,
    causal_attn: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Converts MTP packing mask logic to MagiAttention flexible attention plan.
    
    MagiAttention uses range-based sparse attention patterns:
    - q_ranges/k_ranges define which query/key ranges attend to each other
    - attn_type_map specifies causal(1) or full(0) attention for each range pair
    
    Args:
        block_size: Size of each MTP prediction block.
        position_ids: [1, seq_len] or [seq_len] position IDs.
        data_index: [1, seq_len] or [seq_len] sample index.
        causal_attn: If True, use causal attention within MTP blocks.
        
    Returns:
        Dict containing q_ranges, k_ranges, attn_type_map, max_seqlen_q, max_seqlen_k.
    """
    if position_ids.dim() == 2:
        position_ids = position_ids.squeeze(0)
    if data_index.dim() == 2:
        data_index = data_index.squeeze(0)
        
    device = position_ids.device
    seq_len = position_ids.numel()
    
    x0_lens, sample_starts, sample_ends = _find_sample_x0_len_packed(
        position_ids.unsqueeze(0), data_index.unsqueeze(0)
    )
    
    num_samples = len(sample_starts)
    
    q_ranges_list = []
    k_ranges_list = []
    attn_types_list = []
    
    max_seqlen_q = 0
    max_seqlen_k = 0
    
    position_ids_cpu = position_ids.cpu()
    sample_starts_cpu = sample_starts.cpu().tolist()
    sample_ends_cpu = sample_ends.cpu().tolist()
    x0_lens_cpu = x0_lens.cpu().tolist()
    
    for i in range(num_samples):
        s_start = sample_starts_cpu[i]
        s_end = sample_ends_cpu[i]
        x0_len = x0_lens_cpu[i]
        
        if x0_len > 0:
            x0_end = s_start + x0_len
            q_ranges_list.append([s_start, x0_end])
            k_ranges_list.append([s_start, x0_end])
            attn_types_list.append(1)
            
            max_seqlen_q = max(max_seqlen_q, x0_len)
            max_seqlen_k = max(max_seqlen_k, x0_len)
            
        mtp_start = s_start + x0_len
        if mtp_start < s_end:
            curr_start = mtp_start
            while curr_start < s_end:
                curr_end = min(curr_start + block_size, s_end)
                block_len = curr_end - curr_start
                
                q_ranges_list.append([curr_start, curr_end])
                k_ranges_list.append([curr_start, curr_end])
                attn_types_list.append(1 if causal_attn else 0)
                
                max_seqlen_q = max(max_seqlen_q, block_len)
                max_seqlen_k = max(max_seqlen_k, block_len)
                
                prefix_len = position_ids_cpu[curr_start].item()
                prefix_len = min(prefix_len, x0_len)
                
                if prefix_len > 0:
                    prefix_end = s_start + prefix_len
                    q_ranges_list.append([curr_start, curr_end])
                    k_ranges_list.append([s_start, prefix_end])
                    attn_types_list.append(0)
                    
                    max_seqlen_k = max(max_seqlen_k, prefix_len)

                curr_start += block_size

    if not q_ranges_list:
        return {
            "q_ranges": torch.zeros((0, 2), dtype=torch.int32, device=device),
            "k_ranges": torch.zeros((0, 2), dtype=torch.int32, device=device),
            "attn_type_map": torch.zeros((0,), dtype=torch.int32, device=device),
            "max_seqlen_q": 0,
            "max_seqlen_k": 0,
        }

    return {
        "q_ranges": torch.tensor(q_ranges_list, dtype=torch.int32, device=device).contiguous(),
        "k_ranges": torch.tensor(k_ranges_list, dtype=torch.int32, device=device).contiguous(),
        "attn_type_map": torch.tensor(attn_types_list, dtype=torch.int32, device=device).contiguous(),
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
