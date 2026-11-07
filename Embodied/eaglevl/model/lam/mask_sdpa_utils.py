# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
SDPA / Eager Attention Mask Utilities for MTP Training and Inference.

Provides 2D and 4D attention mask construction for PyTorch SDPA and
explicit matmul attention implementations.
"""
import torch
from typing import Tuple


def find_prefix_seq_length_by_pe(
    pe: torch.Tensor
) -> torch.Tensor:
    """
    Find the sequence length where position encoding drops (indicating prefix boundary).
    Args:
        pe: Position encoding tensor of shape [Batch size, Sequence length ]
            Contains position indices for each token in the sequence.
    Returns:
        torch.Tensor: A tensor of shape [B] containing:
            - The index where position encoding drops for each sequence
            - -1 if no drop occurs in the sequence
    """
    batch_size, seq_len = pe.shape
    prev = pe[:, :-1]
    curr = pe[:, 1:]
    drop_mask = curr < prev  #  [batch_size, seq_len-1]

    seq_len = torch.full((batch_size,), -1, dtype=torch.long)
    
    for b in range(batch_size):
        drop_pos = torch.nonzero(drop_mask[b], as_tuple=False)
        if drop_pos.numel() > 0:
            i = drop_pos[0].item() + 1  # Take first drop position (+1 because we compared shifted sequences)
            seq_len[b] = i

    return seq_len


@torch.no_grad()
def update_causal_mask_with_pad_non_visible_2d(
    input_ids: torch.Tensor,
    attn_mask_2d: torch.Tensor,
    text_mask_token_id: int = 151666,
    block_size: int = 4,
    causal_attn: bool = False
) -> torch.Tensor:
    """
    Updates a 2D attention mask for hole sequence through input_ids and text_mask_token_id

    Args:
        input_ids: Input token IDs (unused in current implementation)
        attn_mask_2d: 2D attention mask matrix of shape [seq_len, seq_len] where:
            - 0.0 indicates allowed attention
            - -inf indicates masked attention
        text_mask_token_id: ID representing masked tokens
        block_size: Size of the diffusion window
        causal_attn: If True, maintains strict causal masking throughout
    
    Returns:
        Modified attention mask with updated visibility patterns
    """
    seq_len = input_ids.shape[0]
    device = input_ids.device

    input_mask = input_ids.eq(text_mask_token_id)
    input_before_mask = torch.zeros_like(input_mask)
    input_before_mask[:-1] = input_mask[1:]
    mask_cols = (input_mask | input_before_mask)
    non_mask = ~mask_cols

    rows = torch.arange(seq_len, device=device)[:, None]
    cols = torch.arange(seq_len, device=device)

    indices = torch.arange(seq_len, device=device)
    prev_non_mask = (indices * non_mask).cummax(dim=0).values

    max_value = torch.iinfo(indices.dtype).max
    mask_indices = torch.where(non_mask, indices, torch.full_like(indices, max_value))
    reversed_mask_indices = torch.flip(mask_indices, dims=[0])
    reversed_cummin = reversed_mask_indices.cummin(dim=0).values
    next_non_mask = torch.flip(reversed_cummin, dims=[0])

    infra_mask = (
            (cols > prev_non_mask) &
            (rows >= next_non_mask[None, :]) &
            mask_cols[None, :]
    )
    attn_mask_2d.masked_fill_(infra_mask, -float('inf'))

    if not causal_attn:
        visible_mask = (
                (rows > prev_non_mask[None, :]) &  
                (rows < cols) &  
                mask_cols[None, :]  
        )
        attn_mask_2d.masked_fill_(visible_mask, 0.0)

    return attn_mask_2d


@torch.no_grad()
def update_causal_mask_for_one_gen_window_2d(
    input_ids: torch.Tensor,
    attn_mask_2d: torch.Tensor,
    block_size: int = 4,
    use_cache: bool = True,
    causal_attn: bool = False
) -> torch.Tensor:
    """
    Updates a 2D attention mask for a diffusion window in transformer inference.

    Args:
        input_ids: Input token IDs (unused in current implementation)
        attn_mask_2d: 2D attention mask matrix of shape [seq_len, seq_len] where:
            - 0.0 indicates allowed attention
            - -inf indicates masked attention
        block_size: Size of the diffusion window
        use_cache: Whether key-value cache is being used
        causal_attn: If True, maintains strict causal masking throughout
    
    Returns:
        Modified attention mask with updated visibility patterns
    """
    if not causal_attn:
        attn_mask_2d[-block_size:, -block_size:] = 0.0
    if use_cache:
        attn_mask_2d[-block_size:, -block_size-1] = -float('inf')

    return attn_mask_2d


@torch.no_grad()
def create_block_diff_mask_by_pe_4d(
    block_size: int, 
    x0_len_list: torch.Tensor, 
    position_ids: torch.Tensor, 
    causal_attn: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a 4D attention mask for block-difference attention patterns.

    The mask consists of three regions:
    1. Causal block (top-left): Standard causal attention for `x0` tokens.
    2. Mutual block (bottom-right): Non-causal attention within the same block for non-`x0` tokens.
    3. Prefix block (bottom-left): Non-`x0` tokens can attend to a prefix of `x0` tokens.

    Args:
        block_size (int): Size of processing blocks for non-`x0` tokens.
        x0_len_list (torch.Tensor): Tensor of shape [B] containing lengths of `x0` segments per batch.
        position_ids (torch.Tensor): Tensor of shape [B, seq_len] containing position IDs.
        causal_attn (bool, optional): If True, enforces causal masking in mutual blocks. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - A float mask of shape [batch_size, 1, seq_len, seq_len] with `-inf` for masked positions (non visiable).
            - A boolean mask of shape [batch_size, 1, seq_len, seq_len] indicating allowed attention positions.
    """
    batch_size, seq_len = position_ids.shape
    device = position_ids.device
    
    q_idx = torch.arange(seq_len, device=device).view(1, seq_len, 1)
    kv_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len)
    
    x0_len = x0_len_list.view(batch_size, 1, 1)
    x0_flag_q = q_idx < x0_len
    x0_flag_kv = kv_idx < x0_len
    
    q_block_idx = (q_idx - x0_len) // block_size
    kv_block_idx = (kv_idx - x0_len) // block_size
    
    block_causal = x0_flag_q & x0_flag_kv & (q_idx >= kv_idx)

    mutual_condition = (q_idx >= kv_idx) if causal_attn else torch.ones_like(q_idx, dtype=torch.bool)
    block_mutual = (~x0_flag_q & ~x0_flag_kv & 
                   (q_block_idx == kv_block_idx) & 
                   mutual_condition)

    q_blk  = torch.div(q_idx - x0_len, block_size, rounding_mode='floor')
    q_blk_start = (x0_len_list.view(batch_size, 1) + q_blk[:, :, 0] * block_size).clamp(min=0, max=seq_len-1)
    prefix_len = position_ids.gather(1, q_blk_start)
    prefix_len = prefix_len.unsqueeze(2)
    block_prefix = (~x0_flag_q & x0_flag_kv) & (kv_idx < prefix_len)

    final_mask = (block_causal | block_mutual | block_prefix)
    customized_mask = torch.full_like(final_mask, float('-inf'), dtype=torch.bfloat16)
    customized_mask.masked_fill_(final_mask, 0.0)
    
    return customized_mask.unsqueeze(1).to(device=device), final_mask.unsqueeze(1).to(device=device)


def find_pred_pos_from_input_ids(
    input_ids: torch.LongTensor = None,
    text_mask_token_id: int = 151666,
) -> torch.Tensor:
    """Compute the relative prediction positions for masked tokens in a sequence.

    For non-masked positions, the output is 0. For masked positions, the value increments
    by 1 for each consecutive mask token, indicating how many steps ahead the prediction is.

    Args:
        input_ids (torch.LongTensor): Input token IDs of shape [batch_size, seq_len].
        text_mask_token_id (int, optional): Token ID representing masked positions. Defaults to 151666.

    Returns:
        torch.Tensor: A tensor of shape [batch_size, seq_len] where:
            - 0 indicates a non-masked token.
            - n > 0 indicates the nth consecutive masked token (e.g., 1 = first mask, 2 = second mask, etc.).
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    is_mask = (input_ids == text_mask_token_id)

    base_mask = torch.zeros((batch_size, seq_len), dtype=torch.int8, device=device)

    for b in range(batch_size):
        for ix in range(1, seq_len):
            if is_mask[b][ix] == True:
                base_mask[b][ix] = base_mask[b][ix-1] + 1

    return base_mask


@torch.no_grad()
def _find_sample_x0_len_packed(
    position_ids: torch.Tensor,
    data_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For packed sequences, find x0 length, sample start, and sample end for each sample.
    
    x0 region is identified by position_ids being monotonically increasing. When position_ids
    drops (i.e., current < previous), that indicates the boundary between x0 and MTP region.
    
    Args:
        position_ids: [batch_size, seq_len] position IDs (batch_size should be 1 for packing)
        data_index: [batch_size, seq_len] sample index for each position
    
    Returns:
        x0_lens: [num_samples] x0 length for each sample
        sample_starts: [num_samples] start position of each sample in the sequence
        sample_ends: [num_samples] end position (exclusive) of each sample
    """
    position_ids = position_ids.squeeze(0)
    data_index = data_index.squeeze(0)
    
    device = position_ids.device
    
    unique_samples = data_index.unique(sorted=True)
    num_samples = len(unique_samples)
    
    x0_lens = torch.zeros(num_samples, dtype=torch.long, device=device)
    sample_starts = torch.zeros(num_samples, dtype=torch.long, device=device)
    sample_ends = torch.zeros(num_samples, dtype=torch.long, device=device)
    
    for s_idx, sample_id in enumerate(unique_samples):
        mask = (data_index == sample_id)
        positions = mask.nonzero(as_tuple=True)[0]
        start = positions[0]
        end = positions[-1] + 1
        
        sample_starts[s_idx] = start
        sample_ends[s_idx] = end
        
        sample_pos = position_ids[start:end]
        if len(sample_pos) > 1:
            drops = sample_pos[1:] < sample_pos[:-1]
            if drops.any():
                x0_len = drops.nonzero(as_tuple=True)[0][0].item() + 1
            else:
                x0_len = len(sample_pos)
        else:
            x0_len = len(sample_pos)
        
        x0_lens[s_idx] = x0_len
    
    return x0_lens, sample_starts, sample_ends


@torch.no_grad()
def create_mtp_packing_mask_4d(
    block_size: int,
    position_ids: torch.Tensor,
    data_index: torch.Tensor,
    causal_attn: bool = False,
) -> torch.Tensor:
    """
    Create 4D attention mask for MTP with stream packing (for SDPA).
    
    Args:
        block_size: Size of each MTP prediction block
        position_ids: [1, seq_len] position IDs for the packed sequence
        data_index: [1, seq_len] sample index for each position
        causal_attn: If True, use causal attention within MTP blocks
    
    Returns:
        attention_mask: [1, 1, seq_len, seq_len] mask with 0.0 for visible and -inf for masked
    """
    seq_len = position_ids.shape[1]
    device = position_ids.device
    
    x0_lens, sample_starts, sample_ends = _find_sample_x0_len_packed(position_ids, data_index)
    
    position_ids_1d = position_ids.squeeze(0)
    data_index_1d = data_index.squeeze(0)
    
    q_idx = torch.arange(seq_len, device=device).view(seq_len, 1)
    kv_idx = torch.arange(seq_len, device=device).view(1, seq_len)
    
    q_sample = data_index_1d[q_idx.squeeze(-1)]
    kv_sample = data_index_1d[kv_idx.squeeze(0)]
    same_sample = q_sample.unsqueeze(1) == kv_sample.unsqueeze(0)
    
    unique_samples = data_index_1d.unique(sorted=True)
    sample_to_idx = {s.item(): i for i, s in enumerate(unique_samples)}
    
    pos_sample_idx = torch.tensor([sample_to_idx[s.item()] for s in data_index_1d], device=device)
    pos_x0_len = x0_lens[pos_sample_idx]
    pos_sample_start = sample_starts[pos_sample_idx]
    
    pos_in_sample = torch.arange(seq_len, device=device) - pos_sample_start
    
    q_in_x0 = pos_in_sample[q_idx.squeeze(-1)] < pos_x0_len[q_idx.squeeze(-1)]
    kv_in_x0 = pos_in_sample[kv_idx.squeeze(0)] < pos_x0_len[kv_idx.squeeze(0)]
    
    q_in_x0_2d = q_in_x0.unsqueeze(1)
    kv_in_x0_2d = kv_in_x0.unsqueeze(0)
    
    block_causal = (
        q_in_x0_2d & kv_in_x0_2d &
        (q_idx >= kv_idx)
    )
    
    q_mtp_offset = pos_in_sample[q_idx.squeeze(-1)] - pos_x0_len[q_idx.squeeze(-1)]
    kv_mtp_offset = pos_in_sample[kv_idx.squeeze(0)] - pos_x0_len[kv_idx.squeeze(0)]
    
    q_block_idx = q_mtp_offset // block_size
    kv_block_idx = kv_mtp_offset // block_size
    
    q_block_idx_2d = q_block_idx.unsqueeze(1)
    kv_block_idx_2d = kv_block_idx.unsqueeze(0)
    
    same_mtp_block = (q_block_idx_2d == kv_block_idx_2d) & (q_block_idx_2d >= 0) & (kv_block_idx_2d >= 0)
    
    if causal_attn:
        mutual_condition = (q_idx >= kv_idx)
    else:
        mutual_condition = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    
    block_mutual = (
        (~q_in_x0_2d) & (~kv_in_x0_2d) &
        same_mtp_block &
        mutual_condition
    )
    
    q_mtp_block_start_in_sample = pos_x0_len[q_idx.squeeze(-1)] + q_block_idx * block_size
    q_mtp_block_start_global = pos_sample_start[q_idx.squeeze(-1)] + q_mtp_block_start_in_sample
    q_mtp_block_start_global = q_mtp_block_start_global.clamp(0, seq_len - 1)
    
    q_prefix_len = position_ids_1d[q_mtp_block_start_global]
    
    kv_pos_in_sample = pos_in_sample[kv_idx.squeeze(0)]
    
    block_prefix = (
        (~q_in_x0_2d) & kv_in_x0_2d &
        (kv_pos_in_sample.unsqueeze(0) < q_prefix_len.unsqueeze(1))
    )
    
    final_mask = same_sample & (block_causal | block_mutual | block_prefix)
    
    attention_mask = torch.full((seq_len, seq_len), float('-inf'), dtype=torch.bfloat16, device=device)
    attention_mask.masked_fill_(final_mask, 0.0)
    
    return attention_mask.unsqueeze(0).unsqueeze(0)
