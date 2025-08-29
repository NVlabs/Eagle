# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist
import torch.nn.functional as F


def split_for_sequence_parallel(inputs, dim: int, sp_group: dist.ProcessGroup, fill_zeros=False):
    """Splits the input tensor along a given dimension for sequence parallel.

    Args:
        input: The input tensor to be split.
        dim: The dimension along which the tensor should be split.
        sp_group: The sequence parallel process group.

    Returns:
        The split tensor corresponding to the current rank's chunk.
    """
    world_size = dist.get_world_size(sp_group)
    if world_size == 1:
        return inputs

    rank = dist.get_rank(sp_group)
    dim_size = inputs.size(dim)
    
    if not fill_zeros:  
        assert dim_size % world_size == 0, (
            f'The dimension to split ({dim_size}) is not a multiple of '
        f'world size ({world_size}), cannot split tensor evenly')
        tensor_list = torch.split(inputs, dim_size // world_size, dim=dim)
        output = tensor_list[rank].contiguous()
        return output
    else:
        zero_padding_len = (world_size - dim_size % world_size) % world_size
        zero_padding_shape = list(inputs.shape)
        zero_padding_shape[dim] = zero_padding_len
        zero_padding = torch.zeros(zero_padding_shape, dtype=inputs.dtype, device=inputs.device)
        inputs = torch.cat([inputs, zero_padding], dim=dim)
        new_dim_size = dim_size + zero_padding_len
        tensor_list = torch.split(inputs, new_dim_size // world_size, dim=dim)
        output = tensor_list[rank].contiguous()
        # split_list = [tensor_list[i].shape[0] for i in range(world_size)]
        # split_list[-1] = split_list[-1] - zero_padding_len
        # print("split_list:", split_list, zero_padding_len)

        return output


def ring_split_for_sequence_parallel(inputs, ulysses_group: dist.ProcessGroup, ring_group: dist.ProcessGroup, sub_sample_lengths, split_by_ulysses=True, ring_zigzag=False):
    """Splits the input tensor along a given dimension for sequence parallel.

    Args:
        input: The input tensor to be split.
        dim: The dimension along which the tensor should be split.
        sp_group: The sequence parallel process group.

    Returns:
        The split tensor corresponding to the current rank's chunk.
    """
    # set seq dim
    if inputs.dim() == 3 or inputs.dim() == 2:
        # hidden states like
        channel_dim = 1
    elif inputs.dim() == 1:
        # reudction none loss
        channel_dim = 0
    else:
        print(inputs.shape, inputs.dim())
        assert False

    ulysses_rank = dist.get_rank(ulysses_group)
    ulysses_world_size = dist.get_world_size(ulysses_group)
    
    ring_rank = dist.get_rank(ring_group)
    ring_world_size = dist.get_world_size(ring_group)
    print("ring:", ring_world_size, ring_rank)
    # print("shape:", inputs.shape, len(sub_sample_lengths))
    sub_sample_lengths = torch.tensor(sub_sample_lengths[0], device=inputs.device, dtype=torch.int32)
    assert torch.all(sub_sample_lengths % ring_world_size == 0)
    
    if ring_world_size > 1:
        cu_seqlens = F.pad(torch.cumsum(sub_sample_lengths, dim=0, dtype=torch.int32), (1, 0))
        # print("inputs", inputs.shape)
        # print("sub_sample_lengths", sub_sample_lengths)
        # print("cu_seqlens", cu_seqlens)
        

        local_values = []
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            if not ring_zigzag:
                local_value = inputs[:, start:end].chunk(ring_world_size, dim=1)[ring_rank]
                local_values.append(local_value)
            else:
                local_value = inputs[:, start:end].chunk(2 * ring_world_size, dim=1)
                local_values.extend(
                    [
                        local_value[ring_rank],
                        local_value[2 * ring_world_size - 1 - ring_rank],
                    ]
                )
            
        local_inputs = torch.cat(local_values, dim=channel_dim).contiguous()
        # print("local_inputs_RING", local_inputs.shape)
    else:
        local_inputs = inputs
    if split_by_ulysses:
        local_inputs = local_inputs.chunk(ulysses_world_size, dim=channel_dim)[ulysses_rank].contiguous()
        # print("local_inputs_ULYSSES", local_inputs.shape)
    return local_inputs




def ring_gather_for_sequence_parallel(inputs, ulysses_group: dist.ProcessGroup, ring_group: dist.ProcessGroup, sub_sample_lengths, split_by_ulysses=True, ring_zigzag=False):
    """Splits the input tensor along a given dimension for sequence parallel.

    Args:
        input: The input tensor to be split.
        dim: The dimension along which the tensor should be split.
        sp_group: The sequence parallel process group.

    Returns:
        The split tensor corresponding to the current rank's chunk.
    """

    # set seq dim
    if inputs.dim() == 3 or inputs.dim() == 2:
        # hidden states like
        channel_dim = 1
    elif inputs.dim() == 1:
        # reudction none loss
        channel_dim = 0
    else:
        print(inputs.shape, inputs.dim())
        assert False

    ulysses_rank = dist.get_rank(ulysses_group)
    ulysses_world_size = dist.get_world_size(ulysses_group)
    
    ring_rank = dist.get_rank(ring_group)
    ring_world_size = dist.get_world_size(ring_group)

    # ulysses gather
    tensor_list = [torch.zeros_like(inputs) for _ in range(ulysses_world_size)]
    # print("device", inputs.device, tensor_list[0].device)
    dist.all_gather(tensor_list, inputs, group=ulysses_group)
    tensor_list[ulysses_rank] = inputs
    inputs = torch.cat(tensor_list, dim=channel_dim).contiguous()

    # ring gather
    if ring_world_size > 1:
        sub_sample_lengths = torch.tensor(sub_sample_lengths[0], device=inputs.device, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(sub_sample_lengths, dim=0, dtype=torch.int32), (1, 0))
        tensor_list = [torch.zeros_like(inputs) for _ in range(ring_world_size)]
        dist.all_gather(tensor_list, inputs, group=ring_group)
        tensor_list[ring_rank] = inputs
        merged_inputs = []
        if not ring_zigzag:
            assert torch.all(sub_sample_lengths % ring_world_size == 0)
            for i in range(len(cu_seqlens) - 1):
                start, end = cu_seqlens[i] // ring_world_size, cu_seqlens[i + 1] // ring_world_size
                for j in range(ring_world_size):
                    merged_inputs.append(tensor_list[j][:,start:end])
            merged_inputs = torch.cat(merged_inputs, dim=channel_dim)
        else:
            assert torch.all(sub_sample_lengths % (2 * ring_world_size) == 0)
            merged_inputs = torch.cat([torch.zeros_like(inputs) for _ in range(ring_world_size)], dim=channel_dim)
            for i in range(len(cu_seqlens)-1):
                start, end = cu_seqlens[i], cu_seqlens[i+1]
                seq_length = end - start
                chunk_size = seq_length // (2 * ring_world_size)
                local_start, local_end = start // ring_world_size, end // ring_world_size
                
                for rank in range(ring_world_size):
                    forward_chunk, backward_chunk = tensor_list[rank][:,local_start:local_end].chunk(2, dim=channel_dim)
                    merged_inputs[:, start + rank * chunk_size : start + (rank + 1) * chunk_size] = forward_chunk
                    merged_inputs[:, end - (rank + 1) * chunk_size : end - rank * chunk_size] = backward_chunk
    else:
        merged_inputs = inputs
    return merged_inputs


def gather_from_sequence_parallel(inputs, dim: int, sp_group: dist.ProcessGroup):
    """Gathers the input tensor along a given dimension for sequence parallel.

    Args:
        input: The input tensor to be gathered.
        dim: The dimension along which the tensor should be gathered.
        sp_group: The sequence parallel process group.

    Returns:
        The gathered tensor.
    """
    world_size = dist.get_world_size(sp_group)
    if world_size == 1:
        return inputs

    rank = dist.get_rank(sp_group)

    tensor_list = [torch.zeros_like(inputs) for _ in range(world_size)]
    dist.all_gather(tensor_list, inputs, group=sp_group)
    tensor_list[rank] = inputs
    output = torch.cat(tensor_list, dim=dim)

    return output



def extract_local_from_list(vaule_list, sp_rank, sp_size):
    quotient, remainder = divmod(len(vaule_list), sp_size)
    start_idx = sp_rank * quotient + min(sp_rank, remainder)
    end_idx = (sp_rank + 1) * quotient + min(sp_rank + 1, remainder)
    return vaule_list[start_idx:end_idx]


def extract_local_input_ids(input_ids, image_positions, sp_rank, sp_size, bos_token_id=1, image_token_len=3):
    quotient, remainder = divmod(len(image_positions), sp_size)
    start_idx = sp_rank * quotient + min(sp_rank, remainder)
    end_idx = (sp_rank + 1) * quotient + min(sp_rank + 1, remainder)

    start_position_idx = image_positions[start_idx]
    if sp_rank != sp_size - 1:
        end_position_idx = image_positions[end_idx]
    else:
        end_position_idx = len(input_ids)

    if sp_rank == 0:  # Handle the head of the sequence
        return input_ids[0:end_position_idx]
    elif sp_rank == sp_size - 1:  # Handle the tail of the sequence
        return input_ids[start_position_idx:]
    else:
        return input_ids[start_position_idx:end_position_idx]


def extract_local_position_ids(input_ids, image_positions, image_ids, sp_rank, sp_size, image_token_len=198):
    quotient, remainder = divmod(len(image_ids), sp_size)
    start_idx = sp_rank * quotient + min(sp_rank, remainder)
    end_idx = (sp_rank + 1) * quotient + min(sp_rank + 1, remainder)
    start_position_idx = image_positions[start_idx] + image_ids[start_idx] * image_token_len
    if sp_rank != sp_size - 1:  # Handle the tail of the sequence
        end_position_idx = image_positions[end_idx] + image_ids[end_idx] * image_token_len  # image_token_len + 3
    else:
        end_position_idx = len(input_ids)
    if sp_rank == 0:  # Handle the head of the sequence
        return input_ids[0:end_position_idx]
    elif sp_rank == sp_size - 1:  # Handle the tail of the sequence
        return input_ids[start_position_idx:]
    else:
        return input_ids[start_position_idx:end_position_idx]



def extract_local(value, rank, world_size, dim=1):
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat([value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim)
    return local_value


def prepare_hybrid_attn_inputs(input_ids, position_ids, target_ids, rank, world_size, device):
    local_input_ids = extract_local(
        input_ids,
        rank,
        world_size,
        device,
    )
    local_position_ids = extract_local(
        position_ids,
        rank,
        world_size,
        device,
    )
    if target_ids is not None:
        local_target_ids = extract_local(
            target_ids,
            rank,
            world_size,
            device,
        )
    else:
        local_target_ids = None
    return {
        "local_input_ids": local_input_ids,
        "local_position_ids": local_position_ids,
        "local_target_ids": local_target_ids,
    }
