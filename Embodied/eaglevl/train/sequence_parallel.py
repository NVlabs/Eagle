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

"""Tensor helpers shared by Eagle Embodied sequence-parallel training."""

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


def graph_connected_fp32_zero(*tensors):
    """Return an FP32 scalar zero that remains connected to every tensor."""
    if not tensors:
        raise ValueError('At least one tensor is required.')
    result = tensors[0].reshape(-1)[0].float()
    for tensor in tensors[1:]:
        result = result + tensor.reshape(-1)[0].float()
    return result * 0.0


def pad_packed_sequence_for_ulysses(
    input_embeds,
    attention_mask,
    position_ids,
    labels,
    sub_sample_lengths,
    sequence_parallel_size,
):
    """Pad a packed sequence with an independent dummy segment."""
    if input_embeds.dim() != 3:
        raise ValueError(
            f'Expected input_embeds with shape [batch, sequence, hidden], got {input_embeds.shape}.'
        )
    if labels is None:
        raise ValueError('Packed Ulysses training requires labels.')
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    elif labels.dim() != 2:
        raise ValueError(f'Expected labels with shape [sequence] or [batch, sequence], got {labels.shape}.')
    if attention_mask is None:
        raise ValueError('Packed Ulysses training requires an integer data-index attention mask.')
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    sequence_length = input_embeds.shape[1]
    expected_shape = (input_embeds.shape[0], sequence_length)
    for name, tensor in (
        ('labels', labels),
        ('attention_mask', attention_mask),
        ('position_ids', position_ids),
    ):
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f'{name} must have shape {expected_shape}, got {tuple(tensor.shape)}.')
    if len(sub_sample_lengths) != input_embeds.shape[0]:
        raise ValueError(
            f'Expected one sub_sample_lengths tensor per batch item, got {len(sub_sample_lengths)}.'
        )
    for lengths in sub_sample_lengths:
        if int(lengths.sum().item()) != sequence_length:
            raise ValueError(
                f'Packed sub-sample lengths must sum to {sequence_length}, got {int(lengths.sum().item())}.'
            )

    padding_length = (-sequence_length) % sequence_parallel_size
    if padding_length == 0:
        return input_embeds, attention_mask, position_ids, labels, sub_sample_lengths

    input_embeds = F.pad(input_embeds, (0, 0, 0, padding_length), value=0.0)
    labels = F.pad(labels, (0, padding_length), value=IGNORE_INDEX)

    pad_segment_id = attention_mask.max(dim=1, keepdim=True).values + 1
    mask_padding = pad_segment_id.expand(-1, padding_length)
    attention_mask = torch.cat([attention_mask, mask_padding], dim=1)

    position_padding = torch.arange(
        padding_length,
        device=position_ids.device,
        dtype=position_ids.dtype,
    ).unsqueeze(0).expand(position_ids.shape[0], -1)
    position_ids = torch.cat([position_ids, position_padding], dim=1)

    padded_lengths = [
        torch.cat([lengths.clone(), lengths.new_tensor([padding_length])])
        for lengths in sub_sample_lengths
    ]
    return input_embeds, attention_mask, position_ids, labels, padded_lengths


def shift_packed_labels(labels, sub_sample_lengths):
    """Build next-token labels without crossing packed sample boundaries."""
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    if labels.dim() != 2:
        raise ValueError(f'Expected labels with shape [sequence] or [batch, sequence], got {labels.shape}.')
    if len(sub_sample_lengths) != labels.shape[0]:
        raise ValueError(
            f'Expected one sub_sample_lengths tensor per batch item, got {len(sub_sample_lengths)}.'
        )
    shifted = torch.roll(labels, shifts=-1, dims=-1)
    for batch_index, lengths in enumerate(sub_sample_lengths):
        sample_ends = torch.cumsum(lengths.to(labels.device, dtype=torch.long), dim=0) - 1
        shifted[batch_index, sample_ends] = IGNORE_INDEX
    return shifted
