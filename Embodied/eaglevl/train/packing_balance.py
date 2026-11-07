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

"""Deterministic media-cost balancing helpers for stream packing."""

import random
from typing import Sequence

import torch


def actual_media_tokens(sample: dict, media_token_id: int) -> int:
    """Return the number of real media tokens represented by a sample or pack."""
    image_flags = sample.get("image_flags")
    if image_flags is not None:
        flags = torch.as_tensor(image_flags)
        if flags.numel() == 0 or int(flags.sum().item()) == 0:
            return 0

    input_ids = sample.get("input_ids")
    if input_ids is None:
        return 0
    ids = torch.as_tensor(input_ids)
    return int((ids == int(media_token_id)).sum().item())


def media_item_count(sample: dict) -> int:
    """Return the number of real image/video items represented by a sample or pack."""
    image_flags = sample.get("image_flags")
    if image_flags is not None:
        return int(torch.as_tensor(image_flags).sum().item())

    image_grid_hws = sample.get("image_grid_hws")
    if image_grid_hws is None:
        return 0
    return int(len(image_grid_hws))


def pack_balance_key(sample: dict, media_token_id: int) -> tuple[int, int, int]:
    """Build a stable media-first cost key for one completed pack."""
    input_ids = sample.get("input_ids")
    total_tokens = int(torch.as_tensor(input_ids).numel()) if input_ids is not None else 0
    return (
        actual_media_tokens(sample, media_token_id),
        media_item_count(sample),
        total_tokens,
    )


def aligned_quantile_order(
    costs: Sequence[tuple[int, ...]],
    *,
    seed: int,
) -> list[int]:
    """Return a shared permutation over rank-local cost quantiles.

    Each data-parallel rank supplies its own costs. Sorting first converts
    rank-local packs into comparable quantile positions; shuffling those
    positions with the same seed aligns expensive and cheap steps without
    communicating or duplicating samples.
    """
    sorted_indices = sorted(range(len(costs)), key=lambda index: (costs[index], index))
    quantile_order = list(range(len(sorted_indices)))
    random.Random(int(seed)).shuffle(quantile_order)
    return [sorted_indices[index] for index in quantile_order]


def reorder_by_media_quantile(
    packs: Sequence[dict],
    *,
    media_token_id: int,
    seed: int,
) -> list[dict]:
    """Order completed packs by a deterministic media-cost quantile schedule."""
    costs = [pack_balance_key(pack, media_token_id) for pack in packs]
    return [packs[index] for index in aligned_quantile_order(costs, seed=seed)]
