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

"""Token-level loss weights for VFlash target and drafter training."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from eaglevl.model.vflash.physical_atoms import (
    DEFAULT_OOF_LOSS_SCALE,
    iter_physical_atoms,
    validate_oof_loss_scale,
)
from eaglevl.train.vflash.drafter_packing import labeled_response_ranges


def build_oof_loss_weights(
    labels: torch.Tensor,
    *,
    oof_token_id: Optional[int],
    oof_loss_scale: float = DEFAULT_OOF_LOSS_SCALE,
) -> torch.Tensor:
    """Return FP32 per-label weights with ``<oof>`` scaled independently.

    Token embeddings and the LM head are dense shared matrices, so a single
    vocabulary row cannot be assigned its own standard optimizer parameter
    group. Per-label weighting is the optimizer- and ZeRO-safe way to give
    ``<oof>`` one tenth of the ordinary training signal.
    """

    oof_loss_scale = validate_oof_loss_scale(oof_loss_scale)
    weights = torch.ones(labels.shape, dtype=torch.float32, device=labels.device)
    if oof_token_id is not None and oof_loss_scale != 1.0:
        weights.masked_fill_(labels.eq(int(oof_token_id)), oof_loss_scale)
    return weights


def apply_oof_loss_scale(
    base_weights: torch.Tensor,
    labels: torch.Tensor,
    *,
    oof_token_id: Optional[int],
    oof_loss_scale: float = DEFAULT_OOF_LOSS_SCALE,
) -> torch.Tensor:
    """Multiply the ``<oof>`` positions of ``base_weights`` by ``oof_loss_scale``.

    ``labels`` must align with ``base_weights`` position for position. Callers
    keep their own denominator, so each ``<oof>`` label contributes an
    absolute 0.1x loss and gradient.
    """

    if base_weights.shape != labels.shape:
        raise ValueError(
            "base_weights and labels must have the same shape, got "
            f"{tuple(base_weights.shape)} and {tuple(labels.shape)}"
        )
    multipliers = build_oof_loss_weights(
        labels,
        oof_token_id=oof_token_id,
        oof_loss_scale=oof_loss_scale,
    )
    return base_weights * multipliers.to(
        device=base_weights.device,
        dtype=base_weights.dtype,
    )


def build_atom_label_weights(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    atom_loss_masks: Optional[Sequence[str]],
    physical_tokens,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return FP32 per-position label weights from a row's ``atom_loss_masks``.

    ``atom_loss_masks`` lists one ``'0'``/``'1'`` string per physical atom of the
    assistant response, in response order; character ``k`` weights payload
    token ``k`` of that atom (start/end tokens and all ordinary text keep
    weight 1).  Prompt-side atoms are not labeled and are skipped.  Missing or
    empty masks mean "all ones".  Mismatched atom counts or payload lengths
    raise, because a silently misaligned mask would train the wrong cells, and
    so does a malformed atom.
    """

    weights = torch.ones(input_ids.shape, dtype=torch.float32)
    if not atom_loss_masks:
        return weights
    if physical_tokens is None:
        raise ValueError(
            "atom_loss_masks requires strict physical atoms so atom payloads can be located"
        )
    masks = [str(mask) for mask in atom_loss_masks]
    ids = input_ids.tolist()
    atom_index = 0
    for response_start, response_end in labeled_response_ranges(labels, ignore_index):
        for start, end, spec in iter_physical_atoms(
            ids, response_start, response_end, physical_tokens
        ):
            if atom_index >= len(masks):
                raise ValueError(
                    f"atom_loss_masks has {len(masks)} entries but the assistant "
                    "response contains more physical atoms"
                )
            mask = masks[atom_index]
            payload_length = end - start - 1
            if len(mask) != payload_length:
                raise ValueError(
                    f"atom_loss_masks[{atom_index}] has {len(mask)} characters but the "
                    f"{spec.schema.name} payload has {payload_length} tokens"
                )
            for offset, char in enumerate(mask):
                if char == "0":
                    weights[start + 1 + offset] = 0.0
                elif char != "1":
                    raise ValueError("atom_loss_masks characters must be '0' or '1'")
            atom_index += 1
    if atom_index != len(masks):
        raise ValueError(
            f"atom_loss_masks has {len(masks)} entries but the assistant response "
            f"contains {atom_index} physical atoms"
        )
    return weights


__all__ = [
    "apply_oof_loss_scale",
    "build_atom_label_weights",
    "build_oof_loss_weights",
]
