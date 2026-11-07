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

"""VFlash drafter blocks built from packed clean training samples.

The frozen target and the diffusion drafter consume different inputs:

* the clean target sequence is the unchanged causal sequence used only to
  extract frozen target hidden states.
* every ``VFlashTrainingBlock`` is a separate drafter query consisting of one
  clean anchor followed by masks.

Draft labels are *same-position* reconstruction labels.  They must not be fed
through a causal next-token label shift: for ``[anchor, MASK, MASK]`` the labels
are ``[IGNORE, token_after_anchor, next_token]`` at those exact positions.

Point and bbox atoms are drafted as fixed blocks, every robot or driving
primitive as one block, and every dense trajectory as one block. On top of
those mandatory atom blocks, up to 512 ordinary-text B16 blocks per sequence
are sampled.

Missing point or box annotations use the plain text literal ``None`` (capital
N) outside physical atoms.  ``none`` and serialized forms such as
``<box>None</box>`` are invalid data.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from eaglevl.model.vflash.physical_atoms import (
    ATOMIC_TRAJECTORY_LAYOUTS,
    ATOMIC_TRAJECTORY_LAYOUT_NAMES,
    TEXT_VFLASH_BLOCK_SIZE,
    PhysicalAtomTokenIds,
    PhysicalAtomValidationError,
    ResolvedPhysicalAtomSpec,
    is_variable_trajectory_atom_name,
    iter_physical_atoms,
)


IGNORE_TOKEN_ID = -100
DEFAULT_VFLASH_GAMMA = 7.0
DEFAULT_ORDINARY_ANCHORS_PER_SEQUENCE = 512
# Dense trajectories are drafted one block per trajectory (a 2D atom followed
# by its 3D atom forms one block).
ATOMIC_SPLIT_LAYOUT_NAMES: Tuple[str, ...] = (
    "drive_3d",
    "drive_2d3d",
    "robot_single_3d",
    "robot_single_2d3d",
    "dense_3d",
    "dense_2d3d",
)
NONE_DATA_CONVENTION = (
    "Missing point or box annotations must be plain None (capital N) outside "
    "physical atoms; lowercase none and forms such as <box>None</box> are invalid."
)


def resolve_vflash_sample_schedule(
    *,
    epoch_samples: int,
    sample_epochs: float,
    samples_per_pack: int,
    data_parallel_world_size: int,
    gradient_accumulation_steps: int = 1,
) -> Mapping[str, Any]:
    """Resolve an optimizer-step budget from raw samples rather than packs."""

    integer_fields = {
        "epoch_samples": epoch_samples,
        "samples_per_pack": samples_per_pack,
        "data_parallel_world_size": data_parallel_world_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    for name, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    sample_epochs = float(sample_epochs)
    if not isfinite(sample_epochs) or sample_epochs <= 0:
        raise ValueError("sample_epochs must be finite and positive")

    requested_samples = int(round(epoch_samples * sample_epochs))
    samples_per_optimizer_step = (
        samples_per_pack
        * data_parallel_world_size
        * gradient_accumulation_steps
    )
    max_steps = (
        requested_samples + samples_per_optimizer_step - 1
    ) // samples_per_optimizer_step
    if max_steps < 1:
        raise ValueError(
            "VFlash sample-epoch schedule is shorter than one distributed optimizer step"
        )
    return {
        "epoch_samples": epoch_samples,
        "requested_sample_epochs": sample_epochs,
        "requested_raw_samples": requested_samples,
        "samples_per_optimizer_step": samples_per_optimizer_step,
        "max_steps": max_steps,
        "trained_raw_samples": max_steps * samples_per_optimizer_step,
        "distributed_tail_padding_samples": (
            max_steps * samples_per_optimizer_step - requested_samples
        ),
    }


class VFlashDataError(ValueError):
    """Raised when a response cannot be converted into strict VFlash blocks."""

    def __init__(self, message: str):
        super().__init__(f"{message} {NONE_DATA_CONVENTION}")


@dataclass(frozen=True)
class PhysicalAtomOccurrence:
    """A validated physical atom in the clean target sequence.

    ``start`` and ``end`` are inclusive clean-sequence positions.
    """

    start: int
    end: int
    spec: ResolvedPhysicalAtomSpec
    actual_block_size: Optional[int] = None

    @property
    def block_size(self) -> int:
        return (
            self.spec.block_size
            if self.actual_block_size is None
            else int(self.actual_block_size)
        )


@dataclass(frozen=True)
class VFlashTrainingBlock:
    """One independent drafter query.

    ``context_start:context_end`` is the clean-target interval visible to the
    block. ``context_end`` is exclusive and always equals ``anchor_position``.
    Consequently the anchor itself lives only in the drafter block.
    """

    anchor_position: int
    context_start: int
    context_end: int
    block_size: int
    kind: str
    draft_input_ids: Tuple[int, ...]
    draft_labels: Tuple[int, ...]
    draft_position_ids: Tuple[int, ...]
    loss_weights: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.context_start <= self.context_end:
            raise ValueError(
                "VFlash context must satisfy 0 <= context_start <= context_end, got "
                f"{self.context_start}:{self.context_end}"
            )
        if self.context_end != self.anchor_position:
            raise ValueError(
                "VFlash target context must end strictly before the anchor: "
                f"context_end={self.context_end}, anchor={self.anchor_position}"
            )
        for field_name, values in (
            ("draft_input_ids", self.draft_input_ids),
            ("draft_labels", self.draft_labels),
            ("draft_position_ids", self.draft_position_ids),
            ("loss_weights", self.loss_weights),
        ):
            if len(values) != self.block_size:
                raise ValueError(
                    f"{field_name} has length {len(values)}, expected {self.block_size}"
                )
        expected_positions = tuple(
            range(self.anchor_position, self.anchor_position + self.block_size)
        )
        if self.draft_position_ids != expected_positions:
            raise ValueError(
                "VFlash block RoPE positions must run from anchor through "
                "anchor + block_size - 1"
            )
        if self.draft_labels[0] != IGNORE_TOKEN_ID:
            raise ValueError("The clean anchor must have an IGNORE same-position label")
        if self.loss_weights[0] != 0.0:
            raise ValueError("The clean anchor must have zero draft-loss weight")
        for label, weight in zip(self.draft_labels, self.loss_weights):
            if label == IGNORE_TOKEN_ID and weight != 0.0:
                raise ValueError("Ignored draft positions must have zero loss weight")


@dataclass(frozen=True)
class VFlashTrainingExample:
    """A clean target sequence plus a ragged collection of draft blocks."""

    clean_target_input_ids: Tuple[int, ...]
    draft_blocks: Tuple[VFlashTrainingBlock, ...]

    def target_context_for(self, block: VFlashTrainingBlock) -> Tuple[int, ...]:
        """Return exactly the frozen target prefix visible to ``block``."""

        if not 0 <= block.context_start <= block.context_end <= len(
            self.clean_target_input_ids
        ):
            raise ValueError(
                "Block context interval "
                f"{block.context_start}:{block.context_end} is outside the clean target"
            )
        return self.clean_target_input_ids[block.context_start : block.context_end]


@dataclass(frozen=True)
class PackedVFlashTrainingBatch:
    """Tensorized clean targets and concatenated, still-independent draft blocks.

    Every draft row may see only its block's strict target prefix
    (``block_context_starts`` to ``block_anchors`` in packed target columns)
    and its own complete bidirectional draft block (``draft_block_ranges``).
    """

    packed_clean_target_input_ids: torch.Tensor
    packed_clean_target_position_ids: torch.Tensor
    clean_target_lengths: torch.Tensor
    packed_target_length: int
    draft_noise_ids: torch.Tensor
    direct_labels: torch.Tensor
    position_ids: torch.Tensor
    loss_weights: torch.Tensor
    block_context_starts: torch.Tensor
    block_anchors: torch.Tensor
    draft_block_ranges: torch.Tensor
    block_sizes: torch.Tensor

    def __post_init__(self) -> None:
        if self.packed_clean_target_input_ids.numel() != self.packed_target_length:
            raise ValueError("packed_target_length does not match packed clean target IDs")


def pack_vflash_training_examples(
    examples: Sequence[VFlashTrainingExample],
    device: Optional[Any] = None,
) -> PackedVFlashTrainingBatch:
    """Concatenate strict-VFlash examples and their block-visibility metadata.

    Clean target IDs remain a distinct tensor.  The returned ``draft_noise_ids``,
    ``direct_labels``, ``position_ids``, and ``loss_weights`` are concatenations
    of complete blocks.  ``direct_labels`` is intentionally named to make clear
    that no causal label shift is permitted.
    """

    if not examples:
        raise ValueError("pack_vflash_training_examples requires at least one example")

    target_lengths = [len(example.clean_target_input_ids) for example in examples]
    packed_target_length = sum(target_lengths)
    packed_clean_ids = tuple(
        token_id
        for example in examples
        for token_id in example.clean_target_input_ids
    )
    packed_clean_positions = tuple(
        position
        for target_length in target_lengths
        for position in range(target_length)
    )

    draft_ids: List[int] = []
    direct_labels: List[int] = []
    position_ids: List[int] = []
    loss_weights: List[float] = []
    context_starts: List[int] = []
    anchors: List[int] = []
    draft_ranges: List[Tuple[int, int]] = []
    block_sizes: List[int] = []

    target_offset = 0
    for example in examples:
        for block in example.draft_blocks:
            # Validate that metadata belongs to this clean target before
            # turning it into absolute packed-target columns.
            example.target_context_for(block)
            draft_start = len(draft_ids)
            draft_ids.extend(block.draft_input_ids)
            direct_labels.extend(block.draft_labels)
            position_ids.extend(block.draft_position_ids)
            loss_weights.extend(block.loss_weights)
            draft_end = len(draft_ids)

            context_starts.append(target_offset + block.context_start)
            anchors.append(target_offset + block.anchor_position)
            draft_ranges.append((draft_start, draft_end))
            block_sizes.append(block.block_size)
        target_offset += len(example.clean_target_input_ids)

    def long_tensor(values: Any) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=device)

    return PackedVFlashTrainingBatch(
        packed_clean_target_input_ids=long_tensor(packed_clean_ids),
        packed_clean_target_position_ids=long_tensor(packed_clean_positions),
        clean_target_lengths=long_tensor(target_lengths),
        packed_target_length=packed_target_length,
        draft_noise_ids=long_tensor(draft_ids),
        direct_labels=long_tensor(direct_labels),
        position_ids=long_tensor(position_ids),
        loss_weights=torch.tensor(loss_weights, dtype=torch.float32, device=device),
        block_context_starts=long_tensor(context_starts),
        block_anchors=long_tensor(anchors),
        draft_block_ranges=long_tensor(draft_ranges).reshape(-1, 2),
        block_sizes=long_tensor(block_sizes),
    )


def labeled_response_ranges(
    labels: torch.Tensor, ignore_index: int = IGNORE_TOKEN_ID
) -> List[Tuple[int, int]]:
    """Return inclusive contiguous ranges of non-ignored response labels."""

    positions = torch.where(labels.ne(ignore_index))[0].tolist()
    if not positions:
        return []

    ranges: List[Tuple[int, int]] = []
    start = previous = int(positions[0])
    for raw_position in positions[1:]:
        position = int(raw_position)
        if position != previous + 1:
            ranges.append((start, previous))
            start = position
        previous = position
    ranges.append((start, previous))
    return ranges


def build_online_packed_vflash_batch(
    packed_clean_target_input_ids: torch.Tensor,
    packed_clean_labels: torch.Tensor,
    clean_target_lengths: Sequence[int],
    physical_tokens: PhysicalAtomTokenIds,
    *,
    ordinary_anchors_per_sequence: int = DEFAULT_ORDINARY_ANCHORS_PER_SEQUENCE,
    gamma: float = DEFAULT_VFLASH_GAMMA,
    anchor_seed: int,
    packed_clean_label_weights: Optional[torch.Tensor] = None,
) -> PackedVFlashTrainingBatch:
    """Split an online clean pack and build one independent VFlash example per segment.

    Clean target tokens stay concatenated for the frozen target's varlen forward,
    while every segment receives its own response ranges, random-anchor budget,
    local RoPE positions, and target-context interval. The returned absolute
    context offsets let the sparse draft mask prevent cross-segment attention.
    """

    if not isinstance(packed_clean_target_input_ids, torch.Tensor):
        packed_clean_target_input_ids = torch.as_tensor(
            packed_clean_target_input_ids, dtype=torch.long
        )
    if not isinstance(packed_clean_labels, torch.Tensor):
        packed_clean_labels = torch.as_tensor(packed_clean_labels, dtype=torch.long)
    if packed_clean_target_input_ids.ndim != 1:
        raise ValueError("packed clean target input IDs must be one-dimensional")
    if packed_clean_labels.ndim != 1:
        raise ValueError("packed clean target labels must be one-dimensional")
    if packed_clean_labels.numel() != packed_clean_target_input_ids.numel():
        raise ValueError(
            "packed clean target IDs and labels must contain the same number of tokens"
        )
    if packed_clean_label_weights is not None:
        if not isinstance(packed_clean_label_weights, torch.Tensor):
            packed_clean_label_weights = torch.as_tensor(
                packed_clean_label_weights, dtype=torch.float32
            )
        if packed_clean_label_weights.ndim != 1 or (
            packed_clean_label_weights.numel() != packed_clean_labels.numel()
        ):
            raise ValueError(
                "packed clean label weights must be one-dimensional and align "
                "with the packed clean labels"
            )

    if isinstance(clean_target_lengths, torch.Tensor):
        if clean_target_lengths.ndim != 1:
            raise ValueError("clean target lengths must be one-dimensional")
        raw_lengths = clean_target_lengths.detach().cpu().tolist()
    else:
        raw_lengths = list(clean_target_lengths)
    lengths = [int(length) for length in raw_lengths]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("an online VFlash pack requires positive clean target lengths")
    if sum(lengths) != packed_clean_target_input_ids.numel():
        raise ValueError(
            f"packed clean target lengths sum to {sum(lengths)}, expected "
            f"{packed_clean_target_input_ids.numel()}"
        )

    examples: List[VFlashTrainingExample] = []
    offset = 0
    for segment_index, length in enumerate(lengths):
        segment_end = offset + length
        segment_ids = packed_clean_target_input_ids[offset:segment_end]
        segment_labels = packed_clean_labels[offset:segment_end]
        segment_weights = (
            None
            if packed_clean_label_weights is None
            else packed_clean_label_weights[offset:segment_end].detach().cpu().tolist()
        )
        examples.append(
            build_vflash_training_example(
                clean_target_input_ids=segment_ids,
                response_ranges=labeled_response_ranges(segment_labels),
                physical_tokens=physical_tokens,
                ordinary_anchors_per_sequence=ordinary_anchors_per_sequence,
                gamma=gamma,
                rng=np.random.default_rng(
                    (int(anchor_seed) + segment_index * 999983) % (1 << 63)
                ),
                label_weights=segment_weights,
            )
        )
        offset = segment_end

    packed = pack_vflash_training_examples(examples)
    if not torch.equal(
        packed.packed_clean_target_input_ids,
        packed_clean_target_input_ids.detach().cpu(),
    ):
        raise RuntimeError("VFlash online packing changed the clean target IDs")
    return packed


def vflash_position_loss_weights(
    block_size: int,
    gamma: float = DEFAULT_VFLASH_GAMMA,
    supervised_tail_length: Optional[int] = None,
) -> Tuple[float, ...]:
    """Return ``[0, exp(-(k-1)/gamma), ...]`` for same-position loss.

    ``k=1`` is the first token after the anchor. Slots without a target (for
    example at the end of a short text response) receive weight zero. This is
    the ordinary-text B16 rule (``gamma=7``); physical atoms use the separate
    uniform-weight helper below and never call this decay path.
    """

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 2:
        raise ValueError(f"block_size must be an integer >= 2, got {block_size!r}")
    gamma = float(gamma)
    if not isfinite(gamma) or gamma <= 0:
        raise ValueError(f"gamma must be finite and positive, got {gamma!r}")
    if supervised_tail_length is None:
        supervised_tail_length = block_size - 1
    if not 0 <= supervised_tail_length <= block_size - 1:
        raise ValueError(
            "supervised_tail_length must be in [0, block_size - 1], got "
            f"{supervised_tail_length} for B{block_size}"
        )

    weights = [0.0] * block_size
    for k in range(1, supervised_tail_length + 1):
        weights[k] = exp(-float(k - 1) / gamma)
    return tuple(weights)


def vflash_physical_atom_loss_weights(
    block_size: int,
    supervised_tail_length: Optional[int] = None,
) -> Tuple[float, ...]:
    """Return uniform same-position weights for one physical-atom block.

    The clean start anchor is never supervised. Every available payload/end
    slot has weight one, and padded slots (if any) have weight zero. Physical
    atoms deliberately do not inherit DFlash's ordinary-text position decay.
    The separate ``<oof>`` multiplier is applied later by the drafter loss.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 2:
        raise ValueError(f"block_size must be an integer >= 2, got {block_size!r}")
    if supervised_tail_length is None:
        supervised_tail_length = block_size - 1
    if not 0 <= supervised_tail_length <= block_size - 1:
        raise ValueError(
            "supervised_tail_length must be in [0, block_size - 1], got "
            f"{supervised_tail_length} for B{block_size}"
        )
    return (
        (0.0,)
        + (1.0,) * supervised_tail_length
        + (0.0,) * (block_size - 1 - supervised_tail_length)
    )


def find_atom_occurrences(
    token_ids: Sequence[int],
    response_start: int,
    response_end: int,
    physical_tokens: PhysicalAtomTokenIds,
) -> Tuple[PhysicalAtomOccurrence, ...]:
    """Find fixed visual atoms and variable trajectory atoms in one response.

    The training recipe mixes boxes/points, robot and driving trajectories,
    and dense point trajectories. Fixed point/bbox atoms retain
    their canonical B4/B6 contract; trajectory atoms close at their matching
    end delimiter and keep their per-row length.
    """
    try:
        return tuple(
            PhysicalAtomOccurrence(
                start,
                end,
                spec,
                actual_block_size=(
                    end - start + 1
                    if is_variable_trajectory_atom_name(spec.schema.name)
                    else None
                ),
            )
            for start, end, spec in iter_physical_atoms(
                token_ids, response_start, response_end, physical_tokens
            )
        )
    except PhysicalAtomValidationError as error:
        raise VFlashDataError(f"{error}.") from error


def group_trajectory_occurrences(occurrences):
    """Split separate primitives while retaining adjacent joint/bimanual layouts.

    A response may repeat a trajectory in reasoning and in its final answer.
    Ordinary text/tool calls separate primitives; they are not draft payload.
    Prefer the longest supported adjacent layout so a bimanual or joint plan
    is still trained as one block. Invalid partial layouts remain errors.
    """
    patterns = sorted(
        (names for names, layout in ATOMIC_TRAJECTORY_LAYOUTS.items()
         if layout in ATOMIC_TRAJECTORY_LAYOUT_NAMES
         and not names[0].startswith("dense_")),
        key=len, reverse=True,
    )
    groups = []
    index = 0
    while index < len(occurrences):
        for names in patterns:
            group = occurrences[index:index + len(names)]
            if tuple(item.spec.schema.name for item in group) != names:
                continue
            if any(right.start != left.end + 1 for left, right in zip(group, group[1:])):
                continue
            _validate_atomic_trajectory_layout(group, ATOMIC_TRAJECTORY_LAYOUT_NAMES)
            groups.append(tuple(group))
            index += len(group)
            break
        else:
            raise VFlashDataError(
                f"Unsupported trajectory starting at token "
                f"{occurrences[index].start}: {occurrences[index].spec.schema.name}"
            )
    return tuple(groups)


def _validate_atomic_trajectory_layout(
    occurrences: Sequence[PhysicalAtomOccurrence],
    allowed_layouts: Sequence[str],
) -> str:
    names = tuple(item.spec.schema.name for item in occurrences)
    layout = ATOMIC_TRAJECTORY_LAYOUTS.get(names)
    if layout is None or layout not in allowed_layouts:
        raise VFlashDataError(
            "Atomic decoding requires one supported 3D-only or adjacent 2D+3D "
            "trajectory "
            f"layout, found {names!r}."
        )
    if any(
        right.start != left.end + 1
        for left, right in zip(occurrences, occurrences[1:])
    ):
        raise VFlashDataError(
            "Atomic decoding trajectory atoms must be adjacent with no text between them."
        )
    waypoint_counts = {
        (item.block_size - 2) // item.spec.schema.dimensions for item in occurrences
    }
    if len(waypoint_counts) != 1:
        raise VFlashDataError(
            "Atomic decoding trajectory atoms must contain the same waypoint count, "
            f"found {sorted(waypoint_counts)}."
        )
    return layout


def group_atomic_split_occurrences(
    occurrences: Sequence[PhysicalAtomOccurrence],
) -> Tuple[Tuple[PhysicalAtomOccurrence, ...], ...]:
    """Group a response's atoms into one block per trajectory.

    A 2D atom (name ending in ``_2d``) immediately followed by a 3D atom forms
    one two-atom block, exactly like the single-entity 2D+3D atomic layouts.
    Every other atom is its own block.
    """
    groups: List[Tuple[PhysicalAtomOccurrence, ...]] = []
    index = 0
    while index < len(occurrences):
        current = occurrences[index]
        following = occurrences[index + 1] if index + 1 < len(occurrences) else None
        if (
            following is not None
            and current.spec.schema.name.endswith("_2d")
            and not following.spec.schema.name.endswith("_2d")
            and following.start == current.end + 1
        ):
            groups.append((current, following))
            index += 2
        else:
            groups.append((current,))
            index += 1
    return tuple(groups)


def _as_clean_token_tuple(token_ids: Sequence[int]) -> Tuple[int, ...]:
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim != 1:
            raise ValueError(
                f"clean target input_ids must be one-dimensional, got {token_ids.shape}"
            )
        token_ids = token_ids.detach().cpu().tolist()
    try:
        clean = tuple(int(token_id) for token_id in token_ids)
    except (TypeError, ValueError) as error:
        raise ValueError("clean target input_ids must be a one-dimensional integer sequence") from error
    if not clean:
        raise ValueError("clean target input_ids must not be empty")
    return clean


def _validate_response_ranges(
    response_ranges: Sequence[Tuple[int, int]], sequence_length: int
) -> Tuple[Tuple[int, int], ...]:
    validated: List[Tuple[int, int]] = []
    previous_end = -1
    for response_index, bounds in enumerate(response_ranges):
        if len(bounds) != 2:
            raise ValueError(
                f"response range {response_index} must contain (start, end), got {bounds!r}"
            )
        start, end = int(bounds[0]), int(bounds[1])
        if not (0 <= start <= end < sequence_length):
            raise ValueError(
                f"response range {response_index} must satisfy "
                f"0 <= start <= end < {sequence_length}, got {(start, end)}"
            )
        if start <= previous_end:
            raise ValueError("response ranges must be ordered and non-overlapping")
        validated.append((start, end))
        previous_end = end
    return tuple(validated)


def _sample_candidate_indices(
    candidate_count: int, sample_count: int, rng: np.random.Generator
) -> List[int]:
    if sample_count == candidate_count:
        return list(range(candidate_count))
    selected = rng.choice(candidate_count, size=sample_count, replace=False)
    return [int(index) for index in np.asarray(selected).reshape(-1).tolist()]


def _ordinary_tail(
    clean_ids: Tuple[int, ...],
    anchor: int,
    response_end: int,
    atom_starts: frozenset[int],
    atom_nonstarts: frozenset[int],
) -> Tuple[int, ...]:
    tail: List[int] = []
    last_position = min(response_end, anchor + TEXT_VFLASH_BLOCK_SIZE - 1)
    for position in range(anchor + 1, last_position + 1):
        if position in atom_nonstarts:
            # Reaching payload without first reaching its start would indicate
            # a construction bug; never supervise a partial physical atom.
            break
        tail.append(clean_ids[position])
        if position in atom_starts:
            # The start may be learned as the final text token.  Its complete
            # tail is taught by the mandatory type-specific block next.
            break
    return tuple(tail)


def _make_block(
    clean_ids: Tuple[int, ...],
    anchor: int,
    block_size: int,
    target_tail: Tuple[int, ...],
    kind: str,
    mask_token_id: int,
    gamma: float,
    decay_loss_weights: bool,
    position_weights: Optional[Sequence[float]] = None,
) -> VFlashTrainingBlock:
    if not 0 <= anchor < len(clean_ids):
        raise ValueError(f"anchor {anchor} is outside a length-{len(clean_ids)} target")
    if not 1 <= len(target_tail) <= block_size - 1:
        raise ValueError(
            f"{kind} has {len(target_tail)} targets for B{block_size}; expected 1..{block_size - 1}"
        )
    if position_weights is not None and len(position_weights) != len(clean_ids):
        raise ValueError(
            "position_weights must carry one weight per clean target token"
        )

    draft_input_ids = (clean_ids[anchor],) + (int(mask_token_id),) * (block_size - 1)
    draft_labels = (
        (IGNORE_TOKEN_ID,)
        + target_tail
        + (IGNORE_TOKEN_ID,) * (block_size - 1 - len(target_tail))
    )
    loss_weights = (
        vflash_position_loss_weights(
            block_size,
            gamma=gamma,
            supervised_tail_length=len(target_tail),
        )
        if decay_loss_weights
        else vflash_physical_atom_loss_weights(
            block_size,
            supervised_tail_length=len(target_tail),
        )
    )
    if position_weights is not None:
        # Data-supplied per-token weights (e.g. occluded dense-trajectory
        # cells carry weight 0). Same-position slot k reconstructs clean token
        # anchor + k, so it inherits that token's weight.
        loss_weights = tuple(
            weight * float(position_weights[anchor + slot])
            if 0 < slot <= len(target_tail) else 0.0
            for slot, weight in enumerate(loss_weights)
        )
    return VFlashTrainingBlock(
        anchor_position=anchor,
        context_start=0,
        context_end=anchor,
        block_size=block_size,
        kind=kind,
        draft_input_ids=draft_input_ids,
        draft_labels=draft_labels,
        draft_position_ids=tuple(range(anchor, anchor + block_size)),
        loss_weights=loss_weights,
    )


def build_vflash_training_example(
    clean_target_input_ids: Sequence[int],
    response_ranges: Sequence[Tuple[int, int]],
    physical_tokens: PhysicalAtomTokenIds,
    ordinary_anchors_per_sequence: int = DEFAULT_ORDINARY_ANCHORS_PER_SEQUENCE,
    gamma: float = DEFAULT_VFLASH_GAMMA,
    rng: Optional[np.random.Generator] = None,
    label_weights: Optional[Sequence[float]] = None,
) -> VFlashTrainingExample:
    """Build mandatory atom blocks and randomly sampled ordinary B16 blocks.

    Response ranges are inclusive assistant-label intervals in the clean target
    sequence.  Every valid physical-atom start is selected.  Ordinary anchors
    are sampled without replacement, excluding all positions inside atoms.  A
    text block that reaches an atom start stops after supervising that trigger,
    leaving the payload to the mandatory atom block.

    The returned clean target sequence is never concatenated with draft inputs
    and carries no AR labels: VFlash trains only the separate drafter
    reconstruction loss against hidden states from the frozen target.
    """

    if (
        isinstance(ordinary_anchors_per_sequence, bool)
        or not isinstance(ordinary_anchors_per_sequence, int)
        or ordinary_anchors_per_sequence < 0
    ):
        raise ValueError("ordinary_anchors_per_sequence must be a non-negative integer")
    clean_ids = _as_clean_token_tuple(clean_target_input_ids)
    ranges = _validate_response_ranges(response_ranges, len(clean_ids))
    rng = np.random.default_rng() if rng is None else rng
    position_weights: Optional[Tuple[float, ...]] = None
    if label_weights is not None:
        position_weights = tuple(float(weight) for weight in label_weights)
        if len(position_weights) != len(clean_ids):
            raise ValueError(
                "label_weights must carry one weight per clean target token, got "
                f"{len(position_weights)} for {len(clean_ids)} tokens"
            )
        if any(weight < 0.0 or not isfinite(weight) for weight in position_weights):
            raise ValueError("label_weights must be finite and non-negative")

    all_blocks: List[VFlashTrainingBlock] = []
    ordinary_candidates: List[Tuple[int, Tuple[int, ...]]] = []
    for response_start, response_end in ranges:
        occurrences = find_atom_occurrences(
            clean_ids, response_start, response_end, physical_tokens
        )
        atom_starts = frozenset(occurrence.start for occurrence in occurrences)
        atom_nonstarts = frozenset(
            position
            for occurrence in occurrences
            for position in range(occurrence.start + 1, occurrence.end + 1)
        )
        atom_positions = atom_starts | atom_nonstarts

        fixed_atoms = [
            item for item in occurrences
            if not is_variable_trajectory_atom_name(item.spec.schema.name)
        ]
        trajectory_atoms = [
            item for item in occurrences
            if is_variable_trajectory_atom_name(item.spec.schema.name)
            and not item.spec.schema.name.startswith("dense_")
        ]
        dense_atoms = [
            item for item in occurrences
            if item.spec.schema.name.startswith("dense_")
        ]

        # Point and bbox primitives keep their fixed B4/B6 blocks.
        for occurrence in fixed_atoms:
            all_blocks.append(
                _make_block(
                    clean_ids=clean_ids,
                    anchor=occurrence.start,
                    block_size=occurrence.block_size,
                    target_tail=clean_ids[occurrence.start + 1:occurrence.end + 1],
                    kind=f"atom:fixed:{occurrence.spec.schema.name}",
                    mask_token_id=physical_tokens.mask_token_id,
                    gamma=float(gamma),
                    decay_loss_weights=False,
                    position_weights=position_weights,
                )
            )

        # Robot/driving calls produce one complete primitive per tool call.
        for trajectory_group in group_trajectory_occurrences(trajectory_atoms):
            layout = _validate_atomic_trajectory_layout(
                trajectory_group,
                allowed_layouts=ATOMIC_TRAJECTORY_LAYOUT_NAMES,
            )
            first, last = trajectory_group[0], trajectory_group[-1]
            all_blocks.append(
                _make_block(
                    clean_ids=clean_ids,
                    anchor=first.start,
                    block_size=last.end - first.start + 1,
                    target_tail=clean_ids[first.start + 1:last.end + 1],
                    kind=f"atom:trajectory:{layout}",
                    mask_token_id=physical_tokens.mask_token_id,
                    gamma=float(gamma),
                    decay_loss_weights=False,
                    position_weights=position_weights,
                )
            )

        # Dense motion repeats one independently drafted trajectory atom.
        for group in group_atomic_split_occurrences(dense_atoms):
            layout = _validate_atomic_trajectory_layout(
                group, allowed_layouts=ATOMIC_SPLIT_LAYOUT_NAMES
            )
            first, last = group[0], group[-1]
            all_blocks.append(
                _make_block(
                    clean_ids=clean_ids,
                    anchor=first.start,
                    block_size=last.end - first.start + 1,
                    target_tail=clean_ids[first.start + 1:last.end + 1],
                    kind=f"atom:dense:{layout}",
                    mask_token_id=physical_tokens.mask_token_id,
                    gamma=float(gamma),
                    decay_loss_weights=False,
                    position_weights=position_weights,
                )
            )

        # DFlash's target model produces the first response token. Drafter
        # anchors therefore start at response_start itself, never at the
        # assistant header/newline immediately before the response.
        for anchor in range(response_start, response_end):
            if anchor in atom_positions:
                continue
            target_tail = _ordinary_tail(
                clean_ids,
                anchor,
                response_end,
                atom_starts,
                atom_nonstarts,
            )
            if target_tail:
                ordinary_candidates.append((anchor, target_tail))

    # Paper contract: 512 ordinary anchors per clean sequence, sampled from
    # all response-token candidates. Mandatory project-specific atom anchors
    # are added on top so rare structured outputs are never dropped.
    sample_count = min(
        ordinary_anchors_per_sequence, len(ordinary_candidates)
    )
    for candidate_index in _sample_candidate_indices(
        len(ordinary_candidates), sample_count, rng
    ):
        anchor, target_tail = ordinary_candidates[candidate_index]
        all_blocks.append(
            _make_block(
                clean_ids=clean_ids,
                anchor=anchor,
                block_size=TEXT_VFLASH_BLOCK_SIZE,
                target_tail=target_tail,
                kind="text",
                mask_token_id=physical_tokens.mask_token_id,
                gamma=float(gamma),
                decay_loss_weights=True,
                position_weights=position_weights,
            )
        )

    all_blocks.sort(key=lambda block: (block.anchor_position, block.kind))

    return VFlashTrainingExample(
        clean_target_input_ids=clean_ids,
        draft_blocks=tuple(all_blocks),
    )


class VFlashPackedCollator:
    """Collate one clean pack and attach the drafter blocks of every sample in it.

    ``packed_collate_fn`` is the trainer's collate function for one clean pack.
    Anchors are drawn from the pack's ``_pack_seed``, so the blocks depend only
    on the pack's sample locations and survive an exact resume.
    """

    def __init__(
        self,
        physical_tokens: PhysicalAtomTokenIds,
        packed_collate_fn,
        ordinary_anchors_per_sequence: int = DEFAULT_ORDINARY_ANCHORS_PER_SEQUENCE,
        gamma: float = DEFAULT_VFLASH_GAMMA,
    ):
        self.physical_tokens = physical_tokens
        self.packed_collate_fn = packed_collate_fn
        self.ordinary_anchors_per_sequence = int(ordinary_anchors_per_sequence)
        self.gamma = float(gamma)

    def __call__(self, features):
        anchor_seed = features[0].get('_pack_seed')
        if anchor_seed is None:
            raise ValueError('VFlash packed sample is missing its deterministic anchor seed')
        result = self.packed_collate_fn(features)
        packed = build_online_packed_vflash_batch(
            packed_clean_target_input_ids=result['input_ids'],
            packed_clean_labels=result['labels'],
            clean_target_lengths=result['sub_sample_lengths'][0],
            physical_tokens=self.physical_tokens,
            ordinary_anchors_per_sequence=self.ordinary_anchors_per_sequence,
            gamma=self.gamma,
            anchor_seed=int(anchor_seed),
            packed_clean_label_weights=result.get('label_weights'),
        )
        result.update(
            vflash_draft_input_ids=packed.draft_noise_ids,
            vflash_direct_labels=packed.direct_labels,
            vflash_clean_position_ids=packed.packed_clean_target_position_ids,
            vflash_draft_position_ids=packed.position_ids,
            vflash_loss_weights=packed.loss_weights,
            vflash_block_context_starts=packed.block_context_starts,
            vflash_block_anchors=packed.block_anchors,
            vflash_block_ranges=packed.draft_block_ranges,
            vflash_block_sizes=packed.block_sizes,
        )
        return result
