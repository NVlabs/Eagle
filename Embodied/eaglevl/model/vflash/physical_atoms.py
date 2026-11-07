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

"""Schema, tokenizer and answer validation for physical atoms.

The block sizes in this module follow the DFlash convention: a block contains
one clean anchor followed by ``block_size - 1`` masked prediction slots.  A
physical atom's start token is its clean anchor, so the predicted tail contains
the numeric payload and the matching end token.

The answer contracts at the end of this module validate one assistant
answer without a tokenizer.  They share the ``atomic_decoding`` tool-call
parser with VFlash decoding.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import cached_property
from math import floor
from typing import AbstractSet, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from eaglevl.train.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    DENSE_2D_END_TOKEN,
    DENSE_2D_START_TOKEN,
    DENSE_END_TOKEN,
    DENSE_START_TOKEN,
    DRIVE_2D_START_TOKEN,
    DRIVE_END_TOKEN,
    DRIVE_START_TOKEN,
    NULL_TOKEN,
    OBJECTS_END_TOKEN,
    OBJECTS_START_TOKEN,
    OOF_TOKEN,
    POINT_END_TOKEN,
    POINT_START_TOKEN,
    ROBOT_END_TOKEN,
    ROBOT_LEFT_2D_START_TOKEN,
    ROBOT_LEFT_START_TOKEN,
    ROBOT_RIGHT_2D_START_TOKEN,
    ROBOT_RIGHT_START_TOKEN,
    ROBOT_SINGLE_2D_START_TOKEN,
    ROBOT_SINGLE_START_TOKEN,
    TEXT_MASK_TOKEN,
    number_tokens_list,
)


TEXT_VFLASH_BLOCK_SIZE = 16
NO_OBJECT_TEXT = "None"
# Loss multiplier for ``<oof>`` targets; drafter bindings record the value
# they were trained with.
DEFAULT_OOF_LOSS_SCALE = 0.1
_NO_OBJECT_LINE_RE = re.compile(
    r"^[ \t]*(?P<sentinel>none)[ \t]*$", re.IGNORECASE | re.MULTILINE
)
_REF_NO_OBJECT_RE = re.compile(
    r"</ref>[ \t]*(?P<sentinel>none)(?!\w)", re.IGNORECASE
)
_REF_TEXT_RE = re.compile(r"<ref>.*?</ref>", re.DOTALL)
_NUMBER_RE = re.compile(r"<(\d+)>")
ATOMIC_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)


@dataclass(frozen=True)
class PhysicalAtomSpec:
    name: str
    start_token: str
    end_token: str
    payload_length: int
    dimensions: int
    allow_oof_pairs: bool = False

    @property
    def block_size(self) -> int:
        """Full block size, including the clean start anchor and end token."""
        return self.payload_length + 2

    @property
    def tail_length(self) -> int:
        """Number of tokens predicted in parallel after the clean anchor."""
        return self.block_size - 1


PHYSICAL_ATOM_SPECS: Tuple[PhysicalAtomSpec, ...] = (
    PhysicalAtomSpec("point_2d", POINT_START_TOKEN, POINT_END_TOKEN, 2, 2),
    PhysicalAtomSpec("bbox_2d", BOX_START_TOKEN, BOX_END_TOKEN, 4, 2),
    PhysicalAtomSpec(
        "drive_2d", DRIVE_2D_START_TOKEN, DRIVE_END_TOKEN, 16, 2,
        allow_oof_pairs=True,
    ),
    PhysicalAtomSpec("drive_3d_bev", DRIVE_START_TOKEN, DRIVE_END_TOKEN, 16, 2),
    PhysicalAtomSpec(
        "robot_single_2d", ROBOT_SINGLE_2D_START_TOKEN, ROBOT_END_TOKEN, 20, 2
    ),
    PhysicalAtomSpec(
        "robot_left_2d", ROBOT_LEFT_2D_START_TOKEN, ROBOT_END_TOKEN, 20, 2
    ),
    PhysicalAtomSpec(
        "robot_right_2d", ROBOT_RIGHT_2D_START_TOKEN, ROBOT_END_TOKEN, 20, 2
    ),
    PhysicalAtomSpec(
        "robot_single_3d", ROBOT_SINGLE_START_TOKEN, ROBOT_END_TOKEN, 30, 3
    ),
    PhysicalAtomSpec(
        "robot_left_3d", ROBOT_LEFT_START_TOKEN, ROBOT_END_TOKEN, 30, 3
    ),
    PhysicalAtomSpec(
        "robot_right_3d", ROBOT_RIGHT_START_TOKEN, ROBOT_END_TOKEN, 30, 3
    ),
    # Dense per-trajectory point atoms (MolmoMotion 8-point forecasting). One
    # atom carries ONE query point's whole future path. The payload lengths
    # below are the F=30 defaults of the fixed contract; the dense task always
    # trains and decodes under the variable-length (atomic) contract, where the
    # end token closes the atom after any whole number of points.
    PhysicalAtomSpec("dense_2d", DENSE_2D_START_TOKEN, DENSE_2D_END_TOKEN, 60, 2),
    PhysicalAtomSpec("dense_3d", DENSE_START_TOKEN, DENSE_END_TOKEN, 90, 3),
)

# Keep this list explicit so tokenizer registration can be checked against the
# physical_atom contract.  ``<box>``, ``</box>``, and ``<oof>`` predate the
# physical-atom extension but are still part of the same structure vocabulary.
PHYSICAL_ATOM_STRUCTURE_TOKENS: Tuple[str, ...] = (
    POINT_START_TOKEN,
    POINT_END_TOKEN,
    BOX_START_TOKEN,
    BOX_END_TOKEN,
    DRIVE_2D_START_TOKEN,
    DRIVE_START_TOKEN,
    DRIVE_END_TOKEN,
    OOF_TOKEN,
    ROBOT_SINGLE_2D_START_TOKEN,
    ROBOT_LEFT_2D_START_TOKEN,
    ROBOT_RIGHT_2D_START_TOKEN,
    ROBOT_SINGLE_START_TOKEN,
    ROBOT_LEFT_START_TOKEN,
    ROBOT_RIGHT_START_TOKEN,
    ROBOT_END_TOKEN,
    DENSE_2D_START_TOKEN,
    DENSE_2D_END_TOKEN,
    DENSE_START_TOKEN,
    DENSE_END_TOKEN,
)

# These delimiters are grammar tokens rather than atoms. They are nevertheless
# validated here because visual-primitive decoding must terminate an objects
# list without retokenizing generated text.
VISUAL_PRIMITIVE_STRUCTURE_TOKENS: Tuple[str, ...] = (
    OBJECTS_START_TOKEN,
    OBJECTS_END_TOKEN,
)

# Trajectory atom families whose waypoint count may vary under the atomic
# (variable-length) contract. Point and box atoms always stay fixed-length.
VARIABLE_TRAJECTORY_ATOM_PREFIXES: Tuple[str, ...] = ("drive_", "robot_", "dense_")

# Atom sequences that form one complete trajectory primitive, keyed to their
# layout names.
ATOMIC_TRAJECTORY_LAYOUTS: Mapping[Tuple[str, ...], str] = {
    ("drive_2d",): "drive_2d",
    ("drive_3d_bev",): "drive_3d",
    ("drive_2d", "drive_3d_bev"): "drive_2d3d",
    ("robot_single_2d",): "robot_single_2d",
    ("robot_single_3d",): "robot_single_3d",
    ("robot_single_2d", "robot_single_3d"): "robot_single_2d3d",
    ("robot_left_2d", "robot_right_2d"): "robot_dual_2d",
    ("robot_left_3d", "robot_right_3d"): "robot_dual_3d",
    (
        "robot_left_2d",
        "robot_left_3d",
        "robot_right_2d",
        "robot_right_3d",
    ): "robot_dual_2d3d",
    ("dense_3d",): "dense_3d",
    ("dense_2d", "dense_3d"): "dense_2d3d",
}
ATOMIC_TRAJECTORY_LAYOUT_NAMES: Tuple[str, ...] = tuple(
    dict.fromkeys(ATOMIC_TRAJECTORY_LAYOUTS.values())
)


def is_variable_trajectory_atom_name(name: str) -> bool:
    """Return whether ``name`` is a drive/robot/dense trajectory atom."""
    return str(name).startswith(VARIABLE_TRAJECTORY_ATOM_PREFIXES)


def validate_oof_loss_scale(value: float) -> float:
    """Return a validated positive finite ``<oof>`` loss multiplier."""

    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"oof_loss_scale must be finite and positive, got {value!r}")
    return value


@dataclass(frozen=True)
class ResolvedPhysicalAtomSpec:
    schema: PhysicalAtomSpec
    start_id: int
    end_id: int

    @property
    def block_size(self) -> int:
        return self.schema.block_size

    @property
    def tail_length(self) -> int:
        return self.schema.tail_length


@dataclass(frozen=True)
class PhysicalAtomTokenIds:
    specs_by_start_id: Mapping[int, ResolvedPhysicalAtomSpec]
    numeric_id_to_value: Mapping[int, int]
    oof_token_id: int
    mask_token_id: int
    null_token_id: int
    objects_start_token_id: int
    objects_end_token_id: int

    @cached_property
    def serialized_specs_by_start_id(self) -> Mapping[int, Mapping]:
        """The atom specs of ``physical_atom_config(self)``, built once."""
        return _atom_specs_by_start_id(physical_atom_config(self))


class PhysicalAtomValidationError(ValueError):
    """Raised when training text contains a malformed serialized atom."""


def _single_token_id(tokenizer, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Physical-atom token must encode to exactly one ID: {token!r} -> {token_ids}"
        )
    return int(token_ids[0])


def validate_physical_atom_tokenizer(tokenizer) -> PhysicalAtomTokenIds:
    """Validate atomic structure/numeric tokens and all fixed atom lengths."""
    structure_ids = {
        token: _single_token_id(tokenizer, token)
        for token in PHYSICAL_ATOM_STRUCTURE_TOKENS
        + VISUAL_PRIMITIVE_STRUCTURE_TOKENS
    }
    numeric_ids = {
        token: _single_token_id(tokenizer, token) for token in number_tokens_list
    }
    mask_token_id = _single_token_id(tokenizer, TEXT_MASK_TOKEN)
    null_token_id = _single_token_id(tokenizer, NULL_TOKEN)

    if len(set(numeric_ids.values())) != len(number_tokens_list):
        raise ValueError("The 1001 physical numeric tokens do not have distinct token IDs")
    registered_ids = (
        list(structure_ids.values())
        + list(numeric_ids.values())
        + [mask_token_id, null_token_id]
    )
    if len(set(registered_ids)) != len(registered_ids):
        raise ValueError(
            "Physical structure, numeric, mask, and legacy-null tokens must have distinct IDs"
        )

    specs_by_start_id: Dict[int, ResolvedPhysicalAtomSpec] = {}
    for spec in PHYSICAL_ATOM_SPECS:
        example = spec.start_token + (number_tokens_list[0] * spec.payload_length) + spec.end_token
        encoded = tokenizer.encode(example, add_special_tokens=False)
        if len(encoded) != spec.block_size:
            raise ValueError(
                f"{spec.name} must encode to {spec.block_size} tokens, got {len(encoded)}: {encoded}"
            )
        resolved = ResolvedPhysicalAtomSpec(
            schema=spec,
            start_id=structure_ids[spec.start_token],
            end_id=structure_ids[spec.end_token],
        )
        specs_by_start_id[resolved.start_id] = resolved

    if len(specs_by_start_id) != len(PHYSICAL_ATOM_SPECS):
        raise ValueError("Physical-atom start tokens must have distinct token IDs")

    return PhysicalAtomTokenIds(
        specs_by_start_id=specs_by_start_id,
        numeric_id_to_value={token_id: i for i, token_id in enumerate(numeric_ids.values())},
        oof_token_id=structure_ids[OOF_TOKEN],
        mask_token_id=mask_token_id,
        null_token_id=null_token_id,
        objects_start_token_id=structure_ids[OBJECTS_START_TOKEN],
        objects_end_token_id=structure_ids[OBJECTS_END_TOKEN],
    )


def _atom_specs_by_start_id(physical_config: Optional[Mapping]) -> dict[int, Mapping]:
    if not physical_config:
        return {}
    return {
        int(spec["start_id"]): spec
        for spec in physical_config.get("atoms", {}).values()
    }


def _atom_payload_error(
    payload: Sequence[int], allow_oof_pairs: bool, numeric_ids, oof_token_id
) -> Optional[str]:
    """Return ``"oof"`` or ``"numeric"`` for a malformed atom payload, else None.

    Payload tokens are numeric. An atom with ``allow_oof_pairs`` may instead
    fill both coordinates of a waypoint with ``<oof>``.
    """
    if allow_oof_pairs:
        oof = int(oof_token_id)
        for offset in range(0, len(payload), 2):
            pair = payload[offset:offset + 2]
            if not (all(token_id in numeric_ids for token_id in pair) or pair == [oof, oof]):
                return "oof"
    elif not all(token_id in numeric_ids for token_id in payload):
        return "numeric"
    return None


def _validate_atom_tail(
    spec: Mapping,
    tail_token_ids: Sequence[int],
    numeric_id_to_value: Mapping[int, int],
    oof_token_id: Optional[int],
    *,
    variable_length: bool = False,
) -> Tuple[bool, str]:
    """Validate the tail after an atom's start token against its serialized spec.

    With ``variable_length``, a drive, robot or dense trajectory may hold any
    whole number of waypoints; otherwise every atom has its fixed length.
    VFlash generation records the returned reason in its stats.
    """
    tail = [int(token_id) for token_id in tail_token_ids]
    variable = variable_length and is_variable_trajectory_atom_name(spec["name"])
    if variable:
        if len(tail) < int(spec["dimensions"]) + 1:
            return False, (
                f"expected at least {int(spec['dimensions']) + 1} tail tokens, "
                f"got {len(tail)}"
            )
    elif len(tail) != int(spec["tail_length"]):
        return False, f"expected {spec['tail_length']} tail tokens, got {len(tail)}"
    if tail[-1] != int(spec["end_id"]):
        return False, f"expected end token ID {spec['end_id']}, got {tail[-1]}"

    payload = tail[:-1]
    if variable and len(payload) % int(spec["dimensions"]):
        return False, (
            f"payload length {len(payload)} is not divisible by dimension "
            f"{spec['dimensions']}"
        )
    error = _atom_payload_error(
        payload, spec.get("allow_oof_pairs", False), numeric_id_to_value, oof_token_id
    )
    if error == "oof":
        return False, "2D driving <oof> must occupy both coordinates of a waypoint"
    if error == "numeric":
        return False, "atom payload contains a non-numeric token"

    if spec.get("name") == "bbox_2d":
        x1, y1, x2, y2 = [numeric_id_to_value[token_id] for token_id in payload]
        if not (x1 < x2 and y1 < y2):
            return False, "bbox must satisfy x1 < x2 and y1 < y2"
    return True, ""


def validate_serialized_atom_tail(
    start_token_id: int,
    tail_token_ids: Sequence[int],
    physical_config: Mapping,
) -> Tuple[bool, str]:
    """Validate one fixed-length atom tail against a ``physical_atom_config``."""
    spec = _atom_specs_by_start_id(physical_config).get(int(start_token_id))
    if spec is None:
        return False, f"unknown atom start token ID {start_token_id}"
    numeric = {
        int(token_id): int(value)
        for token_id, value in physical_config.get("numeric_id_to_value", {}).items()
    }
    return _validate_atom_tail(
        spec, tail_token_ids, numeric, physical_config.get("oof_token_id")
    )


def iter_physical_atoms(
    token_ids: Sequence[int],
    response_start: int,
    response_end: int,
    physical_tokens: PhysicalAtomTokenIds,
    orphan_ids: AbstractSet[int] = frozenset(),
) -> Iterator[Tuple[int, int, ResolvedPhysicalAtomSpec]]:
    """Yield ``(start, end, spec)`` for every atom in one assistant response.

    ``end`` is the position of the atom's end token. Point and box atoms have
    their fixed length; a drive, robot or dense trajectory ends at its first
    end token. Each atom is validated before it is yielded, and a token from
    ``orphan_ids`` outside an atom is an error.
    """
    serialized_specs = physical_tokens.serialized_specs_by_start_id
    position = int(response_start)
    while position <= response_end:
        token_id = int(token_ids[position])
        spec = physical_tokens.specs_by_start_id.get(token_id)
        if spec is None:
            if token_id in orphan_ids:
                raise PhysicalAtomValidationError(
                    f"orphan physical structure token ID {token_id} at "
                    f"assistant token {position}"
                )
            position += 1
            continue

        atom_end = position + spec.tail_length
        if is_variable_trajectory_atom_name(spec.schema.name):
            atom_end = position + 1
            while atom_end <= response_end and int(token_ids[atom_end]) != spec.end_id:
                atom_end += 1
        if atom_end > response_end:
            raise PhysicalAtomValidationError(
                f"{spec.schema.name} atom at assistant token {position} extends "
                f"past response end {response_end}"
            )
        valid, reason = _validate_atom_tail(
            serialized_specs[token_id],
            token_ids[position + 1:atom_end + 1],
            physical_tokens.numeric_id_to_value,
            physical_tokens.oof_token_id,
            variable_length=True,
        )
        if not valid:
            raise PhysicalAtomValidationError(
                f"malformed {spec.schema.name} atom at assistant token "
                f"{position}: {reason}"
            )
        yield position, atom_end, spec
        position = atom_end + 1


def validate_physical_atom_sequence(
    token_ids: Sequence[int],
    response_start: int,
    response_end: int,
    physical_tokens: PhysicalAtomTokenIds,
) -> None:
    """Validate all reserved structure tokens in one assistant response.

    Numeric ``<0>``..``<1000>`` tokens may occur in ordinary text. Atom starts,
    atom ends, and ``<oof>`` are reserved and must belong to one complete
    atom. AR and VFlash training therefore share one wire format.
    """
    if not (0 <= response_start <= response_end < len(token_ids)):
        raise ValueError(
            f"invalid response interval {response_start}:{response_end} for "
            f"sequence length {len(token_ids)}"
        )
    end_ids = {
        int(spec.end_id) for spec in physical_tokens.specs_by_start_id.values()
    }
    reserved_nonstart_ids = end_ids | {
        int(physical_tokens.oof_token_id),
        int(physical_tokens.mask_token_id),
        int(physical_tokens.null_token_id),
    }
    for _ in iter_physical_atoms(
        token_ids, response_start, response_end, physical_tokens,
        orphan_ids=reserved_nonstart_ids,
    ):
        pass


def quantize_physical_value(value: float, minimum: float, maximum: float) -> int:
    """Quantize with strict range checking and round-half-up semantics."""
    if not minimum < maximum:
        raise ValueError(f"Expected minimum < maximum, got [{minimum}, {maximum}]")
    if not minimum <= value <= maximum:
        raise ValueError(f"Value {value} is outside [{minimum}, {maximum}]")
    scaled = (value - minimum) / (maximum - minimum) * 1000.0
    return int(floor(scaled + 0.5))


def dequantize_physical_value(token_value: int, minimum: float, maximum: float) -> float:
    if not 0 <= token_value <= 1000:
        raise ValueError(f"Physical numeric token must be in [0, 1000], got {token_value}")
    if not minimum < maximum:
        raise ValueError(f"Expected minimum < maximum, got [{minimum}, {maximum}]")
    return minimum + token_value / 1000.0 * (maximum - minimum)


def physical_atom_config(token_ids: PhysicalAtomTokenIds) -> dict:
    """Return a JSON-serializable token/schema map for exported inference code."""
    atoms = {}
    for start_id, resolved in token_ids.specs_by_start_id.items():
        atoms[resolved.schema.start_token] = {
            "name": resolved.schema.name,
            "start_id": int(start_id),
            "end_token": resolved.schema.end_token,
            "end_id": int(resolved.end_id),
            "block_size": int(resolved.block_size),
            "tail_length": int(resolved.tail_length),
            "payload_length": int(resolved.schema.payload_length),
            "dimensions": int(resolved.schema.dimensions),
            "allow_oof_pairs": bool(resolved.schema.allow_oof_pairs),
        }
    return {
        "version": 1,
        "text_block_size": TEXT_VFLASH_BLOCK_SIZE,
        "no_object_text": NO_OBJECT_TEXT,
        "mask_token_id": int(token_ids.mask_token_id),
        "null_token_id": int(token_ids.null_token_id),
        "objects_start_token_id": int(token_ids.objects_start_token_id),
        "objects_end_token_id": int(token_ids.objects_end_token_id),
        "visual_primitive_max_count": 900,
        "oof_token_id": int(token_ids.oof_token_id),
        "numeric_id_to_value": {
            str(token_id): int(value)
            for token_id, value in token_ids.numeric_id_to_value.items()
        },
        "atoms": atoms,
    }


def validate_no_object_text(text: str) -> None:
    """Enforce the Eagle Embodied no-object serialization contract.

    ``None`` is ordinary text rather than a special token. Only unambiguous
    sentinel positions are checked: a line consisting solely of ``None`` or a
    null marker immediately following ``</ref>``. Natural prose and OCR labels
    containing the word "none" are not no-object sentinels.
    """
    if text.strip().lower() in {"[]", "null", "n/a"}:
        raise ValueError(
            f"No-object responses must use {NO_OBJECT_TEXT!r} with exact "
            f"capitalization, got legacy null target {text.strip()!r}"
        )

    cursor = 0
    outside_ref_segments: list[str] = []
    for ref_match in _REF_TEXT_RE.finditer(text):
        outside_ref_segments.append(text[cursor : ref_match.start()])
        cursor = ref_match.end()
    outside_ref_segments.append(text[cursor:])
    matches = [
        match
        for segment in outside_ref_segments
        for match in _NO_OBJECT_LINE_RE.finditer(segment)
    ]
    matches.extend(_REF_NO_OBJECT_RE.finditer(text))
    for match in matches:
        if match.group("sentinel") != NO_OBJECT_TEXT:
            raise ValueError(
                f"No-object responses must use {NO_OBJECT_TEXT!r} with exact "
                f"capitalization, got {match.group('sentinel')!r}"
            )


def parse_atomic_tool_arguments(text: str, strict: bool = True) -> dict | None:
    """Return the arguments of the latest ``atomic_decoding`` tool call.

    Returns None when ``text`` contains no tool call.  ``block_size`` is
    mandatory and validated here; task-specific keys (the dense trajectory task
    repeats ``count``, ``fps`` and ``scale_m``) are passed through for the
    caller.

    ``strict=False`` is for scoring answers as generated: the arguments are
    returned without validation, and a malformed call or a call to another
    function yields None instead of an error.
    """
    matches = list(ATOMIC_TOOL_CALL_RE.finditer(text))
    if not matches:
        return None
    if not strict:
        try:
            call = json.loads(matches[-1].group(1))
        except json.JSONDecodeError:
            return None
        if call.get("name") != "atomic_decoding" or not isinstance(call.get("arguments"), dict):
            return None
        return dict(call["arguments"])
    try:
        call = json.loads(matches[-1].group(1))
        arguments = call["arguments"]
        size = arguments["block_size"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("malformed atomic_decoding tool call") from error
    if call.get("name") != "atomic_decoding":
        raise ValueError(f"expected atomic_decoding tool call, got {call.get('name')!r}")
    if not isinstance(arguments, dict):
        raise ValueError("malformed atomic_decoding tool call")
    if isinstance(size, bool) or not isinstance(size, int) or size < 4:
        raise ValueError(f"atomic_decoding block_size must be an integer >= 4, got {size!r}")
    return dict(arguments)


@dataclass(frozen=True)
class AtomSchema:
    name: str
    dimensions: int
    start_tokens: tuple[str, ...]
    end_token: str
    variable_points: bool = True
    fixed_points: int | None = None
    multiple_atoms: bool = False


SCHEMAS: Mapping[str, AtomSchema] = {
    "robot_2d": AtomSchema(
        "robot_2d", 2,
        (ROBOT_SINGLE_2D_START_TOKEN, ROBOT_LEFT_2D_START_TOKEN, ROBOT_RIGHT_2D_START_TOKEN),
        ROBOT_END_TOKEN, multiple_atoms=True,
    ),
    "robot_3d": AtomSchema(
        "robot_3d", 3,
        (ROBOT_SINGLE_START_TOKEN, ROBOT_LEFT_START_TOKEN, ROBOT_RIGHT_START_TOKEN),
        ROBOT_END_TOKEN, fixed_points=10, multiple_atoms=True,
    ),
    "driving_xyh": AtomSchema(
        "driving_xyh", 3, (DRIVE_START_TOKEN,), DRIVE_END_TOKEN,
    ),
    "dense_3d": AtomSchema(
        "dense_3d", 3, (DENSE_START_TOKEN,), DENSE_END_TOKEN, multiple_atoms=True,
    ),
    "point_2d": AtomSchema(
        "point_2d", 2, (POINT_START_TOKEN,), POINT_END_TOKEN, fixed_points=1,
        multiple_atoms=True,
    ),
    "bbox_2d": AtomSchema(
        "bbox_2d", 4, (BOX_START_TOKEN,), BOX_END_TOKEN, fixed_points=1,
        multiple_atoms=True,
    ),
}


@dataclass(frozen=True)
class AnswerReport:
    task: str
    atom_count: int
    points_per_atom: tuple[int, ...]
    block_size: int | None
    tool_count: int | None


def _atoms(text: str, schema: AtomSchema) -> list[tuple[str, list[int]]]:
    matches: list[tuple[int, str, list[int]]] = []
    for start in schema.start_tokens:
        pattern = re.compile(re.escape(start) + r"(.*?)" + re.escape(schema.end_token), re.DOTALL)
        for match in pattern.finditer(text):
            payload = match.group(1)
            numbers = [int(value) for value in _NUMBER_RE.findall(payload)]
            residue = _NUMBER_RE.sub("", payload).strip()
            if residue:
                raise ValueError(f"{schema.name} payload contains non-coordinate text: {residue!r}")
            matches.append((match.start(), start, numbers))
    return [(start, numbers) for _, start, numbers in sorted(matches)]


def validate_answer(
    text: str,
    task: str,
    *,
    expected_points: int | None = None,
    expected_atoms: int | None = None,
) -> AnswerReport:
    """Validate one assistant answer without depending on a tokenizer."""
    calls = list(ATOMIC_TOOL_CALL_RE.finditer(text))
    if len(calls) > 1:
        # CoT may state a plan in its reasoning and repeat it in the answer.
        # Each tool call sizes its own following primitive, not every atom in
        # the entire assistant message.
        reports = [validate_answer(
            text[call.start():calls[i+1].start() if i+1 < len(calls) else len(text)],
            task, expected_points=expected_points, expected_atoms=expected_atoms,
        ) for i, call in enumerate(calls)]
        return AnswerReport(
            reports[0].task, sum(r.atom_count for r in reports),
            tuple(p for r in reports for p in r.points_per_atom), None, None,
        )
    if task == "auto":
        if DENSE_START_TOKEN in text:
            task = "dense_3d"
        elif DRIVE_START_TOKEN in text:
            task = "driving_xyh"
        elif any(token in text for token in SCHEMAS["robot_2d"].start_tokens):
            task = "robot_2d"
        elif any(token in text for token in SCHEMAS["robot_3d"].start_tokens):
            task = "robot_3d"
        elif BOX_START_TOKEN in text:
            task = "bbox_2d"
        elif POINT_START_TOKEN in text:
            task = "point_2d"
        else:
            task = "text"
    if task == "text":
        if not text.strip():
            raise ValueError("text answer must not be empty")
        return AnswerReport(task, 0, (), None, None)
    if task not in SCHEMAS:
        raise ValueError(
            f"unknown task {task!r}; choose from auto, text, {sorted(SCHEMAS)}"
        )
    schema = SCHEMAS[task]
    atoms = _atoms(text, schema)
    if not atoms:
        if task == "bbox_2d" and (
            OBJECTS_START_TOKEN + NO_OBJECT_TEXT + OBJECTS_END_TOKEN
        ) in text:
            return AnswerReport(task, 0, (), None, None)
        raise ValueError(f"{task} answer contains no {schema.start_tokens} atom")
    if not schema.multiple_atoms and len(atoms) != 1:
        raise ValueError(f"{task} requires exactly one atom, found {len(atoms)}")
    if expected_atoms is not None and len(atoms) != expected_atoms:
        raise ValueError(f"{task} requires {expected_atoms} atoms, found {len(atoms)}")

    points = []
    for start, values in atoms:
        if any(value < 0 or value > 1000 for value in values):
            raise ValueError(f"{task} contains a coordinate outside [0, 1000]")
        if len(values) == 0 or len(values) % schema.dimensions:
            raise ValueError(
                f"{start} has {len(values)} coordinates, not divisible by {schema.dimensions}"
            )
        point_count = len(values) // schema.dimensions
        required = expected_points if expected_points is not None else schema.fixed_points
        if required is not None and point_count != required:
            raise ValueError(f"{start} requires {required} points, found {point_count}")
        if task == "bbox_2d":
            x1, y1, x2, y2 = values
            if not (x1 < x2 and y1 < y2):
                raise ValueError("bbox_2d must satisfy x1 < x2 and y1 < y2")
        points.append(point_count)

    arguments = parse_atomic_tool_arguments(text)
    if arguments is not None and "count" in arguments and (
        isinstance(arguments["count"], bool)
        or not isinstance(arguments["count"], int)
        or arguments["count"] < 1
    ):
        raise ValueError("atomic_decoding count must be a positive integer")
    needs_tool = task in {"robot_2d", "robot_3d", "driving_xyh", "dense_3d"}
    if needs_tool and arguments is None:
        raise ValueError(f"{task} requires an atomic_decoding tool call")
    block_size = None if arguments is None else int(arguments["block_size"])
    tool_count = None if arguments is None else arguments.get("count")
    if arguments is not None:
        if task == "dense_3d":
            expected_block = schema.dimensions * points[0] + 2
            if len(set(points)) != 1:
                raise ValueError("all dense_3d atoms must use the same future horizon")
            if tool_count is not None and tool_count != len(atoms):
                raise ValueError("dense_3d tool count must equal the number of trajectories")
        else:
            expected_block = sum(schema.dimensions * count + 2 for count in points)
            if tool_count is not None and len(points) == 1 and tool_count != points[0]:
                raise ValueError("trajectory tool count must equal its waypoint count")
        if block_size != expected_block:
            raise ValueError(
                f"{task} tool block_size is B{block_size}, expected B{expected_block}"
            )
    return AnswerReport(task, len(atoms), tuple(points), block_size, tool_count)
