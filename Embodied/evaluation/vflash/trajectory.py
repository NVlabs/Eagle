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

"""Open-loop trajectory metrics for the robot3d and driving benchmarks.

Predictions are never repaired: the primary metrics require the expected arms
and the exact waypoint count. Summarize prediction files with

    python -m evaluation.vflash.trajectory robot3d_agibot.jsonl [--output metrics.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eaglevl.model.vflash.physical_atoms import dequantize_physical_value, parse_atomic_tool_arguments

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_ATOM_RE = re.compile(
    r"<(drive_2d|drive|robot_(?:single|left|right)(?:_2d)?)>"
    r"((?:<(?:\d+|oof)>)+)</(drive|robot)>"
)
_VALUE_RE = re.compile(r"<(\d+|oof)>")
# Physical ranges of the robot (x, y, z) and legacy 2D drive (x, y) atoms.
_ROBOT_RANGES = ((-5.0, 5.0), (-5.0, 5.0), (0.0, 10.0))
_DRIVE_RANGES = ((-10.0, 200.0), (-50.0, 50.0))


@dataclass
class TrajectoryArm:
    name: str
    trace_2d: list[list[float | None]] = field(default_factory=list)
    trace_3d: list[list[float]] = field(default_factory=list)


@dataclass
class TrajectoryAnswer:
    arms: list[TrajectoryArm]
    wire_format: str
    action: str | None = None
    parse_errors: list[str] = field(default_factory=list)
    requested_block_size: int | None = None
    observed_block_size: int = 0


def _points(values, dimensions: int) -> list[list[Any]]:
    return [list(values[index:index + dimensions]) for index in range(0, len(values), dimensions)]


def parse_trajectory_text(text: str) -> TrajectoryAnswer:
    """Parse the physical trajectory atoms of one answer.

    Coordinates outside ``[0, 1000]`` are reported as parse errors and decoded
    at the nearest bound. An answer without atoms has no arms.
    """
    matches = list(_ATOM_RE.finditer(text))
    if not matches:
        return TrajectoryAnswer(
            arms=[],
            wire_format="bracket",
            action=_action(text),
            parse_errors=["no trajectory atoms or bracket trajectories found"],
        )
    arms: dict[str, TrajectoryArm] = {}
    order: list[str] = []
    errors = []
    if "".join(match.group(0) for match in matches) != text[matches[0].start():matches[-1].end()]:
        errors.append("physical trajectory atoms are not adjacent")
    if text[matches[-1].end():].strip():
        errors.append("unexpected text after physical trajectory")
    observed = 0
    for match in matches:
        kind = match.group(1)
        raw_values = _VALUE_RE.findall(match.group(2))
        expected_end = "drive" if kind.startswith("drive") else "robot"
        if match.group(3) != expected_end:
            errors.append(f"{kind} has wrong closing tag </{match.group(3)}>")
        values: list[int | None] = []
        for raw in raw_values:
            if raw == "oof":
                values.append(None)
            else:
                integer = int(raw)
                if not 0 <= integer <= 1000:
                    errors.append(f"{kind} coordinate {integer} outside [0, 1000]")
                values.append(integer)
        dimensions = 2 if kind in {"drive_2d", "drive"} or kind.endswith("_2d") else 3
        if len(values) % dimensions:
            errors.append(f"{kind} payload is not divisible by {dimensions}")
            continue
        is_2d = kind.endswith("_2d")
        if any(value is None for value in values):
            if kind != "drive_2d" or any(
                (values[index] is None) != (values[index + 1] is None)
                for index in range(0, len(values), 2)
            ):
                errors.append(f"invalid <oof> coordinates in {kind}")
        if kind.startswith("drive"):
            name = "ego"
        elif "left" in kind:
            name = "left"
        elif "right" in kind:
            name = "right"
        else:
            name = "single"
        if name not in arms:
            arms[name] = TrajectoryArm(name=name)
            order.append(name)
        if is_2d:
            decoded = [None if value is None else value / 1000.0 for value in values]
            if arms[name].trace_2d:
                errors.append(f"duplicate 2D atom for {name}")
            arms[name].trace_2d = _points(decoded, 2)
        else:
            ranges = _DRIVE_RANGES if kind == "drive" else _ROBOT_RANGES
            decoded = [
                dequantize_physical_value(
                    min(max(0 if value is None else value, 0), 1000), minimum, maximum
                )
                for value, (minimum, maximum) in zip(values, ranges * (len(values) // dimensions))
            ]
            if arms[name].trace_3d:
                errors.append(f"duplicate 3D atom for {name}")
            arms[name].trace_3d = _points(decoded, dimensions)
        observed += len(values) + 2

    # Any integer block size counts as requested; a mismatch only clears
    # ``dynamic_block_ok`` and does not invalidate the trajectory.
    requested = None
    arguments = parse_atomic_tool_arguments(text, strict=False)
    if arguments is not None:
        value = arguments.get("block_size")
        if isinstance(value, int) and not isinstance(value, bool):
            requested = value
        else:
            errors.append("atomic_decoding block_size is not an integer")
    elif "<tool_call>" in text:
        errors.append("malformed atomic_decoding call")
    return TrajectoryAnswer(
        arms=[arms[name] for name in order],
        wire_format="physical",
        action=_action(text),
        parse_errors=errors,
        requested_block_size=requested,
        observed_block_size=observed,
    )


def _action(text: str) -> str | None:
    match = _ACTION_RE.search(text)
    return match.group(1).strip() if match else None


def _arm_map(answer: TrajectoryAnswer) -> dict[str, TrajectoryArm]:
    mapped = {arm.name.lower(): arm for arm in answer.arms}
    if len(mapped) == 1:
        only = next(iter(mapped.values()))
        mapped.setdefault("single", only)
        mapped.setdefault("ego", only)
    return mapped


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _direction_score(prediction: np.ndarray, target: np.ndarray) -> float:
    scores = []
    for pred_step, target_step in zip(np.diff(prediction, axis=0), np.diff(target, axis=0)):
        pred_norm = np.linalg.norm(pred_step)
        target_norm = np.linalg.norm(target_step)
        if pred_norm > 1e-9 and target_norm > 1e-9:
            scores.append(float(np.dot(pred_step, target_step) / (pred_norm * target_norm)))
    return float(np.mean(scores)) if scores else 1.0


def _shape_score(prediction: np.ndarray, target: np.ndarray) -> float:
    def normalize(trajectory):
        centered = trajectory - trajectory[0]
        arc = np.linalg.norm(np.diff(centered, axis=0), axis=1).sum()
        return centered / arc if arc > 1e-9 else centered

    pred_norm = normalize(prediction)
    target_norm = normalize(target)
    distance = np.linalg.norm(pred_norm - target_norm, axis=1).sum()
    maximum = len(target) * np.sqrt(target.shape[1])
    return float(max(0.0, 1.0 - distance / maximum))


def score_trajectory(
    prediction: TrajectoryAnswer,
    target: TrajectoryAnswer,
    *,
    expected_points: int | None = None,
    space: str = "robot_xyz",
    require_physical: bool = False,
    require_2d: bool = False,
) -> dict[str, Any]:
    """Score exact-length open-loop trajectories without shape repair.

    Only target arms that move (summed per-axis std >= 5 mm) are compared.
    """
    pred_arms, target_arms = _arm_map(prediction), _arm_map(target)
    target_names = [arm.name.lower() for arm in target.arms]
    if target_names == ["single"] and "single" in pred_arms:
        required_names = ["single"]
    elif target_names == ["ego"] and "ego" in pred_arms:
        required_names = ["ego"]
    else:
        required_names = target_names
    active_names = [
        name for name in required_names
        if len(target_arms[name].trace_3d) < 2
        or float(np.asarray(target_arms[name].trace_3d).std(axis=0).sum()) >= 0.005
    ]
    comparable_names = active_names or required_names
    expected = expected_points or (
        len(target.arms[0].trace_3d) if target.arms and target.arms[0].trace_3d else None
    )
    general_failures: list[str] = []
    failures_2d: list[str] = []
    failures_3d: list[str] = []

    def add_unique(destination: list[str], reason: str) -> None:
        if reason not in destination:
            destination.append(reason)

    if require_physical and prediction.wire_format != "physical":
        add_unique(general_failures, "wrong_wire_format")
    if not prediction.arms:
        add_unique(general_failures, "no_trajectory")
    for error in prediction.parse_errors:
        lowered = error.lower()
        if "2d" in lowered or "<oof>" in lowered:
            add_unique(failures_2d, f"parse_2d:{error}")
        elif "3d" in lowered or any(
            marker in lowered
            for marker in ("drive ", "robot_single ", "robot_left ", "robot_right ")
        ):
            add_unique(failures_3d, f"parse_3d:{error}")
        else:
            add_unique(general_failures, f"parse:{error}")
    missing_names = [name for name in required_names if name not in pred_arms]
    if missing_names:
        reason = "missing_arm:" + ",".join(sorted(missing_names))
        add_unique(failures_3d, reason)
        if require_2d:
            add_unique(failures_2d, reason)
    if expected is None:
        add_unique(general_failures, "bad_target")
    else:
        for name in required_names:
            if name not in pred_arms:
                continue
            observed_3d = len(pred_arms[name].trace_3d)
            if observed_3d != expected:
                add_unique(
                    failures_3d,
                    f"wrong_3d_length:{name}:{observed_3d}!={expected}",
                )
            if require_2d:
                observed_2d = len(pred_arms[name].trace_2d)
                if observed_2d != expected:
                    add_unique(
                        failures_2d,
                        f"wrong_2d_length:{name}:{observed_2d}!={expected}",
                    )

    format_3d_ok = not general_failures and not failures_3d
    format_2d_ok = (
        not general_failures and not failures_2d if require_2d else None
    )
    format_ok = format_3d_ok and (format_2d_ok is not False)
    if format_ok:
        status = "ok"
    elif general_failures:
        status = general_failures[0].split(":", 1)[0]
    elif failures_3d:
        status = failures_3d[0].split(":", 1)[0]
    else:
        status = failures_2d[0].split(":", 1)[0]

    errors_3d: list[float] = []
    squared_errors_3d: list[float] = []
    coordinate_squared_errors_3d: list[float] = []
    final_3d: list[float] = []
    lateral: list[float] = []
    depth: list[float] = []
    errors_2d: list[float] = []
    final_2d: list[float] = []
    direction_3d: list[float] = []
    shape_3d: list[float] = []
    direction_2d: list[float] = []
    shape_2d: list[float] = []
    if format_3d_ok:
        for name in comparable_names:
            pred_arm, target_arm = pred_arms[name], target_arms[name]
            pred3 = np.asarray(pred_arm.trace_3d, dtype=np.float64)
            target3 = np.asarray(target_arm.trace_3d, dtype=np.float64)
            if pred3.shape != target3.shape:
                status = "target_shape_mismatch"
                format_3d_ok = False
                format_ok = False
                add_unique(failures_3d, "target_shape_mismatch")
                break
            delta = pred3 - target3
            per_point = np.linalg.norm(delta, axis=1)
            errors_3d.extend(per_point.tolist())
            squared_errors_3d.extend(np.square(per_point).tolist())
            coordinate_squared_errors_3d.extend(np.square(delta).reshape(-1).tolist())
            final_3d.append(float(per_point[-1]))
            direction_3d.append(_direction_score(pred3, target3))
            shape_3d.append(_shape_score(pred3, target3))
            if space == "drive_bev":
                depth.extend(np.abs(delta[:, 0]).tolist())
                lateral.extend(np.abs(delta[:, 1]).tolist())
            else:
                lateral.extend(np.linalg.norm(delta[:, :2], axis=1).tolist())
                if delta.shape[1] >= 3:
                    depth.extend(np.abs(delta[:, 2]).tolist())
    if format_2d_ok:
        for name in comparable_names:
            pred_arm, target_arm = pred_arms[name], target_arms[name]
            if len(target_arm.trace_2d) != expected:
                format_2d_ok = False
                format_ok = False
                status = "target_2d_shape_mismatch"
                add_unique(failures_2d, "target_2d_shape_mismatch")
                break
            pairs = [
                (p, t) for p, t in zip(pred_arm.trace_2d, target_arm.trace_2d)
                if None not in p and None not in t
            ]
            if pairs:
                p2 = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
                t2 = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
                e2 = np.linalg.norm(p2 - t2, axis=1)
                errors_2d.extend(e2.tolist())
                final_2d.append(float(e2[-1]))
                if len(pairs) == expected:
                    direction_2d.append(_direction_score(p2, t2))
                    shape_2d.append(_shape_score(p2, t2))

    ade = _mean(errors_3d) if format_3d_ok else None
    activity_matches = []
    for name in required_names:
        if name not in pred_arms:
            continue
        target_active = name in active_names
        pred_trace = pred_arms[name].trace_3d
        pred_active = bool(
            len(pred_trace) >= 2
            and float(np.asarray(pred_trace).std(axis=0).sum()) >= 0.005
        )
        activity_matches.append(target_active == pred_active)
    return {
        "status": status,
        "format_ok": format_ok,
        "format_2d_required": require_2d,
        "format_2d_ok": format_2d_ok,
        "format_3d_ok": format_3d_ok,
        "format_failure_reasons": general_failures + failures_2d + failures_3d,
        "format_2d_failure_reasons": general_failures + failures_2d,
        "format_3d_failure_reasons": general_failures + failures_3d,
        "expected_points": expected,
        "predicted_arms": sorted({arm.name for arm in prediction.arms}),
        "target_arms": sorted({arm.name for arm in target.arms}),
        "active_target_arms": sorted(active_names),
        "active_arm_accuracy": _mean(activity_matches),
        "wire_format": prediction.wire_format,
        "ade_3d_m": ade,
        "mse_3d_m2": _mean(squared_errors_3d) if format_3d_ok else None,
        "coordinate_mse_3d_m2": (
            _mean(coordinate_squared_errors_3d) if format_3d_ok else None
        ),
        "fde_3d_m": _mean(final_3d) if format_3d_ok else None,
        "depth_mae_m": _mean(depth) if format_3d_ok else None,
        "lateral_mae_m": _mean(lateral) if format_3d_ok else None,
        **{
            f"sr_{centimeters}cm": (
                _mean([float(error < centimeters / 100.0) for error in errors_3d])
                if format_3d_ok else None
            )
            for centimeters in (1, 2, 3, 5, 7, 10, 15, 20)
        },
        "ade_success_5cm": bool(ade is not None and ade <= 0.05),
        "ade_success_10cm": bool(ade is not None and ade <= 0.10),
        "direction_score_3d": _mean(direction_3d) if format_3d_ok else None,
        "shape_score_3d": _mean(shape_3d) if format_3d_ok else None,
        "ade_2d_norm": _mean(errors_2d) if format_2d_ok else None,
        "fde_2d_norm": _mean(final_2d) if format_2d_ok else None,
        "sr_2d_5pct": (
            _mean([float(error < 0.05) for error in errors_2d])
            if format_2d_ok else None
        ),
        "direction_score_2d": _mean(direction_2d) if format_2d_ok else None,
        "shape_score_2d": _mean(shape_2d) if format_2d_ok else None,
        "requested_block_size": prediction.requested_block_size,
        "observed_block_size": prediction.observed_block_size,
        "dynamic_block_ok": bool(
            prediction.requested_block_size is not None
            and prediction.requested_block_size == prediction.observed_block_size
        ),
    }


def _fmean(rows, getter):
    values = [getter(row) for row in rows]
    values = [float(value) for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _score(row: dict) -> dict:
    return row.get("score") or {}


def _format_3d_ok(row: dict) -> bool:
    score = _score(row)
    return bool(score.get("format_3d_ok", score.get("format_ok", False)))


def _requires_2d(row: dict) -> bool:
    score = _score(row)
    return bool(score.get("format_2d_required", row.get("cot_mode") == "2d3d"))


def _format_2d_ok(row: dict) -> bool:
    if not _requires_2d(row):
        return False
    score = _score(row)
    return bool(score.get("format_2d_ok", score.get("format_ok", False)))


def _failure_category(reason: str) -> str:
    """Collapse row-specific details into stable report categories."""
    return str(reason).split(":", 1)[0] or "unknown_format_error"


def failure_analysis(group: list[dict]) -> dict:
    rows = len(group)
    failed = [row for row in group if not _score(row).get("format_ok", False)]
    required_2d = [row for row in group if _requires_2d(row)]
    failed_2d = [row for row in required_2d if not _format_2d_ok(row)]
    failed_3d = [row for row in group if not _format_3d_ok(row)]
    failed_both = [
        row for row in required_2d
        if not _format_2d_ok(row) and not _format_3d_ok(row)
    ]
    categories: Counter[str] = Counter()
    for row in failed:
        score = _score(row)
        reasons = score.get("format_failure_reasons") or [
            score.get("status") or "missing_score"
        ]
        categories.update({_failure_category(reason) for reason in reasons})
    return {
        "failure_rows": len(failed),
        "failure_rate": len(failed) / rows if rows else 0.0,
        "failure_pct": 100.0 * len(failed) / rows if rows else 0.0,
        "format_2d_required_rows": len(required_2d),
        "format_2d_error_rows": len(failed_2d),
        "format_2d_error_rate": (
            len(failed_2d) / len(required_2d) if required_2d else None
        ),
        "format_2d_error_pct": (
            100.0 * len(failed_2d) / len(required_2d) if required_2d else None
        ),
        "format_3d_error_rows": len(failed_3d),
        "format_3d_error_rate": len(failed_3d) / rows if rows else 0.0,
        "format_3d_error_pct": 100.0 * len(failed_3d) / rows if rows else 0.0,
        "format_2d_and_3d_error_rows": len(failed_both),
        "format_2d_and_3d_error_pct": (
            100.0 * len(failed_both) / len(required_2d) if required_2d else None
        ),
        "failure_case_counts": dict(sorted(categories.items())),
        "failure_case_percentages": {
            reason: 100.0 * count / rows if rows else 0.0
            for reason, count in sorted(categories.items())
        },
    }


def summarize(group: list[dict]) -> dict:
    """Aggregate scored prediction rows; metrics average the valid rows."""
    valid = [row for row in group if row.get("score", {}).get("format_ok")]
    valid_3d = [row for row in group if _format_3d_ok(row)]
    valid_2d = [row for row in group if _format_2d_ok(row)]
    timed = [row for row in group if not row.get("timing_warmup")]
    failures = failure_analysis(group)
    outside_codec = [row for row in group if row.get("gt_outside_codec")]
    metric_sources = Counter(
        str(row.get("metric_target_source") or "unknown") for row in group
    )
    return {
        "rows": len(group),
        "valid_rows": len(valid),
        "format_rate": len(valid) / len(group) if group else 0.0,
        **failures,
        "gt_outside_codec_rows": len(outside_codec),
        "gt_outside_codec_rate": (
            len(outside_codec) / len(group) if group else 0.0
        ),
        "gt_outside_codec_point_count": sum(
            int(row.get("gt_outside_codec_point_count") or 0) for row in group
        ),
        "metric_target_source_counts": dict(sorted(metric_sources.items())),
        "valid_3d_rows": len(valid_3d),
        "valid_2d_rows": len(valid_2d),
        "ade_3d_m": _fmean(valid_3d, lambda row: row["score"].get("ade_3d_m")),
        "mse_3d_m2": _fmean(valid_3d, lambda row: row["score"].get("mse_3d_m2")),
        "coordinate_mse_3d_m2": _fmean(
            valid_3d, lambda row: row["score"].get("coordinate_mse_3d_m2")
        ),
        "fde_3d_m": _fmean(valid_3d, lambda row: row["score"].get("fde_3d_m")),
        "depth_mae_m": _fmean(valid_3d, lambda row: row["score"].get("depth_mae_m")),
        "lateral_mae_m": _fmean(valid_3d, lambda row: row["score"].get("lateral_mae_m")),
        **{
            f"sr_{centimeters}cm": _fmean(
                valid_3d, lambda row, cm=centimeters: row["score"].get(f"sr_{cm}cm")
            )
            for centimeters in (1, 2, 3, 5, 7, 10, 15, 20)
        },
        "ade_success_5cm": _fmean(valid_3d, lambda row: row["score"].get("ade_success_5cm")),
        "ade_success_10cm": _fmean(valid_3d, lambda row: row["score"].get("ade_success_10cm")),
        "direction_score_3d": _fmean(valid_3d, lambda row: row["score"].get("direction_score_3d")),
        "shape_score_3d": _fmean(valid_3d, lambda row: row["score"].get("shape_score_3d")),
        "ade_2d_norm": _fmean(valid_2d, lambda row: row["score"].get("ade_2d_norm")),
        "fde_2d_norm": _fmean(valid_2d, lambda row: row["score"].get("fde_2d_norm")),
        "sr_2d_5pct": _fmean(valid_2d, lambda row: row["score"].get("sr_2d_5pct")),
        "direction_score_2d": _fmean(valid_2d, lambda row: row["score"].get("direction_score_2d")),
        "shape_score_2d": _fmean(valid_2d, lambda row: row["score"].get("shape_score_2d")),
        "active_arm_accuracy": _fmean(valid_3d, lambda row: row["score"].get("active_arm_accuracy")),
        "dynamic_block_rate": _fmean(group, lambda row: row.get("score", {}).get("dynamic_block_ok")),
        "seconds_per_row": _fmean(timed, lambda row: row.get("elapsed_s")),
        "mean_acceptance_length": _fmean(
            group, lambda row: row.get("vflash", {}).get("mean_acceptance_length")
        ),
        "atom_fallbacks_per_row": _fmean(group, lambda row: row.get("vflash", {}).get("atom_fallbacks")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", nargs="+", help="Prediction JSONL files of one benchmark")
    parser.add_argument("--output", help="Optional JSON file for the summary")
    args = parser.parse_args()
    rows = []
    for path in args.predictions:
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    summary = summarize(rows)
    summary["runtime_errors"] = sum(bool(row.get("error")) for row in rows)
    text = json.dumps(summary, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
