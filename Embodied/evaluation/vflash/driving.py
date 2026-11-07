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

"""Driving trajectory benchmark (Waymo, NAVSIM, Alpamayo).

The model answers with one ``<drive>`` atom of x/y/heading triples.
Positions are scored against the benchmark's continuous XY trajectory; the
heading error is reported per row.

    python -m evaluation.vflash.driving --model TARGET [--draft DRAFTER] \\
        --dataset waymo --input driving/waymo.jsonl --output driving_waymo.jsonl

An existing output file is resumed after its last complete row. Summarize the
rows with ``python -m evaluation.vflash.trajectory``.
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from eaglevl.model.vflash.physical_atoms import dequantize_physical_value, validate_answer
from evaluation.inference_compat import decode_generation_output, load_model_and_processor
from evaluation.vflash.generation import (
    build_model_inputs,
    count_output_rows,
    vflash_generate_kwargs,
    vflash_stats_record,
    generate_timed,
    read_records,
)
from evaluation.vflash.trajectory import TrajectoryAnswer, TrajectoryArm, score_trajectory

_DRIVE_RE = re.compile(r"<drive>((?:<\d+>)+)</drive>", re.DOTALL)
_NUMBER_RE = re.compile(r"<(\d+)>")
# Physical ranges of the x (forward), y (left) and heading tokens.
XYH_RANGES = ((-10.0, 250.0), (-65.0, 65.0), (-math.pi, math.pi))
# Alpamayo predicts a 64-point horizon; the other datasets at most 20 points.
LONG_HORIZON_DATASETS = {"alpamayo"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Target checkpoint")
    parser.add_argument("--draft", help="VFlash drafter; omit for autoregressive decoding")
    parser.add_argument("--dataset", required=True, help="Dataset name recorded in every row")
    parser.add_argument("--input", required=True, help="Prepared JSONL records")
    parser.add_argument("--output", required=True, help="Prediction JSONL")
    parser.add_argument("--start", type=int, default=0, help="First input line")
    parser.add_argument("--end", type=int, default=-1, help="End input line (exclusive); -1 reads to the end")
    parser.add_argument("--media-root", default="", help="Root of relative media paths in the records")
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Default: 384 for Alpamayo, 192 otherwise",
    )
    return parser.parse_args()


def parse_driving_xyh(text, expected_points=None):
    """Return the ``(points, 3)`` x/y/heading array of the single ``<drive>`` atom."""
    matches = _DRIVE_RE.findall(text)
    if len(matches) != 1:
        raise ValueError(f"expected one <drive> atom, found {len(matches)}")
    values = [int(value) for value in _NUMBER_RE.findall(matches[0])]
    if not values or len(values) % 3:
        raise ValueError("driving payload must contain complete x/y/heading triples")
    if any(value < 0 or value > 1000 for value in values):
        raise ValueError("driving token is outside [0, 1000]")
    points = len(values) // 3
    if expected_points is not None and points != expected_points:
        raise ValueError(f"expected {expected_points} driving points, found {points}")
    array = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    for axis, (minimum, maximum) in enumerate(XYH_RANGES):
        array[:, axis] = [
            dequantize_physical_value(int(value), minimum, maximum) for value in array[:, axis]
        ]
    return array


def heading_error_rad(predicted, target):
    """Mean absolute wrapped heading difference."""
    diff = np.asarray(predicted) - np.asarray(target)
    return float(np.abs(np.arctan2(np.sin(diff), np.cos(diff))).mean())


def main():
    args = parse_args()
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = 384 if args.dataset in LONG_HORIZON_DATASETS else 192
    rows = read_records(args.input, args.start, args.end)
    completed = count_output_rows(args.output)
    if completed > len(rows):
        raise SystemExit(f"output has {completed} rows but this shard contains only {len(rows)}")
    rows = rows[completed:]
    if not rows:
        return
    model, processor = load_model_and_processor(
        args.model,
        use_fast=False,
        local_code=True,
        local_processor=True,
        attn_implementation="flash_attention_2",
        training_runtime=True,
        draft_model_path=args.draft,
    )
    tokenizer = processor.tokenizer
    decoder = "vflash_unverified" if args.draft else "ar"
    if args.draft:
        generate_kwargs = vflash_generate_kwargs(tokenizer, max_new_tokens)
    else:
        generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}

    with open(args.output, "a", encoding="utf-8", buffering=1) as sink:
        for _, row in rows:
            points = row["num_trajectory_points"]
            result = dict(
                index=row["index"],
                name=row["benchmark_key"],
                dataset_name=args.dataset,
                model_variant=Path(args.model).name,
                cot_mode="direct_3d",
                decoder=decoder,
                matched_recipe=True,
                comparison_group="same_case_native_inputs",
                metric_target_source=row["metric_target_source"],
                timing_warmup=False,
                gt_outside_codec=False,
            )
            try:
                inputs = build_model_inputs(processor, row, args.media_root)
                prompt_tokens = inputs["input_ids"].shape[-1]
                output_ids, stats, elapsed = generate_timed(model, inputs, **generate_kwargs)
                text = decode_generation_output(
                    output_ids, inputs["input_ids"], processor, strip_prompt=True, strip_end_tokens=True
                )
                result.update(
                    prediction_text=text,
                    elapsed_s=elapsed,
                    prompt_tokens=prompt_tokens,
                    vflash=vflash_stats_record(stats, tokenizer),
                )
                validate_answer(text, "driving_xyh", expected_points=points)
                predicted = parse_driving_xyh(text, expected_points=points)
                target = TrajectoryAnswer(
                    arms=[TrajectoryArm(name="ego", trace_2d=[], trace_3d=row["metric_xy_m"])],
                    wire_format="physical",
                )
                prediction = TrajectoryAnswer(
                    arms=[TrajectoryArm(name="ego", trace_2d=[], trace_3d=predicted[:, :2].tolist())],
                    wire_format="physical",
                )
                result["score"] = score_trajectory(
                    prediction, target, expected_points=points, space="drive_bev", require_physical=True
                )
                result["heading_mae_rad"] = heading_error_rad(predicted[:, 2], row["metric_heading_rad"])
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
                result["score"] = {
                    "format_ok": False,
                    "format_3d_ok": False,
                    "format_2d_required": False,
                    "format_2d_ok": None,
                    "format_failure_reasons": ["runtime_or_parse_error"],
                    "format_3d_failure_reasons": ["runtime_or_parse_error"],
                    "format_2d_failure_reasons": [],
                }
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(args.dataset, row["index"], result["score"].get("ade_3d_m"), result.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
