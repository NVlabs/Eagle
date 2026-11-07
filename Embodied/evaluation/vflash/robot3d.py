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

"""Robot 3D trajectory benchmark (AgiBot, InternA1, MolmoBot).

Each prepared record holds the training-format conversation and the
continuous per-arm ground truth in metres. With ``--draft`` the target decodes
through the unverified drafter, otherwise autoregressively.

    python -m evaluation.vflash.robot3d --model TARGET [--draft DRAFTER] \\
        --dataset agibot --input robot3d/agibot.jsonl --output robot3d_agibot.jsonl

An existing output file is resumed after its last complete row. Summarize the
rows with ``python -m evaluation.vflash.trajectory``.
"""

import argparse
import copy
import json

from evaluation.inference_compat import decode_generation_output, load_model_and_processor
from evaluation.vflash.generation import (
    build_model_inputs,
    count_output_rows,
    vflash_generate_kwargs,
    vflash_stats_record,
    generate_timed,
    read_records,
)
from evaluation.vflash.trajectory import parse_trajectory_text, score_trajectory

RUNTIME_ERROR_SCORE = {
    "status": "runtime_error",
    "format_ok": False,
    "format_2d_required": False,
    "format_2d_ok": None,
    "format_3d_ok": False,
    "format_failure_reasons": ["runtime_error"],
    "format_2d_failure_reasons": [],
    "format_3d_failure_reasons": ["runtime_error"],
}


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
    parser.add_argument("--max-new-tokens", type=int, default=320)
    return parser.parse_args()


def assistant_text(record):
    return next(turn["value"] for turn in record["conversations"] if turn["from"] == "gpt")


def record_identity(index, record):
    return {
        "index": index,
        "name": record.get("name") or record.get("sample_key"),
        "dataset": record.get("dataset"),
        "task_name": record.get("task_name"),
        "episode_index": record.get("episode_index"),
        "frame_index": record.get("frame_index", record.get("frame_idx")),
        "scene_token": record.get("scene_token"),
        "frame_token": record.get("frame_token"),
    }


def continuous_target(record, target):
    """Replace the quantized answer trajectory with the continuous ground truth."""
    by_arm = record.get("metric_trajectory_3d_by_arm_m")
    if not isinstance(by_arm, dict):
        return target, "quantized_text_target_m"
    target = copy.deepcopy(target)
    for arm in target.arms:
        points = by_arm.get(arm.name)
        if not isinstance(points, list):
            raise ValueError(f"continuous robot GT is missing arm {arm.name!r}")
        arm.trace_3d = [[float(value) for value in point] for point in points]
    return target, "continuous_trajectory_3d_by_arm_m"


def main():
    args = parse_args()
    rows = read_records(args.input, args.start, args.end)
    completed = count_output_rows(args.output)
    if completed > len(rows):
        raise SystemExit(f"output has {completed} rows but this shard contains only {len(rows)}")
    rows = rows[completed:]
    if not rows:
        print(f"already complete: {args.output}", flush=True)
        return
    if completed:
        print(f"resuming {args.output} after {completed} completed rows", flush=True)
    total_rows = completed + len(rows)
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
        generate_kwargs = vflash_generate_kwargs(tokenizer, args.max_new_tokens)
    else:
        generate_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}

    with open(args.output, "a", encoding="utf-8") as sink:
        for offset, (index, record) in enumerate(rows):
            result = {
                **record_identity(index, record),
                "dataset_name": args.dataset,
                "model_variant": "current",
                "cot_mode": "direct_3d",
                "matched_recipe": False,
                "comparison_group": "",
                "decoder": decoder,
                "timing_warmup": offset < 3,
            }
            try:
                inputs = build_model_inputs(processor, record, args.media_root, decode_frames=True)
                output_ids, stats, elapsed = generate_timed(model, inputs, **generate_kwargs)
                prediction_text = decode_generation_output(
                    output_ids, inputs["input_ids"], processor, strip_prompt=True, strip_end_tokens=True
                )
                target_text = assistant_text(record)
                target, target_source = continuous_target(record, parse_trajectory_text(target_text))
                expected = record.get("num_trajectory_points")
                score = score_trajectory(
                    parse_trajectory_text(prediction_text),
                    target,
                    expected_points=int(expected) if expected is not None else None,
                    space="robot_xyz",
                )
                result.update(
                    prediction_text=prediction_text,
                    target_text=target_text,
                    elapsed_s=elapsed,
                    score=score,
                    vflash=vflash_stats_record(stats, tokenizer),
                    metric_target_source=target_source,
                    gt_outside_codec=bool(record.get("gt_outside_codec", False)),
                    gt_outside_codec_point_count=len(record.get("gt_outside_codec_points") or []),
                    error=None,
                )
            except Exception as error:  # keep failures in the denominator
                result.update(
                    prediction_text="",
                    target_text=assistant_text(record),
                    elapsed_s=None,
                    score=dict(RUNTIME_ERROR_SCORE),
                    vflash={},
                    error=f"{type(error).__name__}: {error}",
                )
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            done = completed + offset + 1
            if done <= 3 or done % 25 == 0 or done == total_rows:
                print(f"[{args.dataset}/{decoder}] {done}/{total_rows}", flush=True)


if __name__ == "__main__":
    main()
