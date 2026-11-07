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

"""PointMotionBench dense 3D motion forecasting (HOT3D, WorldTrack, DAVIS).

The model forecasts 8 query points over 30 future frames as ``<dense>``
atoms of anchor-relative tokens, ``(k - 500) * scale_m / 500`` metres from the
record's anchor. With ``--draft`` every record gets ``--best-of`` sequential
VFlash candidates whose drafter samples physical-atom slots at
``--draft-temperature`` after one per-record seed; the candidate with the
lowest ADE over the visible ground truth is kept (oracle best-of-N). A
candidate that fails the publication check is scored as generated.
Without ``--draft`` the target decodes one greedy answer.

    python -m evaluation.vflash.motion --model TARGET [--draft DRAFTER] \\
        --input motion/hot3d.jsonl --output motion_hot3d.jsonl

An existing output file is resumed by ``example_id``. Score the rows with
``python -m evaluation.vflash.motion_metrics``.
"""

import argparse
import hashlib
import json
import re
import time

import numpy as np
import torch

from eaglevl.model.vflash.physical_atoms import parse_atomic_tool_arguments
from evaluation.inference_compat import decode_generation_output, load_model_and_processor
from evaluation.vflash.generation import build_model_inputs, vflash_generate_kwargs

NUM_POINTS = 8
NUM_FUTURE = 30
ZERO_TOKEN = 500
_DENSE_ATOM_RE = re.compile(r"<(dense_2d|dense)>((?:[+-]?<\d+>)*)</(dense_2d|dense)>")
_TOKEN_RE = re.compile(r"<(\d+)>")
# Benchmark bookkeeping copied from the record when present.
_RECORD_KEYS = (
    "protocol",
    "frame_stride",
    "history_frames",
    "future_frames",
    "scored_mask",
    "n_candidates",
    "num_frames",
    "point_indices",
    "point_object_ids",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Target checkpoint")
    parser.add_argument("--draft", help="VFlash drafter; omit for greedy autoregressive decoding")
    parser.add_argument("--input", required=True, help="Prepared PointMotionBench records")
    parser.add_argument("--output", required=True, help="Prediction JSONL")
    parser.add_argument("--start", type=int, default=0, help="First input line")
    parser.add_argument("--end", type=int, default=-1, help="End input line (exclusive); -1 reads to the end")
    parser.add_argument("--media-root", default="", help="Root of relative media paths in the records")
    parser.add_argument("--max-new-tokens", type=int, default=960)
    parser.add_argument("--best-of", type=int, default=5, help="VFlash candidates per record")
    parser.add_argument("--draft-temperature", type=float, default=0.1)
    parser.add_argument("--sampling-seed", type=int, default=42, help="Base of the per-record seed")
    args = parser.parse_args()
    if not args.draft:
        args.best_of = 1
    if args.best_of < 1:
        parser.error("--best-of must be positive")
    return args


def sampling_seed(base, example_id):
    """Per-record seed: the first four bytes (little endian) of sha256("base:id")."""
    return int.from_bytes(hashlib.sha256(f"{base}:{example_id}".encode()).digest()[:4], "little")


def parse_dense_trajectories(text):
    """Token-index arrays ``(frames, 3)`` of the well-formed ``<dense>`` atoms."""
    trajectories = []
    for match in _DENSE_ATOM_RE.finditer(text):
        kind, payload, closing = match.groups()
        if kind != "dense" or closing != "dense" or "+" in payload or "-" in payload:
            continue
        values = np.asarray([int(value) for value in _TOKEN_RE.findall(payload)], dtype=np.int64)
        if values.size == 0 or values.size % 3:
            continue
        trajectories.append(values.reshape(-1, 3))
    return trajectories


def to_benchmark_frame(positions, record):
    """Rotate camera-axis predictions back into the benchmark world frame (WorldTrack)."""
    if "prediction_rotation_world_to_camera" not in record:
        return positions
    anchor = np.asarray(record["anchor"], dtype=np.float64)
    rotation = np.asarray(record["prediction_rotation_world_to_camera"], dtype=np.float64)
    return (positions - anchor) @ rotation + anchor


def parse_prediction(text, record):
    """Return benchmark-frame positions, imputed fraction, issues and trajectory count.

    Points and frames the answer does not cover stay at the common anchor.
    """
    anchor = np.asarray(record["anchor"], dtype=np.float64)
    trajectories = parse_dense_trajectories(text)
    positions = np.tile(anchor, (NUM_POINTS, NUM_FUTURE, 1))
    filled = np.zeros((NUM_POINTS, NUM_FUTURE), dtype=bool)
    step = float(record["scale_m"]) / float(ZERO_TOKEN)
    for point, cells in enumerate(trajectories[:NUM_POINTS]):
        frames = min(NUM_FUTURE, cells.shape[0])
        positions[point, :frames] = (np.asarray(cells[:frames], dtype=np.float64) - ZERO_TOKEN) * step + anchor
        filled[point, :frames] = True
    issues = []
    call = parse_atomic_tool_arguments(text, strict=False)
    if call is None:
        issues.append("no_tool_call")
    else:
        if call.get("block_size") != 3 * NUM_FUTURE + 2:
            issues.append(f"block_size={call.get('block_size')}")
        if call.get("count") != NUM_POINTS:
            issues.append(f"count={call.get('count')}")
    if len(trajectories) != NUM_POINTS:
        issues.append(f"n_trajectories={len(trajectories)}")
    if not filled.all():
        issues.append("missing_coordinates_anchor_filled")
    return to_benchmark_frame(positions, record), float(1 - filled.mean()), issues, len(trajectories)


def ade_visible(prediction, target, visible):
    error = np.linalg.norm(prediction - target, axis=-1)[visible]
    return float(error.mean()) if error.size else float("nan")


def build_inputs(processor, record, media_root):
    # The empty assistant turn makes the thinking handler append " /no_think"
    # to the prompt, as in the training records.
    sample = {
        "image": record["image"],
        "conversations": [
            {"from": "human", "value": record["prompt_dense_3d"]},
            {"from": "gpt", "value": ""},
        ],
    }
    return build_model_inputs(processor, sample, media_root)


def generate_candidates(model, processor, inputs, args):
    """Return the candidate texts, the elapsed seconds and per-candidate VFlash statistics."""
    texts, stats_list = [], []
    torch.cuda.synchronize()
    started = time.perf_counter()
    if not args.draft:
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        texts.append(decode_generation_output(
            output_ids, inputs["input_ids"], processor, strip_prompt=True, strip_end_tokens=True
        ))
    else:
        draft_temperature = args.draft_temperature if args.best_of > 1 else 0.0
        for _ in range(args.best_of):
            output_ids, _, stats = model.generate(
                **inputs,
                **vflash_generate_kwargs(processor.tokenizer, args.max_new_tokens, draft_temperature),
                vflash_publication_errors="return",
            )
            publication_error = getattr(stats, "publication_error", None)
            if publication_error is not None:
                # Score the rejected candidate exactly as generated (the ids the
                # publication check saw, decoded without clean-up).
                token_ids = output_ids[0].tolist()
                texts.append(processor.tokenizer.decode(token_ids, skip_special_tokens=False))
                stats_list.append({
                    "candidate_publication_error": publication_error,
                    "num_output_tokens": len(token_ids),
                })
                continue
            texts.append(decode_generation_output(
                output_ids, inputs["input_ids"], processor, strip_prompt=True, strip_end_tokens=True
            ))
            stats_list.append({
                "acceptance_lengths": [int(value) for value in stats.acceptance_lengths],
                "atom_fallbacks": int(stats.atom_fallbacks),
                "atomic_blocks_published": int(getattr(stats, "atomic_blocks_published", 0)),
                "num_output_tokens": int(stats.num_output_tokens),
            })
    torch.cuda.synchronize()
    return texts, time.perf_counter() - started, stats_list


def main():
    args = parse_args()
    decoder = "vflash_draft" if args.draft else "ar"
    records = []
    with open(args.input, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < args.start:
                continue
            if 0 <= args.end <= index:
                break
            if line.strip():
                records.append(json.loads(line))
    done_ids = set()
    try:
        with open(args.output, encoding="utf-8") as handle:
            done_ids = {json.loads(line)["example_id"] for line in handle if line.strip()}
    except FileNotFoundError:
        pass
    todo = [record for record in records if record["example_id"] not in done_ids]
    print(f"{len(records)} records, {len(done_ids)} already done, {len(todo)} to run", flush=True)
    if not todo:
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

    with open(args.output, "a", encoding="utf-8") as sink:
        for index, record in enumerate(todo):
            gt = np.asarray(record["gt_future_raw"], dtype=np.float64)
            visible = np.asarray(record["gt_future_vis"], dtype=bool)
            result = {key: record[key] for key in ("example_id", "bench", "video", "obj", "t0", "batch")}
            result.update({key: record[key] for key in _RECORD_KEYS if key in record})
            result.update(
                variant="dense_3d",
                decoder=decoder,
                best_of=args.best_of,
                ar_sample=False,
                ar_temperature=None,
                ar_coord_decoding="argmax",
                gt_future_raw=record["gt_future_raw"],
                gt_future_vis=record["gt_future_vis"],
            )
            try:
                seed = sampling_seed(args.sampling_seed, record["example_id"])
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                result["sampling_seed"] = seed
                inputs = build_inputs(processor, record, args.media_root)
                texts, elapsed, stats_list = generate_candidates(model, processor, inputs, args)
                candidates = [parse_prediction(text, record) for text in texts]
                ades = [ade_visible(positions, gt, visible) for positions, _, _, _ in candidates]
                best = int(np.nanargmin(ades)) if any(np.isfinite(ade) for ade in ades) else 0
                positions, imputed, issues, n_trajectories = candidates[best]
                result["candidate_texts"] = texts
                result["draft_temperature"] = args.draft_temperature if args.draft else None
                result.update(
                    pred_raw_combined=positions.round(6).tolist(),
                    pred_text=texts[best],
                    candidate_ades=ades,
                    candidate_text_hashes=[hashlib.sha1(text.encode()).hexdigest()[:10] for text in texts],
                    chosen_candidate=best,
                    imputed_fraction=imputed,
                    parsed_ok=not issues,
                    issues=issues,
                    n_trajectories=n_trajectories,
                    prompt_tokens=int(inputs["input_ids"].shape[-1]),
                    elapsed_s=elapsed,
                    vflash=stats_list,
                    error=None,
                )
            except Exception as error:  # keep the record in the denominator at the anchor
                anchor = np.asarray(record["anchor"], dtype=np.float64)
                result.update(
                    pred_raw_combined=np.tile(anchor, (NUM_POINTS, NUM_FUTURE, 1)).tolist(),
                    pred_text="",
                    candidate_ades=[],
                    chosen_candidate=None,
                    imputed_fraction=1.0,
                    parsed_ok=False,
                    issues=["runtime_error"],
                    n_trajectories=0,
                    prompt_tokens=None,
                    elapsed_s=None,
                    vflash=[],
                    error=f"{type(error).__name__}: {error}",
                )
            sink.write(json.dumps(result) + "\n")
            sink.flush()
            if index < 3 or (index + 1) % 25 == 0 or index + 1 == len(todo):
                print(
                    f"[dense_3d/{decoder}/b{args.best_of}] {index + 1}/{len(todo)} "
                    f"ade={result.get('candidate_ades')}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
