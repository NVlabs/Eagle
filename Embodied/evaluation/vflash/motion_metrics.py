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
#
# This file is adopted from the MolmoMotion project
# (https://github.com/allenai/molmo-motion), licensed under the Apache License, Version 2.0.
#
# Modifications © 2026 NVIDIA CORPORATION & AFFILIATES, licensed under
# the Apache License, Version 2.0.
#
# --------------------------------------------------------
# MolmoMotion
# Copyright 2026 Allen Institute for AI
# Licensed under the Apache License, Version 2.0
# --------------------------------------------------------

"""PointMotionBench metrics as defined by the paper.

The reported metrics pool every scored (point, frame) pair of all records:
ADE, FDE at the last horizon frame (the clip's last ground-truth frame for
short clips, over the points visible there) and PWT, the mean fraction of
errors strictly below 1, 2, 5, 10 and 20 cm. Per-clip and per-object means
and the FDE at each point's last visible frame are reported for reference.

    python -m evaluation.vflash.motion_metrics --input motion/hot3d.jsonl motion_hot3d.jsonl

Prediction rows are checked against the input records and scored in input
order, because the pooled float sums depend on the order.
"""

import argparse
import collections
import json

import numpy as np

THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20)
KEYS = ("ADE", "FDE_last_frame", "FDE_last_visible", "PWT")


def _new_group():
    return {"err": [], "fde_last": [], "fde_lastvis": []}


def _summarize(errs, fde_last, fde_lastvis):
    e = np.concatenate(errs) if errs else np.array([])
    out = {
        "ADE": float(e.mean()) if e.size else float("nan"),
        "FDE_last_frame": (
            float(np.concatenate(fde_last).mean()) if fde_last else float("nan")
        ),
        "FDE_last_visible": (
            float(np.concatenate(fde_lastvis).mean()) if fde_lastvis else float("nan")
        ),
        "PWT": (
            float(np.mean([(e < t).mean() for t in THRESHOLDS]))
            if e.size
            else float("nan")
        ),
    }
    out["PWT_per_threshold"] = (
        {str(t): float((e < t).mean()) for t in THRESHOLDS} if e.size else {}
    )
    return out


def _group_mean(groups):
    stats = [_summarize(g["err"], g["fde_last"], g["fde_lastvis"]) for g in groups]
    return {k: float(np.nanmean([s[k] for s in stats])) for k in KEYS}


def score_rows(rows) -> dict:
    """Score prediction rows; rows without a comparable scored cell are skipped."""
    pooled = _new_group()
    per_clip = collections.defaultdict(_new_group)
    per_object = collections.defaultdict(_new_group)
    n_rows = 0
    n_scored_points = 0
    protocols = collections.Counter()
    for row in rows:
        gt = np.asarray(row["gt_future_raw"], dtype=np.float64)
        pred = np.asarray(row["pred_raw_combined"], dtype=np.float64)
        vis = np.asarray(row["gt_future_vis"], dtype=bool)
        if gt.shape != pred.shape or vis.shape != gt.shape[:2]:
            continue
        mask = vis & np.isfinite(gt).all(-1) & np.isfinite(pred).all(-1)
        scored = np.asarray(row.get("scored_mask", [True] * gt.shape[0]), dtype=bool)
        mask &= scored[:, None]
        if not mask.any():
            continue
        err = np.linalg.norm(pred - gt, axis=-1)
        clip_key = (row.get("video"), row.get("obj"), row.get("t0"))
        oids = row.get("point_object_ids") or [None] * gt.shape[0]
        # FDE uses the last horizon frame that exists in the clip and only the
        # points visible there; points that vanish earlier do not count.
        last_col = gt.shape[1] - 1
        if "future_frames" in row and "num_frames" in row:
            valid_cols = [
                j for j, f in enumerate(row["future_frames"]) if f < row["num_frames"]
            ]
            if valid_cols:
                last_col = valid_cols[-1]
        for p in range(gt.shape[0]):
            if not mask[p].any():
                continue
            e = err[p, mask[p]]
            targets = (pooled, per_clip[clip_key], per_object[clip_key + (oids[p],)])
            j_last_vis = int(np.where(mask[p])[0].max())
            for g in targets:
                g["err"].append(e)
                g["fde_lastvis"].append(np.array([err[p, j_last_vis]]))
                if mask[p, last_col]:
                    g["fde_last"].append(np.array([err[p, last_col]]))
        n_rows += 1
        n_scored_points += int(scored.sum())
        protocols[row.get("protocol") or "unknown"] += 1
    if n_rows == 0:
        empty = {k: float("nan") for k in KEYS}
        return {
            "n_records": 0,
            "n_clips": 0,
            "n_objects": 0,
            "n_scored_points": 0,
            "pooled": empty,
            "per_clip": empty,
            "per_object": empty,
        }
    return {
        "n_records": n_rows,
        "n_clips": len(per_clip),
        "n_objects": len(per_object),
        "n_scored_points": n_scored_points,
        "protocols": dict(protocols),
        "pooled": _summarize(pooled["err"], pooled["fde_last"], pooled["fde_lastvis"]),
        "per_clip": _group_mean(per_clip.values()),
        "per_object": _group_mean(per_object.values()),
    }


def benchmark_report(rows, records, best_of=None, draft_temperature=None):
    """Check the predictions of one benchmark against its input records and score them.

    ``best_of`` and ``draft_temperature`` additionally require every row to
    carry that many candidates decoded at that drafter temperature.
    """
    ids = [row["example_id"] for row in rows]
    expected = [record["example_id"] for record in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Predictions must contain every input record exactly once")
    by_id = {row["example_id"]: row for row in rows}
    rows = [by_id[example_id] for example_id in expected]
    errors = [row["error"] for row in rows if row.get("error")]
    if errors:
        raise ValueError(f"{len(errors)} prediction rows failed, e.g. {errors[0]}")
    if best_of is not None and not all(
        row["best_of"] == best_of and len(row["candidate_texts"]) == best_of
        and len(row["candidate_ades"]) == best_of for row in rows
    ):
        raise ValueError(f"Every row must carry {best_of} candidates")
    if draft_temperature is not None and not all(
        row["draft_temperature"] == draft_temperature for row in rows
    ):
        raise ValueError(f"Every row must be decoded at draft temperature {draft_temperature}")
    paper = score_rows(rows)
    pooled = paper["pooled"]
    return {
        "rows": len(rows),
        "metrics": {"ADE": pooled["ADE"], "FDE": pooled["FDE_last_frame"], "PWT": pooled["PWT"]},
        "paper": paper,
        "parse_failures": sum(not row.get("parsed_ok") for row in rows),
        "mean_imputed_fraction": sum(row["imputed_fraction"] for row in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", nargs="+", help="Prediction JSONL files (shards) of one benchmark")
    parser.add_argument("--input", required=True, help="Input records that define the benchmark population")
    parser.add_argument("--best-of", type=int, help="Required candidates per row")
    parser.add_argument("--draft-temperature", type=float, help="Required drafter temperature")
    parser.add_argument("--output", help="Optional JSON file for the report")
    args = parser.parse_args()
    rows = []
    for path in args.predictions:
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    with open(args.input, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    report = benchmark_report(rows, records, args.best_of, args.draft_temperature)
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
