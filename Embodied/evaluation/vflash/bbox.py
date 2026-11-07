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

"""COCO and Dense200 box detection with the visual-primitive prompts.

Runs the shared detection harness (``evaluation/inference_detection_ddp.py``)
with the VFlash protocol and scores the predictions with the shared metric
programs:

- COCO: one conversation per image that asks for the positive categories in
  turn (image in the first turn only), 512 new tokens per turn; reports
  Avg F1, F1@0.50 and F1@0.95 (``metrics/coco_lvis_metric.py``, which needs
  the FastEvaluate extension built from ``evaluation/fastevaluate``).
- Dense200: one single-turn query per category, 4096 new tokens; reports the
  mean over IoU 0.50:0.95 of F1(mean precision, mean recall).

With ``--draft`` the target decodes through the drafter with the bbox
hybrid decoder, otherwise greedily and autoregressively.

    python -m evaluation.vflash.bbox --model TARGET [--draft DRAFTER] --dataset COCO \\
        --data EvalData --output results/bbox_COCO/predictions.jsonl
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parents[1]
PROTOCOLS = {
    "COCO": {"max_new_tokens": 512, "prompt": "--multiturn_training_prompt"},
    "Dense200": {"max_new_tokens": 4096, "prompt": "--per_category_training_prompt"},
}
_COCO_METRIC_RE = re.compile(r"^(Avg F1|F1@0\.50|F1@0\.95)\s*\|\s*([\d.]+)", re.M)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Target checkpoint")
    parser.add_argument("--draft", help="VFlash drafter; omit for autoregressive decoding")
    parser.add_argument("--dataset", required=True, choices=sorted(PROTOCOLS))
    parser.add_argument("--data", required=True, help="Rex-Omni evaluation data root (images and annotations)")
    parser.add_argument("--input", help="Benchmark JSONL; default <data>/_annotations/box_eval/<dataset>.jsonl")
    parser.add_argument("--output", required=True, help="Merged prediction JSONL")
    parser.add_argument("--coco-annotations", help="COCO GT JSON; default <data>/coco/instances_val2017.json")
    parser.add_argument("--nproc-per-node", type=int, default=8, help="GPUs used for inference (torchrun if > 1)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-inference", action="store_true", help="Only score an existing --output")
    return parser.parse_args()


def dense_mean_f1(eval_results):
    """Mean over IoU thresholds of F1 computed from the mean precision and recall."""
    f1 = {}
    for threshold, block in eval_results.items():
        metrics = block["basic_metrics"]["common_object_detection_Dense200"]
        precision = statistics.mean(metrics["precisions"])
        recall = statistics.mean(metrics["recalls"])
        f1[threshold] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"mF1": statistics.mean(f1.values()), "F1_by_iou": f1}


def run_inference(args, output):
    protocol = PROTOCOLS[args.dataset]
    data = Path(args.data)
    command = [sys.executable]
    if args.nproc_per_node > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.nproc_per_node}"]
    command += [
        str(EVALUATION_DIR / "inference_detection_ddp.py"),
        "--model_path", args.model,
        "--test_jsonl_path", args.input or str(data / "_annotations" / "box_eval" / f"{args.dataset}.jsonl"),
        "--image_root_dir", str(data),
        "--save_path", str(output),
        "--max_new_tokens", str(protocol["max_new_tokens"]),
        "--num_workers", str(args.num_workers),
        protocol["prompt"], "--vflash_box_prompt",
    ]
    if args.draft:
        command += ["--generation_mode", "vflash", "--draft_model_path", args.draft]
    else:
        command += ["--generation_mode", "slow"]
    subprocess.run(command, check=True)


def score(args, output):
    log = output.parent / "metrics.log"
    if args.dataset == "COCO":
        annotations = args.coco_annotations or str(Path(args.data) / "coco" / "instances_val2017.json")
        tsv = output.parent / "fast_eval.tsv"
        subprocess.run([
            sys.executable, str(EVALUATION_DIR / "utils" / "convert_coco_lvis_to_standard_format.py"),
            "--our_pred_jsonl", str(output), "--coco_json", annotations, "--out_tsv", str(tsv),
            "--positive_only",
        ], check=True)
        command = [
            sys.executable, str(EVALUATION_DIR / "metrics" / "coco_lvis_metric.py"),
            "--gt", annotations, "--pred_tsv", str(tsv),
        ]
    else:
        command = [
            sys.executable, str(EVALUATION_DIR / "metrics" / "other_metric.py"),
            "--data_path", str(output), "--output_path", str(output.parent / "eval_results.json"),
        ]
    with log.open("w") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    if args.dataset == "COCO":
        metrics = {name: float(value) for name, value in _COCO_METRIC_RE.findall(log.read_text())}
        if len(metrics) != 3:
            raise ValueError(f"Missing COCO F1 metrics in {log}")
        return metrics
    return dense_mean_f1(json.loads((output.parent / "eval_results.json").read_text()))


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_inference:
        run_inference(args, output)
    metrics = score(args, output)
    (output.parent / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
