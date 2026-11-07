#!/usr/bin/env bash
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
# Train the VFlash drafter against a frozen VFlash target.
#
# Usage: shell/vflash/train_drafter.sh RECIPE_PATH OUTPUT_DIR TARGET_MODEL [extra trainer arguments...]
#
# TARGET_MODEL is the exported target from shell/vflash/train_ar.sh; the
# drafter is bound to its exact weights and tokenizer. Checkpoints hold only the
# five-layer drafter. The defaults below are the reference recipe: packs of
# exactly 8 samples, 512 ordinary-text anchors per sample and 3 epochs of raw
# samples (90113 steps on 64 GPUs: NNODES=8, GPUs=8).
#
# The drafter trains with compiled FlexAttention, which requires PyTorch 2.8.x
# (validated with Triton 3.3; see the vflash extra in pyproject.toml), ZeRO-2 and
# sequence_parallel_degree=1.
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: $0 RECIPE_PATH OUTPUT_DIR TARGET_MODEL [extra trainer arguments...]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_PATH=$1
OUTPUT_DIR=$2
TARGET_MODEL=$3
shift 3

export RUN_NAME=${RUN_NAME:-vflash-drafter-4b}
export LEARNING_RATE=${LEARNING_RATE:-2e-4}
export WARMUP_STEPS=${WARMUP_STEPS:-0}
export MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-262144}
export MAX_NUM_TOKENS_PER_SAMPLE=${MAX_NUM_TOKENS_PER_SAMPLE:-32768}
export MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-262144}
export PACKING_BUFFER_SIZE=${PACKING_BUFFER_SIZE:-32}
export BALANCE_MEDIA_TOKENS=${BALANCE_MEDIA_TOKENS:-True}
export BALANCE_WINDOW_SIZE=${BALANCE_WINDOW_SIZE:-8}
export BALANCE_SEED=${BALANCE_SEED:-42}
export SAVE_STEPS=${SAVE_STEPS:-100}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
export SAMPLE_LOG_INTERVAL=${SAMPLE_LOG_INTERVAL:-1}
export SEQUENCE_PARALLEL_DEGREE=1

# max_steps follows from --vflash_train_sample_epochs and the world size.
# --strict_data True stops on any data error (for example a video read
# timeout) instead of skipping the row; resume the job from its last checkpoint.
exec bash "${SCRIPT_DIR}/../embodied/finetune.sh" "${RECIPE_PATH}" "${OUTPUT_DIR}" "${TARGET_MODEL}" \
  --training_mode vflash \
  --warmup_ratio "${WARMUP_RATIO:-0.04}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --oof_loss_scale "${OOF_LOSS_SCALE:-0.1}" \
  --vflash_anchors_per_sequence "${VFLASH_ANCHORS_PER_SEQUENCE:-512}" \
  --vflash_loss_gamma "${VFLASH_LOSS_GAMMA:-7.0}" \
  --vflash_max_samples_per_pack "${VFLASH_MAX_SAMPLES_PER_PACK:-8}" \
  --vflash_train_sample_epochs "${NUM_SAMPLE_EPOCHS:-3}" \
  --strict_data "${STRICT_DATA:-True}" \
  --packing_log_interval "${PACKING_LOG_INTERVAL:-100}" \
  --save_every_n_hours "${SAVE_EVERY_N_HOURS:-0}" \
  "$@"
