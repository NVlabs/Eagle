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
# Train the VFlash target: an Eagle Embodied model that writes physical atoms
# (points, boxes, robot/driving/dense trajectories) with dedicated tokens.
#
# Usage: shell/vflash/train_ar.sh RECIPE_PATH OUTPUT_DIR BASE_MODEL [extra trainer arguments...]
#
# BASE_MODEL is an Eagle Embodied checkpoint (the VFlash target was trained from
# the 4B model). The defaults below are the reference recipe, trained on
# 64 GPUs (NNODES=8, GPUS=8 from shell/embodied/finetune.sh) for 33149 steps.
# Cluster layout and the other finetune.sh settings are environment variables;
# extra trainer arguments override the flags set here.
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: $0 RECIPE_PATH OUTPUT_DIR BASE_MODEL [extra trainer arguments...]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_PATH=$1
OUTPUT_DIR=$2
BASE_MODEL=$3
shift 3

export RUN_NAME=${RUN_NAME:-vflash-ar-4b}
export MAX_STEPS=${MAX_STEPS:-33149}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
export LEARNING_RATE=${LEARNING_RATE:-2e-5}
export WARMUP_STEPS=${WARMUP_STEPS:-500}
export MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-32768}
export MAX_NUM_TOKENS_PER_SAMPLE=${MAX_NUM_TOKENS_PER_SAMPLE:-32768}
export MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-32768}
export PACKING_BUFFER_SIZE=${PACKING_BUFFER_SIZE:-32}
export BALANCE_MEDIA_TOKENS=${BALANCE_MEDIA_TOKENS:-True}
export BALANCE_WINDOW_SIZE=${BALANCE_WINDOW_SIZE:-8}
export BALANCE_SEED=${BALANCE_SEED:-42}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
export SAMPLE_LOG_INTERVAL=${SAMPLE_LOG_INTERVAL:-1}

# --strict_data True stops on any data error (for example a video read
# timeout) instead of skipping the row; resume the job from its last checkpoint.
exec bash "${SCRIPT_DIR}/../embodied/finetune.sh" "${RECIPE_PATH}" "${OUTPUT_DIR}" "${BASE_MODEL}" \
  --training_mode ar \
  --oof_loss_scale "${OOF_LOSS_SCALE:-0.1}" \
  --strict_data "${STRICT_DATA:-True}" \
  --packing_log_interval "${PACKING_LOG_INTERVAL:-100}" \
  --save_every_n_hours "${SAVE_EVERY_N_HOURS:-0}" \
  "$@"
