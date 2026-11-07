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

set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 META_PATH OUTPUT_DIR MODEL_PATH [extra trainer arguments...]" >&2
  exit 2
fi

CALLER_DIR=$PWD
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

resolve_from_caller() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${CALLER_DIR}" "$1" ;;
  esac
}

META_PATH=$(resolve_from_caller "$1")
OUTPUT_DIR=$(resolve_from_caller "$2")
MODEL_PATH=${MODEL_PATH:-}
shift 2
if (( $# > 0 )) && [[ "$1" != -* ]]; then
  MODEL_PATH=$1
  shift
fi
if [[ "${1:-}" == "--" ]]; then shift; fi
if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required: pass it as the third argument or set the MODEL_PATH environment variable." >&2
  exit 2
fi

if [[ "${MODEL_PATH}" != /* && -e "${CALLER_DIR}/${MODEL_PATH}" ]]; then
  MODEL_PATH=$(resolve_from_caller "${MODEL_PATH}")
fi
if [[ ! -f "${META_PATH}" ]]; then
  echo "Dataset recipe not found: ${META_PATH}" >&2
  exit 2
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
  : "${HF_TOKEN:?Set HF_TOKEN in the environment before loading a remote model}"
fi

PYTHON_BIN=${PYTHON_BIN:-python}
"${PYTHON_BIN}" -c 'import flash_attn' >/dev/null 2>&1 || {
  echo "flash-attn is required by Eagle Embodied training; install the training extra." >&2
  exit 1
}
GPUS=${GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
if [[ -z "${PORT:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    PORT=$((20000 + SLURM_JOB_ID % 20000))
  else
    PORT=29500
  fi
fi
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
MAX_STEPS=${MAX_STEPS:--1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WARMUP_STEPS=${WARMUP_STEPS:-500}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16384}
MAX_NUM_TOKENS_PER_SAMPLE=${MAX_NUM_TOKENS_PER_SAMPLE:-${MAX_SEQ_LENGTH}}
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-${MAX_SEQ_LENGTH}}
REPORT_TO=${REPORT_TO:-none}
BALANCE_MEDIA_TOKENS=${BALANCE_MEDIA_TOKENS:-False}
BALANCE_WINDOW_SIZE=${BALANCE_WINDOW_SIZE:-8}
BALANCE_SEED=${BALANCE_SEED:-42}
SEQUENCE_PARALLEL_DEGREE=${SEQUENCE_PARALLEL_DEGREE:-1}
RING_SEQUENCE_PARALLEL_DEGREE=${RING_SEQUENCE_PARALLEL_DEGREE:-1}
ROPE_SCALING_TYPE=${ROPE_SCALING_TYPE:-auto}
GRADIENT_CHECKPOINTING_USE_REENTRANT=${GRADIENT_CHECKPOINTING_USE_REENTRANT:-False}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-False}
FREEZE_LLM=${FREEZE_LLM:-False}
USE_LLM_LORA=${USE_LLM_LORA:-0}
USE_BACKBONE_LORA=${USE_BACKBONE_LORA:-0}
DEBUG_ASSERT_SP_BATCH_SYNC=${DEBUG_ASSERT_SP_BATCH_SYNC:-False}
if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
  DEEPSPEED_CONFIG=$(resolve_from_caller "${DEEPSPEED_CONFIG}")
else
  DEEPSPEED_CONFIG="${REPO_ROOT}/deepspeed_configs/zero_stage2_config.json"
fi
if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "DeepSpeed config not found: ${DEEPSPEED_CONFIG}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"
export LAUNCHER=pytorch
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${PORT}" \
  eaglevl/train/eagle_embodied_finetune_stream_packing.py \
  --model_name_or_path "${MODEL_PATH}" \
  --attn_implementation flash_attention_2 \
  --meta_path "${META_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --conv_style qwen2-chat \
  --auto_thinking_handler True \
  --overwrite_output_dir False \
  --freeze_llm "${FREEZE_LLM}" \
  --freeze_mlp False \
  --freeze_backbone "${FREEZE_BACKBONE}" \
  --use_llm_lora "${USE_LLM_LORA}" \
  --use_backbone_lora "${USE_BACKBONE_LORA}" \
  --vision_select_layer -1 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --bf16 True \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-100}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --save_every_n_hours "${SAVE_EVERY_N_HOURS:-0}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --lr_scheduler_type cosine \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --sample_log_interval "${SAMPLE_LOG_INTERVAL:-100}" \
  --video_total_pixels "${VIDEO_TOTAL_PIXELS:-12845056}" \
  --packing_buffer_size "${PACKING_BUFFER_SIZE:-12}" \
  --balance_media_tokens "${BALANCE_MEDIA_TOKENS}" \
  --balance_window_size "${BALANCE_WINDOW_SIZE}" \
  --balance_seed "${BALANCE_SEED}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --max_num_tokens_per_sample "${MAX_NUM_TOKENS_PER_SAMPLE}" \
  --max_num_tokens "${MAX_NUM_TOKENS}" \
  --sequence_parallel_degree "${SEQUENCE_PARALLEL_DEGREE}" \
  --ring_sequence_parallel_degree "${RING_SEQUENCE_PARALLEL_DEGREE}" \
  --debug_assert_sp_batch_sync "${DEBUG_ASSERT_SP_BATCH_SYNC}" \
  --rope_scaling_type "${ROPE_SCALING_TYPE}" \
  --max_frames "${MAX_FRAMES:-256}" \
  --do_train True \
  --grad_checkpoint True \
  --gradient_checkpointing_use_reentrant "${GRADIENT_CHECKPOINTING_USE_REENTRANT}" \
  --group_by_length False \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --report_to "${REPORT_TO}" \
  --run_name "${RUN_NAME:-eagle-embodied-finetune}" \
  "$@" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
