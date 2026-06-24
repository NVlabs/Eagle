#!/usr/bin/env bash
set -euo pipefail

unset CONDA_SHLVL
unset CONDA_EXE
unset _CE_CONDA
unset CONDA_PREFIX
unset CONDA_PROMPT_MODIFIER
unset CONDA_PYTHON_EXE
unset CONDA_DEFAULT_ENV
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'anaconda3' | paste -sd ':' -)
pip install hf_xet

export WANDB_PROJECT="${WANDB_PROJECT:-star-nemo}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-locany-lora-visual-prompt}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Please set HF_TOKEN before launching training." >&2
  exit 1
fi
export HF_TOKEN

GPUS=${GPUS:-8}
NNODES=${1:-1}
OUTPUT_DIR=${2:-"work_dirs/locany_lora_visual_prompt_single_turn"}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

MODEL_PATH=${MODEL_PATH:-"nvidia/LocateAnything-3B"}
if [[ -z "${META_PATH:-}" ]]; then
  echo "Please set META_PATH to a training meta json. Set visual_prompt=true for datasets that should use visual prompts." >&2
  exit 1
fi
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"deepspeed_configs/zero_stage1_config.json"}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=${GRADIENT_ACC:-1}
MAX_STEPS=${MAX_STEPS:-5000}
SAVE_STEPS=${SAVE_STEPS:-100}
LR=${LR:-2e-5}
WARMUP_STEPS=${WARMUP_STEPS:-500}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16384}
MAX_NUM_TOKENS_PER_SAMPLE=${MAX_NUM_TOKENS_PER_SAMPLE:-16384}
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-16384}
PACKING_BUFFER_SIZE=${PACKING_BUFFER_SIZE:-32}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}

USE_LLM_LORA=${USE_LLM_LORA:-64}
USE_BACKBONE_LORA=${USE_BACKBONE_LORA:-0}
FREEZE_LLM=${FREEZE_LLM:-True}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-True}
FREEZE_MLP=${FREEZE_MLP:-False}

mkdir -p "$OUTPUT_DIR"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

script_name=$(basename "${BASH_SOURCE[0]}")

LAUNCHER=pytorch python -m torch.distributed.run \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --nproc_per_node="$GPUS" \
  --master_port="$PORT" \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path "$MODEL_PATH" \
  --max_steps "$MAX_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  --meta_path "$META_PATH" \
  --overwrite_output_dir False \
  --block_size 6 \
  --attn_implementation magi \
  --causal_attn False \
  --freeze_llm "$FREEZE_LLM" \
  --freeze_mlp "$FREEZE_MLP" \
  --freeze_backbone "$FREEZE_BACKBONE" \
  --use_llm_lora "$USE_LLM_LORA" \
  --use_backbone_lora "$USE_BACKBONE_LORA" \
  --vision_select_layer -1 \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACC" \
  --save_strategy "steps" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 3 \
  --learning_rate "$LR" \
  --weight_decay 0.01 \
  --warmup_steps "$WARMUP_STEPS" \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --video_total_pixels 8192 \
  --sample_log_interval 1 \
  --packing_buffer_size "$PACKING_BUFFER_SIZE" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --max_num_tokens_per_sample "$MAX_NUM_TOKENS_PER_SAMPLE" \
  --max_num_tokens "$MAX_NUM_TOKENS" \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --report_to "tensorboard" \
  --run_name "$script_name" \
  --use_onelogger True \
  --mlp_connector_layers 2 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
