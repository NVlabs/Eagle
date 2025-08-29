#!/usr/bin/env bash
set -x
GPUS=${GPUS:-8}

META_PATH=${1:-"local_playground/recipe/stage1.json"}
NNODES=${2:-1}
LOG_DIR=${3:-"work_dirs/prepare_data"}
TOKENIZER=${4:-"Qwen/Qwen3-1.7B"}
# TOKENIZER=${4:-"HuggingFaceTB/SmolLM2-1.7B-Instruct"}
LAUNCHER=${5:-"pytorch"}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

META_NAME=$(basename $META_PATH .json)

BATCH_SIZE=${BATCH_SIZE:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / (GPUS* NNODES)))

export NCCL_TIMEOUT=6000
export LAUNCHER=${LAUNCHER:-"pytorch"}
MAX_LEN=49152
SAMPLE_DIV=1
SAMPLE_FRAME_STRATEGY="auto_v2"
default_fps=4
min_num_frames=8
max_num_frames=24

# set conv_style
if [[ "$TOKENIZER" == *"Qwen3"* ]]; then
    conv_style="qwen3-chat"
else
    echo "Error: Unsupported TOKENIZER '$TOKENIZER'. Please modify the script to support this model."
    exit 1
fi

META_NAME=eagle_2_5_${conv_style}_maxlen${MAX_LEN}_sample_div${SAMPLE_DIV}_${META_NAME}

if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi
export RESIZE_IMAGE_SIZE_TO_HALF_SIZE=0

# Common parameters for both launchers
COMMON_PARAMS="
  --model_name_or_path ${TOKENIZER} \
  --conv_style ${conv_style} \
  --output_dir ${LOG_DIR} \
  --save_root ./local_playground/prepared_data/${META_NAME} \
  --meta_path ${META_PATH} \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_tiles 12 \
  --down_sample_ratio 0.5 \
  --pad2square False \
  --use_data_resampling False \
  --dataloader_num_workers 8 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --batch_size ${BATCH_SIZE} \
  --sample_frame_strategy ${SAMPLE_FRAME_STRATEGY} \
  --max_seq_length ${MAX_LEN} \
  --sample_length_div ${SAMPLE_DIV} \
  --group_by_length True \
  --dynamic_image_size True \
  --default_fps ${default_fps} \
  --min_num_frames ${min_num_frames} \
  --max_num_frames ${max_num_frames} \
  --deepspeed deepspeed_configs/zero_stage2_config.json \
  --use_thumbnail True \
  "

if [ "$LAUNCHER" == "pytorch" ]; then
  LAUNCHER=pytorch torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    eaglevl/train/data_prepare.py \
    ${COMMON_PARAMS} \
    2>&1 | tee -a "${LOG_DIR}/training_log.txt"
elif [ "$LAUNCHER" == "slurm" ]; then
  LAUNCHER=slurm python -u eaglevl/train/data_prepare.py \
    ${COMMON_PARAMS} \
    2>&1 | tee -a "${LOG_DIR}/training_log.txt"
fi
