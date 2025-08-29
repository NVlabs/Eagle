#!/usr/bin/env bash
set -x

set -x
# export NCCL_IB_DISABLE=1
# export TORCHINDUCTOR_WORKER_START=fork
export NCCL_TIMEOUT=1800
# export NCCL_IB_QPS_PER_CONNECTION=1
export WANDB_PROJECT="eagle-video"
DATE=$(TZ=Asia/Shanghai date '+%Y_%m_%d_%H_%M_%S')

GPUS=${GPUS:-8}
NNODES=$1
OUTPUT_DIR=$2
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

TOTAL_GPUS=$((GPUS * NNODES))

BATCH_SIZE=1
TOKEN_PER_SEQUENCE_PARALLEL_GROUP=49152 # decided by packing size
TOKEN_PER_GPU=49152 # SIGLIP + 7B LLM, decided by Vision Encoder and LLM
TOTAL_TOKENS_EXP=${ITER_TOTAL_TOKENS_EXP:-24} # 18->0.25M, 19->0.5M, 20->1M, 21->2M, 22->4M, 23->8M, 24->16M
TOTAL_TOKENS_PER_ITER=$((2 ** TOTAL_TOKENS_EXP))
TOTAL_TOKENS_PER_ITER=$((TOKEN_PER_GPU * TOTAL_GPUS))

SEQUENCE_PARALLEL_SIZE=$((TOKEN_PER_SEQUENCE_PARALLEL_GROUP / TOKEN_PER_GPU))
SEQUENCE_PARALLEL_NUM_GROUP=$((TOTAL_GPUS / SEQUENCE_PARALLEL_SIZE))
DATA_PARALLEL_WORLD_SIZE=$((TOTAL_GPUS / SEQUENCE_PARALLEL_SIZE))

GRADIENT_ACC=$(($TOTAL_TOKENS_PER_ITER / ($TOKEN_PER_GPU * $TOTAL_GPUS)))
LOSS_VERSION="efficient_v2_cp_head"
SAMPLE_DIV=1

echo $NODE_RANK

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
NCCL_DEBUG=INFO


script_path=${BASH_SOURCE[0]}
script_name=$(basename "$script_path")
LAUNCHER=pytorch torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
  eaglevl/train/eagle_2_5_vl_finetune.py \
  --model_name_or_path "./work_dirs/eagle_er-qwen3_8B-Siglip2_400M_stage1_5_128gpu_v19" \
  --conv_style "qwen3-chat" \
  --normalize_type "siglip" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "local_playground/recipe/stage1.prepared.json" \
  --overwrite_output_dir False \
  --force_image_size 448 \
  --max_dynamic_tiles 12 \
  --down_sample_ratio 0.5 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 8 \
  --bf16 True \
  --use_online_packing True \
  --num_train_epochs 1 \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "steps" \
  --save_steps 250 \
  --save_total_limit 5 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length ${TOKEN_PER_SEQUENCE_PARALLEL_GROUP} \
  --sequence_parallel_degree ${SEQUENCE_PARALLEL_SIZE} \
  --sample_length_div ${SAMPLE_DIV} \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --deepspeed "deepspeed_configs/zero_stage3_config.json" \
  --loss_version ${LOSS_VERSION} \
  --report_to "wandb" \
  --run_name $script_name \
  --save_every_n_hours 4 \
  --use_onelogger True \
  --use_pixel_shuffle True \
  --mlp_connector_layers 2 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"