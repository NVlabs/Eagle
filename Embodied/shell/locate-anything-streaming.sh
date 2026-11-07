#!/usr/bin/env bash
set -x

unset CONDA_SHLVL
unset CONDA_EXE
unset _CE_CONDA
unset CONDA_PREFIX
unset CONDA_PROMPT_MODIFIER
unset CONDA_PYTHON_EXE
unset CONDA_DEFAULT_ENV
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'anaconda3' | paste -sd ':' -)
echo "Conda has been disabled. Running training script..."
pip install hf_xet
export WANDB_PROJECT="star-nemo"
export WANDB_RUN_ID="finetune"
export WANDB_RESUME="allow"
export HF_TOKEN="${HF_TOKEN:?Please set HF_TOKEN environment variable}"

GPUS=${GPUS:-8}
NNODES=${1:-1}
OUTPUT_DIR=${2:-"work_dirs/lam_debug"}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=1
echo $NODE_RANK

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
export NCCL_DEBUG=INFO

script_path=${BASH_SOURCE[0]}
script_name=$(basename "$script_path")
MODEL_PATH=${MODEL_PATH:-"nvidia/LocateAnything-3B"}

LAUNCHER=pytorch torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
  eaglevl/train/lam_finetune_magi_stream.py \
  --model_name_or_path ${MODEL_PATH} \
  --max_steps 8000 \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./recipe/ablation.json" \
  --overwrite_output_dir False \
  --block_size 6 \
  --attn_implementation magi \
  --causal_attn False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 3 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --video_total_pixels 8192 \
  --sample_log_interval 1 \
  --packing_buffer_size 32 \
  --max_seq_length 6400 \
  --max_num_tokens_per_sample 6400 \
  --max_num_tokens 6400 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --deepspeed "deepspeed_configs/zero_stage1_config.json" \
  --report_to "tensorboard"\
  --run_name $script_name \
  --use_onelogger True \
  --mlp_connector_layers 2 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"