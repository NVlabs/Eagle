#!/bin/bash

NAME="qwen_test"
MASTER_ADDR="localhost"
SLURM_PROCID=0 
PATH_TO_PRETRAINING_DATA="/mnt/models/total_data/pretrain_data/test"

export WANDB_PROJECT="eagle"
export WANDB_RUN_ID="${NAME}"
export WANDB_RESUME="allow"

echo "MASTER_ADDR=$MASTER_ADDR"
echo "number of nodes: 8"
echo "node rank: $SLURM_PROCID"

python -m torch.distributed.run \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank "$SLURM_PROCID" \
    --master_addr "$MASTER_ADDR" \
    --master_port 25031 \
    train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/models/Qwen2.5-3B-Instruct \
    --version plain \
    --data_path "${PATH_TO_PRETRAINING_DATA}/test.json" \
    --image_folder "${PATH_TO_PRETRAINING_DATA}/images" \
    --vision_tower "convnext-1024;timm-vit;qwen2_5-vl;" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "./checkpoints/$NAME" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --attn_implementation sdpa \
    --run_name "${NAME}"