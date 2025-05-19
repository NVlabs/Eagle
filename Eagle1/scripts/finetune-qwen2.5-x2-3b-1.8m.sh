#!/bin/bash

NAME=$1

export WANDB_PROJECT="eagle"
export WANDB_RUN_ID="${NAME}"
export WANDB_RESUME="allow"
echo "MASTER_ADDR=$MASTER_ADDR"
n_node=$SLURM_JOB_NUM_NODES
echo "number of nodes: $n_node"
echo "node rank: $SLURM_PROCID"

python -m torch.distributed.run \
    --nproc_per_node 8 \
    --nnodes "$SLURM_NNODES" \
    --node_rank "$SLURM_PROCID" \
    --master_addr "$MASTER_ADDR" \
    --master_port 25031 \
    train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version qwen \
    --data_path "$PATH_TO_SFT_DATA/eagle-sft-v1-1_8m.json" \
    --image_folder "$PATH_TO_SFT_DATA/images" \
    --vision_tower "timm-vit;qwen2.5-v1-2b-instruct" \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter "$PATH_TO_PRETRAINED_PROJECTOR/mm_projector.bin" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "./checkpoints/$NAME" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${NAME}"