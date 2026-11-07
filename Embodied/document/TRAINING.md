# Training Guide — Continual SFT for LocateAnything

This guide covers full-parameter supervised fine-tuning (SFT) from a pretrained LocateAnything checkpoint.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Attention Implementation](#attention-implementation)
3. [Data Preparation](#data-preparation)
4. [Single-Node Training](#single-node-training)
5. [Multi-Node Training](#multi-node-training)
6. [Training Arguments Reference](#training-arguments-reference)
7. [DeepSpeed Configuration](#deepspeed-configuration)
8. [Streaming Packing](#streaming-packing)
9. [Checkpoint Resume](#checkpoint-resume)
10. [Tips & Troubleshooting](#tips--troubleshooting)

---

## Prerequisites

```bash
pip install -e .    # install eagle_vl + all deps
```

Hardware tested: 8× H100 80 GB (single-node), 2×8 H100 (multi-node).

---

## Attention Implementation

| Backend | `--attn_implementation` | GPU Architecture | Max Seq Length | Notes |
|---------|------------------------|------------------|----------------|-------|
| **Magi Attention** | `magi` | **Hopper** (H100/H800/H20) or **Blackwell** | **32K+** | Block-parallel attention for PBD. Recommended. |
| **SDPA** | `sdpa` | Any GPU | **~4K** | PyTorch native `F.scaled_dot_product_attention`. Short-context only. |

Magi Attention **only supports Hopper and Blackwell GPU architectures**. On non-Hopper/Blackwell hardware (A100, L40, etc.), use `sdpa` — but note that SDPA only supports fine-tuning with sequences up to ~4K tokens. Long-context training (16K–32K+) requires Hopper or Blackwell GPUs.

### Installing Magi Attention

Follow the [official MagiAttention installation guide](https://sandai-org.github.io/MagiAttention/docs/main/user_guide/install.html). We recommend using NGC-PyTorch containers with CUDA 13+ for optimal performance.

```bash
git clone https://github.com/SandAI-org/MagiAttention.git
cd MagiAttention
git checkout v1.0.5
git submodule update --init --recursive
pip install -r requirements.txt
pip install --no-build-isolation .        # Hopper
```

For **Blackwell** GPUs:

```bash
export MAGI_ATTENTION_PREBUILD_FFA=0
pip install --no-build-isolation .
export MAGI_ATTENTION_FA4_BACKEND=1      # always set when using Blackwell
```

> **Note:** The initial build takes ~10–20 minutes and is CPU-intensive. See the [MagiAttention docs](https://sandai-org.github.io/MagiAttention/docs/main/user_guide/install.html) for advanced options (IBGDA, FA4 kernel precompilation, etc.).

---

## Data Preparation

### 1. Annotation Format (JSONL)

Each line is a JSON object using ShareGPT-style conversations.

```jsonl
{"conversations": [{"from": "human", "value": "Detect all objects in <image-1>."}, {"from": "gpt", "value": "<ref>car</ref><box>(100,200,400,500)</box><ref>person</ref><box>(250,100,450,600)</box>"}], "image": "train/00001.jpg"}
{"conversations": [{"from": "human", "value": "Locate a single instance that matches the following description: red car."}, {"from": "gpt", "value": "<ref>red car</ref><box>(320,150,680,420)</box>"}], "image": "train/00002.jpg"}
```

Key conventions:
- Coordinates in `<box>` are normalized integers in `[0, 1000]`.
- Use `<image-1>`, `<image-2>`, ... as placeholders in conversation text.
- Images are resolved relative to the recipe's `root` directory.

### 2. Data Recipe JSON

Create a recipe file (e.g. `lam_recipe/my_recipe.json`) that maps dataset names to configurations:

```json
{
  "my_detection_data": {
    "annotation": "path/to/detection.jsonl",
    "root": "/data/images/",
    "repeat_time": 1.0,
    "data_augment": true
  },
  "my_gui_data": {
    "annotation": [
      "path/to/gui_part1.jsonl",
      "path/to/gui_part2.jsonl"
    ],
    "root": "/data/gui_screenshots/",
    "repeat_time": 2.0,
    "data_augment": false
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `annotation` | str / list | *required* | Path(s) to JSONL annotation file(s). Multiple files are merged. |
| `root` | str | `""` | Root directory for resolving relative media paths. |
| `repeat_time` | float | `1.0` | Sampling weight. `>=1`: repeat dataset N times; `<1`: downsample to N%. |
| `data_augment` | bool | `false` | Apply resize augmentation. |

---

## Single-Node Training

```bash
GPUS=8
OUTPUT_DIR=work_dirs/lam_sft

torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
    --master_port=29500 \
  eaglevl/train/lam_finetune_magi_stream.py \
  --model_name_or_path nvidia/LocateAnything-3B \
  --max_steps 25000 \
  --output_dir $OUTPUT_DIR \
  --meta_path "./lam_recipe/my_recipe.json" \
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
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 3 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --lr_scheduler_type "cosine" \
  --max_grad_norm 1.0 \
  --logging_steps 1 \
  --packing_buffer_size 32 \
  --max_seq_length 16384 \
  --max_num_tokens_per_sample 16384 \
  --max_num_tokens 25600 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --deepspeed "deepspeed_configs/zero_stage2_config.json" \
  --report_to "tensorboard" \
  --mlp_connector_layers 2 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
```

---

## Multi-Node Training

```bash
# Set these on each node
export NNODES=2
export NODE_RANK=0          # 0 on master, 1 on worker
export MASTER_ADDR=master_ip
export MASTER_PORT=29500
export GPUS=8

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$MASTER_PORT \
  eaglevl/train/lam_finetune_magi_stream.py \
  --model_name_or_path nvidia/LocateAnything-3B \
  ... # same arguments as single-node
```

---

## Training Arguments Reference

### Model Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name_or_path` | — | Pretrained checkpoint path or HuggingFace model ID |
| `--block_size` | 4 | MTP block length for parallel box decoding |
| `--causal_attn` | False | Causal attention in LLM (set `False` for MTP training) |
| `--attn_implementation` | `magi` | `magi` = Magi Attention (Hopper/Blackwell only, 32K+) or `sdpa` (any GPU, ~4K only) |
| `--freeze_llm` | False | Freeze language model weights |
| `--freeze_backbone` | False | Freeze vision encoder weights |
| `--freeze_mlp` | False | Freeze MLP projector weights |
| `--grad_checkpoint` | False | Gradient checkpointing to reduce memory |
| `--mlp_connector_layers` | 2 | Number of MLP projector layers |
| `--vision_select_layer` | -1 | ViT layer to extract features from (-1 = last) |

### Data Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--meta_path` | — | Path to data recipe JSON |
| `--chat_template_path` | `None` | Path to custom chat template file |
| `--max_seq_length` | 2048 | Max tokenized sequence length |
| `--max_num_tokens_per_sample` | 32768 | Samples exceeding this are skipped |
| `--max_num_tokens` | 36864 | Token budget per packed batch |
| `--packing_buffer_size` | 32 | Online packing buffer size |
| `--max_frames` | 64 | Max video frames to sample |
| `--target_fps` | 2 | Video sampling FPS |
| `--video_total_pixels` | ~10M | Pixel budget for video frames |
| `--sample_log_interval` | 100 | Log sample stats every N steps |

### Training Arguments (HuggingFace Trainer)

| Argument | Default | Description |
|----------|---------|-------------|
| `--learning_rate` | 2e-5 | Peak learning rate |
| `--lr_scheduler_type` | `cosine` | LR schedule |
| `--warmup_steps` | 500 | Linear warmup steps |
| `--max_steps` | 25000 | Total training steps |
| `--weight_decay` | 0.01 | AdamW weight decay |
| `--max_grad_norm` | 1.0 | Gradient clipping |
| `--bf16` | True | bfloat16 mixed precision |
| `--per_device_train_batch_size` | 1 | Batch size per GPU (use 1 with streaming packing) |
| `--gradient_accumulation_steps` | 1 | Gradient accumulation |
| `--deepspeed` | — | Path to DeepSpeed config |
| `--save_strategy` | `steps` | Checkpoint strategy |
| `--save_steps` | 100 | Checkpoint interval |
| `--save_total_limit` | 3 | Max checkpoints to keep |

---

## DeepSpeed Configuration

Two configs are provided in `deepspeed_configs/`:

| Config | File | Use Case |
|--------|------|----------|
| ZeRO Stage 1 | `zero_stage1_config.json` | Lower communication overhead, more memory per GPU |
| ZeRO Stage 2 | `zero_stage2_config.json` | Better memory efficiency (**recommended**) |

Both default to `"auto"` for lr, batch size, and gradient clipping, inheriting values from HuggingFace Trainer arguments.

---

## Streaming Packing

The training pipeline uses online streaming packing to efficiently batch variable-length sequences without padding waste:

- **Best-Fit:** Fills remaining batch space with the largest fitting sample from a buffer.
- **Big-Rocks-First:** Starts each new batch with the largest buffered sample.
- **Buffer size** is controlled by `--packing_buffer_size` (default 32). Larger buffers improve packing efficiency at the cost of memory.

Key parameters:
- `--per_device_train_batch_size 1` — always set to 1; the packing logic handles effective batch sizing.
- `--max_num_tokens` — total token budget per packed batch (e.g., 25600).
- `--max_num_tokens_per_sample` — samples longer than this are dropped.

For algorithm details, see [Streaming Packing Documentation](STREAMING_PACKING.md).

---

## Checkpoint Resume

Training automatically resumes when `output_dir` already contains a checkpoint. The streaming packing state is fully persisted:

- Iterator positions for all datasets
- RNG states
- Current batch composition
- Buffer contents

This guarantees **bit-wise identical** resume — the training will produce the exact same data order as if it had never stopped.

Checkpoint files per rank:
```
checkpoint-{step}/
├── model weights, optimizer, scheduler (standard HF/DeepSpeed)
└── dataloader_state_rank{rank}.pt    # streaming packing state
```

---

## Tips & Troubleshooting

**OOM during training:**
- Enable gradient checkpointing: `--grad_checkpoint True`
- Reduce `--max_num_tokens` (e.g., 16384)
- Reduce `--packing_buffer_size` (e.g., 16)
- Use ZeRO Stage 2

**Low packing efficiency (<70%):**
- Increase `--packing_buffer_size` (e.g., 64 or 128)
- Ensure `--max_num_tokens` is 2–3× `--max_num_tokens_per_sample`

**Training doesn't resume correctly:**
- Verify `dataloader_state_rank*.pt` exists in the checkpoint directory
- Ensure dataset metadata (recipe JSON) hasn't changed since the checkpoint was saved

**Non-H-series GPU:**
- Use `--attn_implementation sdpa` with `--max_seq_length 4096`
- SDPA does not support long-context (16K+) training
- Magi Attention (`magi`) only supports Hopper and Blackwell architectures
