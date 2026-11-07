# VFlash Training

[VFlash](README.md) · [Training](TRAINING.md) · [Data Preparation](DATA_PREPARATION.md) · [Evaluation](EVALUATION.md) · [Detailed Results](RESULTS.md)

VFlash training has two stages: fine-tune an autoregressive target on physical-primitive answers, then freeze it and train a matched drafter. This guide covers the model, dependencies, and reference training settings. Prepare the input recipe and records with [Data Preparation](DATA_PREPARATION.md).

## Table of Contents

- [Installation](#installation)
- [Model and objective](#model-and-objective)
- [Training commands and settings](#training)
- [Packing and training budget](#packing-and-training-budget)
- [Checkpoints and resume](#checkpoints)
- [Implementation references](#implementation-references)

## Installation

Run commands from `Embodied/`. Install a CUDA build of PyTorch 2.8 before installing the training dependencies. NGC PyTorch containers with 2.8.0a0 builds are also supported.

```bash
git clone https://github.com/NVlabs/Eagle.git eagle
cd eagle/Embodied
pip install -e '.[vflash]'
```

The `vflash` extra pins `torch>=2.8.0a0,<2.9`, `torchvision>=0.23.0a0,<0.24` and `triton>=3.3,<3.4` and adds `flash-attn`; the drafter trainer needs PyTorch 2.8 with compiled FlexAttention and stops otherwise.

For inference and benchmark dependencies, see [Evaluation](EVALUATION.md#dependencies).

## Model and Objective

The VFlash target uses the Eagle Embodied VLM implementation with MoonViT-SO-400M, a two-layer MLP projector, and Qwen3-4B (about 4B parameters). It is fine-tuned to produce physical primitives with dedicated coordinate tokens, then frozen while the drafter trains.

The drafter has five full-width Qwen3 decoder layers. It reads the hidden states of target layers 1, 9, 17, 25, and 33 and reuses the target's embedding table and LM head. Every annotated primitive forms one training block with uniform loss weights. Up to 512 sampled 16-token text blocks per sample train the text path, with weights that decay with distance from the anchor.

See [Physical Primitives](DATA_PREPARATION.md#physical-primitives) for the coordinate layouts and [Decoding Rules](EVALUATION.md#decoding-rules) for how these two paths are used at inference.

## Training

Use [`train_ar.sh`](../../shell/vflash/train_ar.sh) to fine-tune the target, then [`train_drafter.sh`](../../shell/vflash/train_drafter.sh) to train its drafter. Each command takes the recipe, output directory, and initial model or target checkpoint.

Run the same command on each allocated node, setting `GPUS`, `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `PORT` for that node. Additional trainer arguments after the three positional arguments override the script defaults. The launcher requires `HF_TOKEN` when the model argument is a remote Hugging Face ID.

```bash
# Stage 1: AR target. Fine-tunes an Eagle Embodied checkpoint on physical-primitive answers.
GPUS=8 NNODES=8 NODE_RANK=<node-rank> MASTER_ADDR=<master-address> \
  bash shell/vflash/train_ar.sh <recipe.json> <target-output-dir> <base-eagle-embodied-checkpoint>

# Stage 2: drafter. Freezes the target from stage 1 and trains the five-layer drafter.
GPUS=8 NNODES=8 NODE_RANK=<node-rank> MASTER_ADDR=<master-address> \
  bash shell/vflash/train_drafter.sh <recipe.json> <drafter-output-dir> <target-checkpoint>
```

| Setting | Target (`--training_mode ar`) | Drafter (`--training_mode vflash`) |
|:--|:--|:--|
| Start point | 4B Eagle Embodied checkpoint (MoonViT + Qwen3-4B) | frozen stage-1 target and a new five-layer drafter |
| GPUs | 64 (8 nodes × 8) | 64 (8 nodes × 8) |
| Optimization | lr 2e-5, 500 warmup steps, cosine | lr 2e-4, warmup ratio 0.04, cosine |
| Length | Set `MAX_STEPS` for your dataset and training budget | 3 sample epochs (`--vflash_train_sample_epochs 3`) |
| Packing | 32,768-token packs | packs of exactly 8 samples (`--vflash_max_samples_per_pack`), each up to 32,768 tokens |
| Loss | next-token cross-entropy; `<oof>` (out-of-frame) targets weighted 0.1 (`--oof_loss_scale`) | every primitive block with uniform weights, plus up to 512 text blocks per sample (`--vflash_anchors_per_sequence`) with position decay γ = 7 (`--vflash_loss_gamma`) |
| Data checks | the physical primitives of every answer are always validated; `--strict_data True` | same |
| Requirements | flash-attn | flash-attn, PyTorch 2.8 with compiled FlexAttention (Triton 3.3), ZeRO-2 (not ZeRO-3), `SEQUENCE_PARALLEL_DEGREE=1` |

With `--strict_data True`, any data error (for example a video read timeout) stops the job instead of skipping the row; resume it from the last checkpoint.

## Packing and Training Budget

AR training packs samples up to the 32,768-token budget. Drafter training uses exactly eight samples per pack, each capped at 32,768 tokens, with a 262,144-token pack capacity. These are the reference launch settings; changing the pack size changes the number of samples processed per update.

The drafter's three sample epochs refer to the weighted dataset mixture. The trainer derives the optimizer-step budget from the mixture, samples per pack, distributed world size, and gradient accumulation. Set source weights in the [data recipe](DATA_PREPARATION.md#recipe-format).

## Checkpoints

Drafter checkpoints contain the configuration, weights, and `modeling_dflash.py`. Each checkpoint records a fingerprint of its target's weights, text configuration, and tokenizer. Loading checks that the target and tokenizer match those used to train the drafter.

Target and drafter checkpoints will be released later. The examples use `<target-checkpoint>` and `<drafter-checkpoint>` placeholders; a drafter should be loaded with its matched target. See [Library Usage](EVALUATION.md#library-usage) for loading and generation.

### Resume

Resume with the same launch command and add `--resume_from_checkpoint /path/to/checkpoint-N` after its three positional arguments. The trainer also detects an existing checkpoint in the output directory. Keep the original recipe, target checkpoint, packing settings, worker count, and distributed topology when restoring the run. Checkpoint directories include per-rank `dataloader_state_rank<N>.pt` files for the saved data-loading state.

## Implementation References

The two VFlash scripts delegate process launch to [`shell/embodied/finetune.sh`](../../shell/embodied/finetune.sh) and use the [target/drafter trainer](../../eaglevl/train/eagle_embodied_finetune_stream_packing.py). These are the underlying implementations of the commands above. Model loading uses the repository's `EagleEmbodiedForConditionalGeneration` class, as shown in [Library Usage](EVALUATION.md#library-usage).
