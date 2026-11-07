<div align="center">

# 🦅 Eagle Embodied

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](../LICENSE)
[![Model License](https://img.shields.io/badge/Model%20License-NVIDIA%20License-red.svg)](LICENSE_MODEL)

<sub>🎯 <a href="document/lam/README.md">LocateAnything</a> &nbsp;·&nbsp; ⚡ <a href="document/vflash/README.md">VFlash</a> &nbsp;·&nbsp; 📚 <a href="#-documentation">Documentation</a></sub>

</div>

**Eagle Embodied** is NVIDIA's family of vision-language models for visual grounding, spatial reasoning, and visual planning. It connects image and video understanding with predictions of where objects are and how robots, vehicles, and objects move.

This repository provides training, inference, and evaluation code for **LocateAnything**, which accelerates visual localization, and **VFlash**, which accelerates visual planning.

## 🧩 Methods

### 🎯 [LocateAnything](document/lam/README.md)

Fast, accurate visual grounding through **Parallel Box Decoding**. LocateAnything predicts complete bounding boxes and points in parallel, supporting object detection, referring expressions, GUI grounding, text localization, and document layout analysis.

[Method and demos](document/lam/README.md) · [Model weights](https://huggingface.co/nvidia/LocateAnything-3B) · [Results](document/RESULTS.md)

### ⚡ [VFlash](document/vflash/README.md)

Fast, accurate visual planning through **Flash Primitive Decoding**. A lightweight drafter predicts complete visual primitives in parallel, supporting robot and driving trajectories, dense 3D motion, boxes, and points.

[Method and demos](document/vflash/README.md) · [Technical report](https://research.nvidia.com/labs/lpr/vflash/assets/paper/vflash.pdf) · [Project page](https://research.nvidia.com/labs/lpr/vflash/)

VFlash training and evaluation code is available; target and drafter checkpoints are forthcoming.

## 🚀 Installation

Clone the repository and run commands from `Embodied/`:

```bash
git clone https://github.com/NVlabs/Eagle.git eagle
cd eagle/Embodied
```

Install a CUDA-compatible PyTorch build first, then choose the dependencies for your task. The base package can be installed with `pip install -e .`.

| Task | Install | Requirements |
|:--|:--|:--|
| LocateAnything training | `pip install -e '.[locate]'` | MagiAttention on Hopper or Blackwell GPUs |
| VFlash training | `pip install -e '.[vflash]'` | PyTorch 2.8, Triton 3.3, FlashAttention |
| Benchmark evaluation | `pip install -e '.[evaluation]'` | Additional metric dependencies |

See the [LocateAnything](document/lam/README.md#-installation) and [VFlash](document/vflash/README.md#-installation) guides for method-specific setup and usage.

## 📚 Documentation

| Guide | LocateAnything | VFlash |
|:--|:--|:--|
| Overview | [README](document/lam/README.md) | [README](document/vflash/README.md) |
| Inference | [Quick start](document/lam/README.md#-quick-start) | [Quick start](document/vflash/README.md#-quick-start) |
| Training | [Training guide](document/TRAINING.md) | [Training guide](document/vflash/TRAINING.md) |
| Data preparation | [Data guide](document/DATA_PREPARATION.md) | [Data guide](document/vflash/DATA_PREPARATION.md) |
| Evaluation | [Evaluation guide](evaluation/README.md) | [Evaluation guide](document/vflash/EVALUATION.md) |
| Results | [Detailed results](document/RESULTS.md) | [Detailed results](document/vflash/RESULTS.md) |

## 📁 Repository Layout

The code is organized around shared Eagle Embodied components and method-specific training and decoding modules.

<details>
<summary>Browse the source directories</summary>

```text
Embodied/
├── eaglevl/
│   ├── model/
│   │   ├── moon_vit/        # Vision encoder
│   │   ├── eagle_embodied/  # Eagle Embodied VLM
│   │   ├── lam/             # LocateAnything model and decoding
│   │   └── vflash/          # Drafter and Flash Primitive Decoding
│   ├── train/              # Training and data loading, including VFlash adapters
│   └── utils/              # Processors and checkpoint support
├── evaluation/             # Grounding evaluation and VFlash benchmark runners
├── shell/                  # Training launchers
├── deepspeed_configs/      # Distributed training configurations
├── locateanything_worker.py
├── assets/                 # Figures, videos, and reports
└── document/               # Method pages and usage guides
```

</details>

## 📜 License

See [LICENSE](../LICENSE) for the code and [LICENSE_MODEL](LICENSE_MODEL) for the models.
