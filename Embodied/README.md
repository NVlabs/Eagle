<div align="center">

# 🎯 LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding

<p>
  <img src="assets/images/teaser.jpg" alt="LocateAnything Teaser" width="92%">
</p>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](../LICENSE)
[![Model License](https://img.shields.io/badge/Model%20License-NVIDIA%20License-red.svg)](../LICENSE_MODEL)

[[📘Paper](https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf)] [[🤗HF Model](https://huggingface.co/nvidia/LocateAnything-3B)] [[🤗HF Demo](https://huggingface.co/spaces/nvidia/LocateAnything)] [[🌐Project Page](https://research.nvidia.com/labs/lpr/locate-anything/)] [[💻GitHub](https://github.com/NVlabs/Eagle)]

<sub>📚 <a href="document/TRAINING.md">Training</a> &nbsp;·&nbsp; <a href="document/DATA_PREPARATION.md">Data Preparation</a> &nbsp;·&nbsp; <a href="evaluation/README.md">Evaluation</a> &nbsp;·&nbsp; <a href="document/RESULTS.md">Detailed Results</a></sub>

</div>


**LocateAnything** is a vision-language model for fast and high-quality visual grounding, enabling precise object localization, dense detection, and point-based localization across diverse domains in both Enterprise Intelligence and Physical AI. The model adopts a generalist design, supporting tasks such as referring expression grounding, multi-object detection, GUI element grounding, and text localization, with strong performance in complex and cluttered scenes.

- ⚡ **Parallel Box Decoding (PBD)** — atomic, single-step decoding of full bounding boxes / points.
- 🔁 **Hybrid Inference** — Fast Mode (MTP) by default, with seamless NTP fallback for stability.
- 📚 **LocateAnything-Data** — 138M language queries, 785M boxes, covering detection, GUI grounding, referring comprehension, OCR, layout, and pointing.
- 🏆 **State-of-the-Art** — 12.7 BPS on a single H100 (≈ 10× Qwen3-VL, 2.5× Rex-Omni), with SOTA accuracy on LVIS, M6Doc, ScreenSpot-Pro, and more.


## 🎬 Visual Demo

<table>
<tr>
<td width="60.4%" align="center" valign="top">
<video src="https://github.com/user-attachments/assets/814e042c-baf4-41ba-b7c9-655e909f82d6" autoplay loop muted playsinline controls width="100%"></video>

<b>Dense Object Detection</b><br>
<sub>LocateAnything performs diverse localization tasks under a unified VLM — document understanding, GUI grounding, dense object detection, and OCR.</sub>
</td>
<td width="39.6%" align="center" valign="top">
<video src="https://github.com/user-attachments/assets/154b5f61-e26e-451b-9518-88c63d437cc4" autoplay loop muted playsinline controls width="100%"></video>

<b>Fast Decoding Speed</b><br>
<sub>Parallel Box Decoding (PBD) vs. Quantized Coordinate Decoding — PBD predicts each bounding box atomically in a single forward pass for substantially faster throughput.</sub>
</td>
</tr>
</table>

## 🧠 Method

Vision-language models (VLMs) commonly formulate visual grounding and detection as a coordinate-token generation problem, serializing each 2D box into multiple 1D tokens that are learned and decoded largely independently. This token-by-token decoding mismatches the coupled structure of box geometry and creates a practical inference bottleneck due to strictly sequential generation.

We introduce **LocateAnything**, a unified generative grounding and detection framework based on **Parallel Box Decoding (PBD)**. By decoding geometric elements such as bounding boxes and points as atomic units in a single step, LocateAnything preserves intra-box geometric coherence and unlocks substantial parallelism. We further develop a scalable data engine and curate **LocateAnything-Data**, a large-scale dataset with **138M+ training samples**. Extensive evaluations show that LocateAnything advances the speed–accuracy frontier on diverse benchmarks.

### Feature Summary

| | |
|:--|:--|
| ⚡ **Parallel Box Decoding (PBD)** | Treats each bounding box (or point) as an **atomic unit** and predicts the full coordinate set in a single forward pass — preserves intra-box geometric coherence and prevents irregular structural tokens. |
| 🔁 **Hybrid Inference Mode** | **Fast Mode (MTP)** by default; seamless fallback to **Slow Mode (NTP)** when parallel outputs are unreliable. Most of the speed gains with robust, format-correct outputs. |
| 📚 **LocateAnything-Data** | **138M** language queries and **785M** boxes across detection, GUI grounding, referring comprehension, OCR, layout, and pointing. |
| 🏆 **State-of-the-Art** | **12.7 BPS** on a single H100 — **10×** faster than Qwen3-VL (1.1 BPS), **2.5×** faster than Rex-Omni (5.0 BPS), with SOTA accuracy on LVIS, M6Doc, ScreenSpot-Pro, and more. |

### Parallel Box Decoding

<p align="center">
  <img src="assets/images/method_decoding_comparison.jpg" alt="NTP vs MTP vs PBD" width="88%">
  <br>
  <sub><b>Comparison of standard token decoding methods vs. Parallel Box Decoding (PBD).</b> NTP generates coordinate values one by one; standard MTP produces irregular distributions; <b>PBD</b> generates a complete atomic box in a single parallel step.</sub>
</p>

<table>
<tr>
<td width="50%" valign="top">

#### 📦 Box-Aligned Atomic Units
- **Input.** An image and a natural language query. The vision encoder extracts visual tokens at native resolution.
- **Parallel Decoding.** Each bounding box (or point) is an *atomic unit* of constant length; the full coordinate set `(x₁, y₁, x₂, y₂)` is predicted in one parallel step.
- **Architecture.** Moon-ViT vision encoder + Qwen2.5 language decoder, bridged by an MLP projector.

</td>
<td width="50%" valign="top">

#### 🚦 Flexible Inference Modes
- **Fast Mode (MTP).** Full boxes in parallel for maximum throughput — on-device robotics and embodied agents.
- **Slow Mode (NTP).** Autoregressive coordinate decoding for maximum stability — high-precision labeling and offline evaluation.
- **Hybrid Mode.** MTP by default with NTP fallback on format irregularity or spatial ambiguity.

</td>
</tr>
</table>

<p align="center">
  <img src="assets/images/method_overview.jpg" alt="LocateAnything architecture" width="92%">
  <br>
  <sub><b>Architecture overview of LocateAnything with Parallel Box Decoding.</b></sub>
</p>

#### On-Demand Inference: Corrected NTP Re-decoding

When parallel decoding encounters *format irregularity* (malformed syntax at category boundaries) or *spatial ambiguity* (intermediate coordinates between densely arranged objects), the compromised block is discarded and generation reverts to the last verified prefix. NTP then autoregressively generates tokens for the problematic block before switching back to MTP.

<p align="center">
  <img src="assets/images/method_redecoding.jpg" alt="Corrected NTP Re-decoding" width="78%">
</p>



## 📚 LocateAnything-Data

> **138M diverse language queries · 785M boxes · 12M unique images**

<p align="center">
  <img src="assets/images/data_distribution.jpg" alt="LocateAnything-Data" width="88%">
</p>

| Task | Share of Queries | Description |
|:--|:--:|:--|
| 🎯 **General Object Detection** | **66.9%** | Dense bounding box supervision for precise coordinate alignment (83.1% of all boxes). |
| 🖥️ **GUI Element Grounding** | **16.5%** | Embodied agents and graphical user interface navigation. |
| 💬 **Referring Comprehension** | **7.3%** | Linking complex natural language intents to specific spatial regions. |
| 🔤 **Text Localization (OCR)** | **3.6%** | Perceiving and tightly grounding textual information in images. |
| 📐 **Layout Grounding** | **3.5%** | Structural reasoning for document and scene layout understanding. |
| 📍 **Point-Based Localization** | **2.2%** | Fine-grained coordinate predictions. |



## 🚀 Installation

```bash
git clone https://github.com/NVlabs/Eagle.git eagle
cd eagle/Embodied
pip install -e .
```

Key dependencies (auto-installed): `transformers==4.57.1`, `tokenizers==0.22.0`, `deepspeed==0.15.4`, `accelerate==1.5.2`, `timm>=1.0.11`, `liger_kernel==0.3.1`, `peft==0.12.0`, `decord`.

<details>
<summary><b>⚙️ Magi Attention (Hopper / Blackwell only)</b></summary>

<br>

For long-context training and inference (16K–32K+), install [MagiAttention](https://sandai-org.github.io/MagiAttention/docs/main/user_guide/install.html). It **only supports Hopper and Blackwell** GPU architectures.

```bash
git clone https://github.com/SandAI-org/MagiAttention.git
cd MagiAttention
git checkout v1.0.5
git submodule update --init --recursive
pip install -r requirements.txt
pip install --no-build-isolation .
```

> For non-Hopper/Blackwell GPUs (A100, L40, etc.), the Hugging Face model release also includes the `la_flash` batch runtime described below. It uses FlashAttention varlen sparse range plans and avoids the dense SDPA masks used by the stock path.

</details>



## ⚡ Quick Start

```python
import torch
from PIL import Image
from locateanything_worker import LocateAnythingWorker

worker = LocateAnythingWorker("nvidia/LocateAnything-3B")
img = Image.open("example.jpg").convert("RGB")

# Object Detection
print(worker.detect(img, ["person", "car", "bicycle"])["answer"])

# Phrase Grounding
print(worker.ground_multi(img, "people wearing red shirts")["answer"])

# Scene Text Detection
print(worker.detect_text(img)["answer"])

# GUI Grounding (point)
print(worker.ground_gui(img, "the search button", output_type="point")["answer"])

# Pointing
print(worker.point(img, "the traffic light")["answer"])
```

See [`locateanything_worker.py`](locateanything_worker.py) for the full worker API.

## 🚀 Batch Inference Release

The [Hugging Face model repository](https://huggingface.co/nvidia/LocateAnything-3B) now includes optional high-throughput inference utilities:

- `batch_infer.py`: JSONL/image-query batch inference CLI.
- `batch_utils/`: batched hybrid MTP/NTP scheduler and sampling runtime.
- `kernel_utils/`: **LA Flash** sparse range utilities implemented with FlashAttention varlen. This path does not build or ship a custom C++/CUDA extension.

`LA_FLASH_ATTN=la_flash` keeps LocateAnything's hybrid decoding path while running sparse range plans through FlashAttention. It is intended for inference/evaluation; training should continue to use the standard model code path.

```bash
hf download nvidia/LocateAnything-3B --local-dir LocateAnything-3B
cd LocateAnything-3B
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python batch_infer.py \
  --model . \
  --attn la_flash \
  --vision-attn flash_attention_2 \
  --scheduler pipeline \
  --batch-size 4 \
  --image /path/to/image.jpg \
  --query "person</c>car"
```

A100 4K probe, real 3840x2160 street image, `query=vehicle`, `batch_size=4`, raw PIL input, `in_token_limit=25600`, hybrid MTP inference:

| Backend | Attention path | Time | Peak reserved memory |
|:--|:--|--:|--:|
| `sdpa` | Dense SDPA masks | 8.2600 s | 35.12 GB |
| `la_flash` | FlashAttention sparse range plan | 8.0314 s | 11.71 GB |

The worker API can also use the released batch runtime when `batch_utils/` and `kernel_utils/` are on `PYTHONPATH`:

```python
from PIL import Image
from locateanything_worker import LocateAnythingWorker

worker = LocateAnythingWorker(
    "nvidia/LocateAnything-3B",
    use_batch_runtime=True,
    attn="la_flash",
    vision_attn="flash_attention_2",
    scheduler="pipeline",
)

img1 = Image.open("street_1.jpg").convert("RGB")
img2 = Image.open("street_2.jpg").convert("RGB")

results = worker.detect_batch([
    (img1, ["person", "car"]),
    (img2, ["traffic light", "bus"]),
])

for result in results:
    print(result["answer"])
```

### 🧾 Output Format

The model outputs special tokens to represent bounding boxes and points:

- **Bounding box:** `<ref>label</ref><box><x1><y1><x2><y2></box>` — coordinates are integers in `[0, 1000]` (divide by 1000 for relative coordinates).
- **Point:** `<box><x><y></box>`
- **No object:** `<box>none</box>`

```python
import re

def parse_boxes(answer: str, image_width: int, image_height: int):
    boxes = []
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = [int(g) for g in m.groups()]
        boxes.append({
            "x1": x1 / 1000 * image_width,  "y1": y1 / 1000 * image_height,
            "x2": x2 / 1000 * image_width,  "y2": y2 / 1000 * image_height,
        })
    return boxes
```


## 🏋️ Training (Continual SFT)

Full fine-tuning from a pretrained checkpoint — single command:

```bash
torchrun --nproc_per_node=8 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path nvidia/LocateAnything-3B \
  --meta_path "./locany_recipe/your_recipe.json" \
  --output_dir work_dirs/my_sft \
  --max_steps 25000 \
  --learning_rate 2e-5 \
  --bf16 True \
  --block_size 6 \
  --attn_implementation magi \
  --max_seq_length 16384 \
  --deepspeed "deepspeed_configs/zero_stage2_config.json"
```

For the complete training guide (all arguments, data recipe format, multi-node setup, streaming packing, checkpoint resume), see **[Training Documentation](document/TRAINING.md)**.

For data format details, annotation conventions, and recipe configuration, see **[Data Preparation](document/DATA_PREPARATION.md)**.


## 📈 Evaluation

```bash
# COCO
bash evaluation/scripts/eval_coco.sh --model_path path/to/model --test_jsonl ... --image_root ... --output_dir ...

# LVIS
bash evaluation/scripts/eval_lvis.sh --model_path path/to/model --test_jsonl ... --image_root ... --output_dir ...

# Grounding (Dense200, DocLayNet, HumanRef, RefCOCOg, VisDrone, etc.)
bash evaluation/scripts/eval_grounding.sh --dataset Dense200 --eval_type box_eval --model_path ... --image_root ... --output_base ...

# ScreenSpot-Pro
bash evaluation/scripts/eval_sspro.sh --model_path ... --test_jsonl ... --image_root ... --output_dir ...
```

See the [Evaluation Guide](evaluation/README.md) for setup and dataset preparation.


## 🏆 Results at a Glance

> **State-of-the-art accuracy at 10× the throughput.** Full benchmark tables, ablation studies, and qualitative visualizations are documented separately in **[📊 Detailed Results](document/RESULTS.md)**.

<div align="center">

| Benchmark | Metric | LocateAnything-3B | vs. Best Baseline |
|:--|:--:|:--:|:--:|
| **Throughput (H100)** | BPS | **12.7** | **10×** Qwen3-VL · **2.5×** Rex-Omni |
| **LVIS** | F1@Mean | **50.7** | **+3.8** vs. Rex-Omni |
| **COCO** | F1@Mean | **54.7** | **+1.8** vs. Rex-Omni |
| **Dense200** | F1@Mean | **58.7** | **+0.4** vs. Rex-Omni |
| **VisDrone** | F1@Mean | **39.9** | **+1.4** vs. G-DINO-Swin-T |
| **DocLayNet** | F1@Mean | **76.8** | **+6.1** vs. Rex-Omni |
| **M6Doc** | F1@Mean | **70.1** | **+14.5** vs. Rex-Omni |
| **TotalText** (OCR) | F1@Mean | **43.3** | **+2.7** vs. Rex-Omni |
| **ScreenSpot-Pro** | Avg | **60.3** | **+2.3** vs. GUI-Owl-32B |
| **HumanRef** | F1@0.95 | **68.8** | **+3.4** vs. Rex-Omni |
| **RefCOCOg val** | F1@Mean | **76.7** | **+2.0** vs. Qwen3-VL-8B |

</div>

> 👉 **For full per-benchmark tables, decoding-mode comparisons, box-ordering ablation, throughput scaling curves, and qualitative visualizations, see [`document/RESULTS.md`](document/RESULTS.md).**


## 📖 Citation

```bibtex
@article{wang2025locateanything,
  title   = {LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding},
  author  = {Shihao Wang and Shilong Liu and Yuanguo Kuang and Xinyu Wei and
             Yangzhou Liu and Zhiqi Li and Yunze Man and Guo Chen and
             Andrew Tao and Guilin Liu and Jan Kautz and Lei Zhang and Zhiding Yu},
  journal = {arXiv:2605.27365},
  year    = {2026},
}
```


## 🙏 Acknowledgement

We thank the [Rex-Omni](https://github.com/IDEA-Research/Rex-Omni) Team for their evaluation framework.

## 📜 License

- See [LICENSE](../LICENSE) for the code of this repository.
- See [LICENSE_MODEL](./LICENSE_MODEL) for the models of LocateAnything.

<div align="center">
<sub>© 2026 LocateAnything Team · Built with a unified grounding mindset.</sub>
</div>
