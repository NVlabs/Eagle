<div align="center">

# ⚡ VFlash: Fast, Accurate Visual Planning via Flash Primitive Decoding

<p>
  <img src="../../assets/vflash/overview.png" alt="VFlash overview" width="92%">
</p>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](../../../LICENSE)
[![Model License](https://img.shields.io/badge/Model%20License-NVIDIA%20License-red.svg)](../../LICENSE_MODEL)

[[📘Technical Report](https://arxiv.org/abs/2611.08449)] [[🌐Project Page](https://research.nvidia.com/labs/lpr/vflash/)] [[💻GitHub](https://github.com/NVlabs/Eagle)]

<sub>📚 <a href="TRAINING.md">Training</a> &nbsp;·&nbsp; <a href="DATA_PREPARATION.md">Data Preparation</a> &nbsp;·&nbsp; <a href="EVALUATION.md">Evaluation</a> &nbsp;·&nbsp; <a href="RESULTS.md">Detailed Results</a></sub>

</div>


**VFlash** is a generalist decoding framework for fast, accurate visual planning in Physical AI. It uses the predictable structure of visual plans to generate complete primitives in parallel, reducing the sequential cost of planning for robots and autonomous vehicles.

- ⚡ **Flash Primitive Decoding** — a complete visual primitive in one parallel drafter pass.
- 🎯 **Higher trajectory accuracy** — 18–61% lower 3D robot trajectory ADE than the autoregressive target in the technical report.
- 🧩 **Generalist visual planning** — boxes, points, robot and driving trajectories, dense 3D motion, and sub-task text.
- 🚀 **Faster robot inference** — 5.1–5.2× lower per-step MolmoSpaces latency on Jetson Thor and DGX Spark in the technical report.

> **Model availability:** Target and drafter checkpoints will be released later. The examples below use paths to locally trained checkpoints. [Detailed Results](RESULTS.md) documents the paper's experiments and results.


## 🎬 Visual Demo

<!-- Keep the Markdown fallback: GitLab strips the HTML video tag. -->
<video src="https://github.com/user-attachments/assets/935934d0-78ca-4d0b-8fc9-b1f8e9f970e9" controls playsinline width="100%">

![VFlash film](../../assets/vflash/full-h264.mp4)

</video>

[Watch the full video on the project website](https://research.nvidia.com/labs/lpr/vflash/assets/media/v27/full-4k-h264.mp4)


## 🧠 Method

Vision-language models express spatial predictions as coordinate sequences. Autoregressive decoding generates these sequences one token at a time, so longer trajectories and larger sets of objects require more sequential computation. Visual primitives offer a useful structure: their type, dimensionality, and requested length determine what each output position represents.

**VFlash** turns that structure into **Flash Primitive Decoding**. A lightweight drafter reads the target VLM's visual and language context and predicts a complete primitive in parallel. It learns from annotated geometry and accepts valid primitives without requiring token-level agreement with the target, allowing spatial predictions to improve beyond the target's autoregressive output.

### Feature Summary

| Feature | Design |
|:--|:--|
| ⚡ **Primitive-aligned blocks** | Train and decode complete boxes, points, and trajectories as single blocks. |
| 🧭 **Task-dependent lengths** | Tool calls specify trajectory lengths; primitive types determine coordinate layouts. |
| 🎯 **Learning from geometry** | Train the drafter on annotated primitives while keeping the target frozen. |
| 🔁 **Text and box decoding** | Retain verified speculative decoding for text and spatial-confidence checks for boxes. |

### Flash Primitive Decoding

<p align="center">
  <img src="../../assets/vflash/pipeline.png" alt="Flash Primitive Decoding" width="92%">
  <br>
  <sub><b>Flash Primitive Decoding.</b> The requested structure defines a complete primitive block. A lightweight drafter predicts it in parallel from frozen VLM features, and the target processes the accepted plan to update its context.</sub>
</p>

<table>
<tr>
<td width="50%" valign="top">

#### 🧱 Primitive-Aligned Drafting

- **Input.** Images or video frames together with a task instruction.
- **Context.** The target VLM encodes the observation and preceding text.
- **Prediction.** A five-layer drafter fills the primitive's coordinate slots in one parallel pass.

</td>
<td width="50%" valign="top">

#### 🚦 Flexible Decoding

- **Trajectories and motion.** Accept complete, valid primitives without target-token verification.
- **Text.** Verify drafted text blocks against the target's predictions.
- **Boxes.** Use a spatial-confidence guard with AR fallback for uncertain coordinates.

</td>
</tr>
</table>

See [model architecture and training objectives](TRAINING.md#model-and-objective) and [decoding rules](EVALUATION.md#decoding-rules) for the implementation details.


## 📚 Visual Planning Tasks

VFlash supports the following visual planning outputs:

| Task | Prediction |
|:--|:--|
| 🦾 **Robot planning** | Single-arm or dual-arm gripper trajectories in 2D or 3D |
| 🚗 **Driving** | Ego-frame positions and headings |
| 🌀 **Dense motion** | Future 3D trajectories of queried scene points |
| 📦 **Visual grounding** | Bounding boxes and points |
| 💬 **Sub-task planning** | Textual action descriptions |

For coordinate ranges, input conventions, and training examples, see **[Data Preparation](DATA_PREPARATION.md)**.


## 🚀 Installation

Install a CUDA build of PyTorch 2.8, then install the VFlash dependencies:

```bash
git clone https://github.com/NVlabs/Eagle.git eagle
cd eagle/Embodied
pip install -e '.[vflash]'
```

Key dependencies: `transformers==4.57.1`, `torch>=2.8.0a0,<2.9`, `torchvision>=0.23.0a0,<0.24`, `triton>=3.3,<3.4`, and FlashAttention. Drafter training requires compiled FlexAttention. See [environment requirements](TRAINING.md#installation) for details.

For evaluation, install `pip install -e '.[evaluation]'`. COCO scoring also requires `pip install ./evaluation/fastevaluate`.


## ⚡ Quick Start

Load the VFlash autoregressive target and its matched drafter. Replace the checkpoint, image, and prompt placeholders with your local inputs:

```python
from evaluation.inference_compat import decode_generation_output, load_model_and_processor
from evaluation.vflash.generation import build_model_inputs

model, processor = load_model_and_processor(
    "<target-checkpoint>",
    use_fast=False,
    local_code=True,                          # repository model; VFlash does not run through remote code
    local_processor=True,
    attn_implementation="flash_attention_2",
    training_runtime=True,
    draft_model_path="<drafter-checkpoint>",  # attaches the drafter with model.load_vflash_draft()
)
tokenizer = processor.tokenizer

# A record in the training format. The empty answer makes the prompt end with " /no_think", as in training.
record = {
    "image": "scene.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\n<task prompt in the training format>"},
        {"from": "gpt", "value": ""},
    ],
}
inputs = build_model_inputs(processor, record)  # bfloat16 tensors on CUDA

output_ids, _, stats = model.generate(
    **inputs,
    max_new_tokens=320,
    do_sample=False,
    generation_mode="vflash",
    verify_atoms=False,             # commit schema-valid primitives without target verification
    return_vflash_stats=True,       # returns (ids, None, stats)
    atomic_decoding_tokenizer=tokenizer,
    vflash_draft_temperature=0.0,   # argmax in every primitive slot
)
answer = decode_generation_output(
    output_ids, inputs["input_ids"], processor, strip_prompt=True, strip_end_tokens=True
)
print(answer, stats.acceptance_lengths)
```

For the AR baseline, use the same target and inputs with the following generation and decoding calls. AR returns the generated token IDs directly:

```python
output_ids = model.generate(
    **inputs,
    max_new_tokens=320,
    do_sample=False,
)
answer = decode_generation_output(
    output_ids, inputs["input_ids"], processor, strip_prompt=True, strip_end_tokens=True
)
print(answer)
```

The [inference guide](EVALUATION.md#library-usage) covers model loading, box decoding, and generation statistics.

### 🧾 Output Format

The model represents coordinates with number tokens `<0>` through `<1000>`. Structural tokens identify each primitive, and trajectory tool calls specify the block length. For example:

```text
<action>Pull the drawer open.</action><tool_call>{"name":"atomic_decoding","arguments":{"block_size":32}}</tool_call><robot_single><521><489><61>…<509><470><85></robot>
```

Each prompt defines the coordinate frame and scale. See [primitive formats and block lengths](DATA_PREPARATION.md#physical-primitives) for robot, driving, motion, box, and point outputs.


## 🏋️ Training

First fine-tune an autoregressive target on the task mixture, then freeze it and train the VFlash drafter. Run these launchers on each allocated node with the appropriate node rank and master address:

```bash
# Stage 1: AR target. Fine-tunes an Eagle Embodied checkpoint on physical-primitive answers.
GPUS=8 NNODES=8 NODE_RANK=<node-rank> MASTER_ADDR=<master-address> \
  bash shell/vflash/train_ar.sh <recipe.json> <target-output-dir> <base-eagle-embodied-checkpoint>

# Stage 2: drafter. Freezes the target from stage 1 and trains the five-layer drafter.
GPUS=8 NNODES=8 NODE_RANK=<node-rank> MASTER_ADDR=<master-address> \
  bash shell/vflash/train_drafter.sh <recipe.json> <drafter-output-dir> <target-checkpoint>
```

For the reference settings, packing budgets, checkpoint binding, and launch options, see **[Training Documentation](TRAINING.md)**. For annotation and recipe formats, see **[Data Preparation](DATA_PREPARATION.md)**.


## 📈 Evaluation

The task runners compare VFlash with autoregressive decoding of the same target. Add `--draft` to use VFlash, or omit it for the AR baseline. For example, generate and score robot trajectories with:

```bash
python -m evaluation.vflash.robot3d --model <target-checkpoint> --draft <drafter-checkpoint> \
  --dataset agibot --input <robot3d/agibot.jsonl> --output <out>/robot3d_agibot.jsonl \
  --media-root <media> --max-new-tokens 320
python -m evaluation.vflash.trajectory <out>/robot3d_agibot.jsonl --output <out>/robot3d_agibot.metrics.json
```

| Task | Benchmarks | Guide |
|:--|:--|:--|
| Robot 3D trajectories | AgiBot, InternA1, MolmoBot | [Evaluation guide](EVALUATION.md#robot-3d-trajectories) |
| Driving trajectories | Waymo, NAVSIM, Alpamayo | [Evaluation guide](EVALUATION.md#driving-trajectories) |
| Dense 3D motion | HOT3D, WorldTrack, DAVIS | [Evaluation guide](EVALUATION.md#dense-3d-motion) |
| Boxes | COCO, Dense200 | [Evaluation guide](EVALUATION.md#box-detection) |

The **[Evaluation Guide](EVALUATION.md)** specifies the prepared data fields, decoding settings, and metrics for each benchmark.


## 🏆 Results at a Glance

> **Visual planning with parallel primitive decoding.** Full benchmark tables, model variants, downstream applications, and speed measurements are in **[📊 Detailed Results](RESULTS.md)**.

All results below are from the technical report.

### 3D Robot Trajectories

ADE and FDE in meters (↓). See [model settings and all target variants](RESULTS.md#robot-3d-trajectories).

| Benchmark | AR ADE | VFlash ADE | AR FDE | VFlash FDE |
|:--|--:|--:|--:|--:|
| InternA1 | 0.130 | **0.060** | 0.196 | **0.091** |
| AgiBot | 0.084 | **0.036** | 0.129 | **0.057** |
| MolmoBot | 0.066 | **0.052** | 0.102 | **0.080** |

### 2D Trajectories

Errors in normalized image coordinates (↓). [Full comparison](RESULTS.md#robot-2d-trajectories).

| Benchmark | AR DFD | VFlash DFD | AR HD | VFlash HD | AR RMSE | VFlash RMSE |
|:--|--:|--:|--:|--:|--:|--:|
| ShareRobot-Bench | 0.202 | **0.194** | 0.175 | **0.162** | 0.142 | **0.133** |
| VABench-V | 0.130 | **0.126** | 0.125 | **0.116** | 0.093 | **0.089** |

### Box Detection

Mean F1 in percent (↑) and boxes per second on one H100 (BPS, ↑). VFlash uses the spatial-confidence guard with AR fallback. [Decoding-mode comparison](RESULTS.md#box-detection).

| Benchmark | AR F1 | VFlash F1 | AR BPS | VFlash BPS |
|:--|--:|--:|--:|--:|
| COCO | **48.68** | 47.79 | 3.15 | **8.76** |
| Dense200 | 60.10 | **61.78** | 6.73 | **17.46** |

### Dense 3D Motion

Frozen MolmoMotion-AR target. ADE and FDE in meters (↓); PWT (↑). [Full settings](RESULTS.md#dense-3d-motion).

| Benchmark | AR ADE | VFlash ADE | AR FDE | VFlash FDE | AR PWT | VFlash PWT |
|:--|--:|--:|--:|--:|--:|--:|
| HOT3D | **0.109** | 0.122 | 0.217 | **0.200** | **0.444** | 0.391 |
| WorldTrack | **0.143** | 0.145 | **0.261** | 0.300 | **0.445** | 0.408 |
| DAVIS | 1.227 | **0.893** | 2.108 | **1.716** | **0.153** | 0.143 |

[Downstream robot control and driving](RESULTS.md#downstream-applications) · [Planning latency](RESULTS.md#planning-latency)


## 📖 Citation

```bibtex
@article{chou2026vflash,
  title   = {{VFlash}: Fast, Accurate Visual Planning via Flash Primitive Decoding},
  author  = {Chou, Yu-Cheng and Wang, Shihao and Huang, Jin and Huang, Chi-Pin and Yang, Chiao-An and Fan, Xianzhe and Wang, Jing and Li, Zhiqi and Kuang, Yuanguo and Zheng, Ruijie and Bjorck, Johan and Liu, Sifei and Su, Hang and Badki, Abhishek and Zhang, Lei and Yuille, Alan and Fan, Linxi and Kautz, Jan and Yu, Zhiding},
  journal = {arXiv:2611.08449},
  year    = {2026}
}
```


## 🙏 Acknowledgement

VFlash builds on [DFlash](https://github.com/z-lab/dflash): the drafter and the speculative decoding loop are adapted from its code. The PointMotionBench metrics are adapted from [MolmoMotion](https://github.com/allenai/molmo-motion), the photometric augmentation reproduces the sensor-noise family of LIBERO-plus, and the box evaluation follows the [Rex-Omni](https://github.com/IDEA-Research/Rex-Omni) protocol.

## 📜 License

- See [LICENSE](../../../LICENSE) for the code of this repository.
- See [LICENSE_MODEL](../../LICENSE_MODEL) for the models of VFlash.
