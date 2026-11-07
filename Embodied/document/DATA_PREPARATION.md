# Data Preparation Guide

This guide explains how to prepare training data for LocateAnything fine-tuning.

## Table of Contents

1. [Overview](#overview)
2. [Annotation Format (JSONL)](#annotation-format-jsonl)
3. [Data Recipe (JSON)](#data-recipe-json)
4. [Special Tokens](#special-tokens)
5. [Task-Specific Formats](#task-specific-formats)
6. [Image & Media Requirements](#image--media-requirements)
7. [Directory Structure Example](#directory-structure-example)

---

## Overview

LocateAnything uses a two-level data configuration:

1. **JSONL annotation files** — each line is a training sample in ShareGPT conversation format.
2. **Recipe JSON** — a single JSON file that maps dataset names to their annotation files, image roots, and sampling weights. This is passed to training via `--meta_path`.

```
your_recipe.json          ← recipe: ties everything together
├── detection.jsonl       ← annotation: one sample per line
├── gui_grounding.jsonl
├── ocr.jsonl
└── ...
```

---

## Annotation Format (JSONL)

Each line in a JSONL file is a self-contained JSON object with the following structure:

```jsonl
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}], "image": "relative/path/to/image.jpg"}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `conversations` | list | Yes | ShareGPT-style conversation turns |
| `image` | str | No | Single image path (relative to recipe `root`) |
| `image_list` | list[str] | No | Multiple image paths |
| `video` | str | No | Video file path |
| `video_list` | list[str] | No | Multiple video paths |

### Conversation Format

Each turn in `conversations` has two fields:

| Field | Values | Description |
|-------|--------|-------------|
| `from` | `"human"` / `"gpt"` | Speaker role |
| `value` | str | Message content |

### Image Placeholders

Use `<image-1>`, `<image-2>`, ... in the conversation text to indicate where each image should be inserted. If no placeholder is present but an image is provided, `<image-1>` is automatically prepended to the first user message.

### Coordinate Convention

All bounding box coordinates are **normalized integers in `[0, 1000]`**. To convert to pixel coordinates, divide by 1000 and multiply by image width/height.

---

## Data Recipe (JSON)

The recipe file is a single JSON object where each key is a dataset name and the value is its configuration:

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
  },
  "my_text_only_data": {
    "annotation": "path/to/text_qa.jsonl",
    "root": "",
    "repeat_time": 0.5
  }
}
```

### Recipe Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `annotation` | str or list[str] | *required* | Path(s) to JSONL annotation files. Multiple files are merged. |
| `root` | str | `""` | Root directory for resolving relative image/video paths. |
| `repeat_time` | float | `1.0` | Sampling weight. `>=1`: repeat the dataset N epochs. `<1`: downsample (e.g., `0.5` uses 50% of data). |
| `data_augment` | bool | `false` | Apply resize augmentation during training. |

---

## Special Tokens

LocateAnything uses the following special tokens (automatically added to the tokenizer during training):

### Localization Tokens

| Token | Description |
|-------|-------------|
| `<ref>` / `</ref>` | Wraps a category or description label |
| `<box>` / `</box>` | Wraps coordinate values |
| `</c>` | Separator between multiple categories in a detection prompt |
| `<0>` ~ `<1000>` | 1001 coordinate tokens representing normalized positions |

### Output Format in Training Labels

Bounding boxes in `gpt` responses use the following format:

```
<ref>label</ref><box><x1><y1><x2><y2></box>
```

Where `<x1>`, `<y1>`, `<x2>`, `<y2>` are coordinate tokens like `<100>`, `<200>`, etc.

Points use:

```
<box><x><y></box>
```

When no object is found:

```
<box>none</box>
```

---

## Task-Specific Formats

### Object Detection

Detect multiple categories. Use `</c>` as category separator in the prompt.

```jsonl
{"conversations": [{"from": "human", "value": "Locate all the instances that matches the following description: car</c>person</c>bicycle."}, {"from": "gpt", "value": "<ref>car</ref><box><120><200><450><500></box><ref>person</ref><box><50><100><200><600></box><ref>person</ref><box><300><150><420><580></box>"}], "image": "coco/train2017/000001.jpg"}
```

### Phrase Grounding (Single Instance)

Locate one specific instance matching a description.

```jsonl
{"conversations": [{"from": "human", "value": "Locate a single instance that matches the following description: the red car on the left."}, {"from": "gpt", "value": "<ref>the red car on the left</ref><box><100><200><400><500></box>"}], "image": "train/00002.jpg"}
```

### Phrase Grounding (Multiple Instances)

Locate all instances matching a description.

```jsonl
{"conversations": [{"from": "human", "value": "Locate all the instances that match the following description: people wearing hats."}, {"from": "gpt", "value": "<ref>people wearing hats</ref><box><100><50><300><400></box><box><500><80><700><450></box>"}], "image": "train/00003.jpg"}
```

### Scene Text Detection (OCR)

Detect all text regions in the image.

```jsonl
{"conversations": [{"from": "human", "value": "Detect all the text in box format."}, {"from": "gpt", "value": "<ref>STOP</ref><box><100><200><250><280></box><ref>Main St</ref><box><400><50><650><120></box>"}], "image": "ocr/scene_001.jpg"}
```

### GUI Grounding (Box)

Locate a UI element as a bounding box.

```jsonl
{"conversations": [{"from": "human", "value": "Locate the region that matches the following description: the search button."}, {"from": "gpt", "value": "<ref>the search button</ref><box><800><20><950><60></box>"}], "image": "gui/screenshot_001.png"}
```

### GUI Grounding / Pointing (Point)

Point to a specific element.

```jsonl
{"conversations": [{"from": "human", "value": "Point to: the settings icon."}, {"from": "gpt", "value": "<box><950><30></box>"}], "image": "gui/screenshot_002.png"}
```

### Document Layout Analysis

Detect document layout elements by category.

```jsonl
{"conversations": [{"from": "human", "value": "Locate all the instances that matches the following description: title</c>paragraph</c>figure</c>table."}, {"from": "gpt", "value": "<ref>title</ref><box><50><20><950><80></box><ref>paragraph</ref><box><50><100><950><500></box><ref>figure</ref><box><100><520><900><850></box>"}], "image": "doclayout/page_001.png"}
```

### Multi-Image Input

Use `image_list` and reference images with `<image-1>`, `<image-2>`, etc.

```jsonl
{"conversations": [{"from": "human", "value": "Compare these two images. <image-1> <image-2> Locate all cars in the first image."}, {"from": "gpt", "value": "<ref>car</ref><box><100><200><400><500></box>"}], "image_list": ["img_a.jpg", "img_b.jpg"]}
```

### Pure Text (No Image)

Samples without images are treated as text-only conversation data.

```jsonl
{"conversations": [{"from": "human", "value": "What is object detection?"}, {"from": "gpt", "value": "Object detection is a computer vision task that involves identifying and locating objects within an image by predicting bounding boxes and class labels."}]}
```

---

## Image & Media Requirements

- **Supported image formats:** JPEG, PNG, WebP, BMP, TIFF
- **Supported video formats:** MP4, AVI, MKV, WebM (decoded via `decord`)
- **Resolution:** No fixed requirement; images are dynamically resized by the processor. Higher resolution images produce more visual tokens.
- **Path resolution:** All paths in JSONL are resolved relative to the `root` specified in the recipe JSON.

### Video-Specific Parameters

These are controlled via training arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_frames` | 64 | Max frames to sample from a video |
| `--target_fps` | 2 | Target sampling FPS |
| `--video_total_pixels` | ~10M | Total pixel budget for video frames |

---

## Directory Structure Example

A complete training setup might look like:

```
project/
├── lam_recipe/
│   └── my_recipe.json                  ← recipe file (--meta_path)
├── data/
│   ├── annotations/
│   │   ├── detection_coco.jsonl        ← JSONL annotations
│   │   ├── gui_grounding.jsonl
│   │   ├── ocr_totaltext.jsonl
│   │   └── referring_refcocog.jsonl
│   └── images/
│       ├── coco/
│       │   └── train2017/              ← image files
│       ├── gui_screenshots/
│       ├── ocr_images/
│       └── refcocog/
├── deepspeed_configs/
│   └── zero_stage2_config.json
└── work_dirs/
    └── my_sft/                         ← output (--output_dir)
```

The corresponding recipe JSON:

```json
{
  "detection_coco": {
    "annotation": "data/annotations/detection_coco.jsonl",
    "root": "data/images/",
    "repeat_time": 1.0,
    "data_augment": true
  },
  "gui_grounding": {
    "annotation": "data/annotations/gui_grounding.jsonl",
    "root": "data/images/",
    "repeat_time": 2.0,
    "data_augment": false
  },
  "ocr_totaltext": {
    "annotation": "data/annotations/ocr_totaltext.jsonl",
    "root": "data/images/",
    "repeat_time": 1.0,
    "data_augment": false
  },
  "referring_refcocog": {
    "annotation": "data/annotations/referring_refcocog.jsonl",
    "root": "data/images/",
    "repeat_time": 1.0,
    "data_augment": true
  }
}
```

Then launch training:

```bash
torchrun --nproc_per_node=8 \
  eaglevl/train/lam_finetune_magi_stream.py \
  --model_name_or_path nvidia/LocateAnything-3B \
  --meta_path "./lam_recipe/my_recipe.json" \
  --output_dir work_dirs/my_sft \
  --max_steps 25000 \
  --learning_rate 2e-5 \
  --bf16 True \
  --block_size 6 \
  --attn_implementation magi \
  --max_seq_length 16384 \
  --deepspeed "deepspeed_configs/zero_stage2_config.json"
```

For the full list of training arguments, see [Training Documentation](TRAINING.md).
