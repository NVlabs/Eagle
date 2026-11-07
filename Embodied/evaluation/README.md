# LocateAnything Evaluation Guide

This guide shows how to download evaluation data, unpack images, and run LocateAnything evaluations across datasets and task types.
For VFlash benchmark commands and decoding settings, use the [VFlash Evaluation Guide](../document/vflash/EVALUATION.md).

### 1 Install FastEvaluate (required for COCO/LVIS metrics)

You need to download the `fastevaluate` module from the Rex-Omni repository and place it in the `evaluation` directory:
```bash
# Download fastevaluate from Rex-Omni
svn export https://github.com/IDEA-Research/Rex-Omni/trunk/evaluation/fastevaluate evaluation/fastevaluate
```

Then install it:
```bash
cd evaluation/fastevaluate
pip install -e .

pip install shapely
```

### 2 Download datasets

- Source: `https://huggingface.co/datasets/Mountchicken/Rex-Omni-EvalData`
- For ScreenSpot Pro evaluation, also download: `https://huggingface.co/datasets/likaixin/ScreenSpot-Pro`
- You also need to download `converted_box.jsonl` from `TODO` and place it under the `ScreenSpot-Pro/` directory.
- After downloading, the directory layout should look like `EvalData/` with images packaged as `.tar.gz` files. Example on disk:

```
/.../EvalData
  *.tar.gz               # per-dataset image archives (e.g., coco.tar.gz, hiertext.tar.gz, ...)
  _annotations/          # JSONL annotations (multiple eval types)
  ScreenSpot-Pro/        # ScreenSpot-Pro dataset
    images/              # ScreenSpot-Pro images
    converted_box.jsonl  # Downloaded converted_box.jsonl
  _locate_anything_eval_results # The evaluation results of LocateAnything
```

Unpack the image archives before running:

```bash
cd path/to/EvalData
for f in *.tar.gz; do
  echo "Extracting $f" && tar -xzf "$f"
done
```

### 3 Evaluation
The evaluation is seperated into two categories:
1. COCO/LVIS text-prompt evaluation
2. Other datasets (box/point)
3. ScreenSpot Pro evaluation

#### COCO/LVIS text-prompt evaluation in box format
For text prompt evaluation on COCO and LVIS dataset (box format), run the following script

- For COCO evaluation
  
```bash
bash evaluation/scripts/eval_coco.sh \
    --model_path path/to/LocateAnything \
    --test_jsonl path/to/EvalData/_annotations/box_eval/COCO.jsonl \
    --image_root path/to/EvalData \
    --coco_json path/to/EvalData/coco/instances_val2017.json \
    --output_dir path/to/EvalData/_locate_anything_eval_results/box_eval/COCO
```

- For LVIS evaluation

```bash
bash evaluation/scripts/eval_lvis.sh \
    --model_path /path/to/LocateAnything \
    --test_jsonl path/to/EvalData/_annotations/box_eval/LVIS.jsonl \
    --image_root path/to/EvalData \
    --lvis_json path/to/EvalData/coco/lvis_v1_val_with_filename2.json \
    --output_dir path/to/EvalData/_locate_anything_eval_results/box_eval/LVIS

```

#### Other datasets and task (box/point/gui)

- For text prompt task (output box)

```bash
bash evaluation/scripts/eval_grounding.sh \
    --dataset Dense200 \ # choice in Dense200, DocLayNet, HierText, HumanRef, IC15, M6Doc, RefCOCOg_test, RefCOCOg_val, SROIE, TotalText, VisDrone
    --eval_type box_eval \
    --model_path path/to/LocateAnything \
    --image_root path/to/EvalData \
    --output_base path/to/EvalData/_locate_anything_eval_results/box_eval/
```

- For text prompt task (output point)
```bash
bash evaluation/scripts/eval_grounding.sh \
    --dataset COCO \ # choice in COCO, Dense200, HumanRef, LVIS, RefCOCOg_test, RefCOCOg_val, VisDrone
    --eval_type point_eval \
    --model_path path/to/LocateAnything \
    --image_root path/to/EvalData \
    --output_base path/to/EvalData/_locate_anything_eval_results/point_eval/
```

- For ScreenSpot Pro evaluation (box format)
```bash
bash evaluation/scripts/eval_sspro.sh \
    --model_path path/to/LocateAnything \
    --test_jsonl path/to/EvalData/ScreenSpot-Pro/converted_box.jsonl \
    --image_root path/to/EvalData/ScreenSpot-Pro/images \
    --output_dir path/to/EvalData/_locate_anything_eval_results/box_eval/sspro
```

## Acknowledgement

We would like to thank the authors of [Rex-Omni](https://github.com/IDEA-Research/Rex-Omni) for their excellent evaluation framework, which served as a great reference for our evaluation code.
