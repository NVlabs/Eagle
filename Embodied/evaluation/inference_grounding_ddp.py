# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Multi-GPU DDP inference script for LocateAnything grounding benchmarks.
Supports multi-node multi-GPU distributed inference.
"""
import argparse
import json
import os
import re
import socket
import datetime

import torch
import torch.distributed as dist
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from inference_compat import (
    apply_chat_template,
    build_generate_kwargs,
    decode_generation_output,
    prepare_generation_inputs,
    process_vision_info,
)

# Set longer NCCL timeout (2 hours)
os.environ["NCCL_TIMEOUT"] = "7200"

OCR_DATASETS = {"TotalText", "SROIE", "IC15", "HierText"}
POINTING_PER_CATEGORY_DATASETS = {"COCO", "LVIS_fullval", "VisDrone"}


def normalize_categories(categories):
    """
    Normalize categories field to List[str].
    - Supports: ["person", "car"]
    - Supports: "person" / "person, car, truck"
    - Supports: ["person, car, truck"] (some preprocessing may pack multiple categories into one string)
    """
    if categories is None:
        return []
    if isinstance(categories, str):
        return [c.strip() for c in categories.split(",") if c.strip()]
    if isinstance(categories, (list, tuple)):
        out = []
        for item in categories:
            if item is None:
                continue
            if isinstance(item, str):
                parts = [c.strip() for c in item.split(",") if c.strip()]
                out.extend(parts)
            else:
                out.append(str(item))
        # Deduplicate while preserving order
        seen = set()
        dedup = []
        for c in out:
            if c not in seen:
                seen.add(c)
                dedup.append(c)
        return dedup
    return [str(categories)]


def scale_predictions(preds_by_cat, scale_factor):
    """
    Map resized prediction coordinates back to original image size.
    Supports both points (len==2) and boxes (len==4).
    """
    if scale_factor is None or scale_factor == 1.0:
        return preds_by_cat
    scaled = {}
    for cat, items in preds_by_cat.items():
        scaled_items = []
        for item in items:
            if not isinstance(item, (list, tuple)):
                continue
            if len(item) == 2:
                scaled_items.append([item[0] / scale_factor, item[1] / scale_factor])
            elif len(item) == 4:
                scaled_items.append(
                    [item[0] / scale_factor, item[1] / scale_factor, item[2] / scale_factor, item[3] / scale_factor]
                )
        scaled[cat] = scaled_items
    return scaled


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="./work_dirs/lam_baseline",
    )
    parser.add_argument(
        "--test_jsonl_path",
        type=str,
        default="path/to/EvalData/annotations/box_eval/COCO.jsonl",
        help="Path to the test JSONL file containing benchmark data",
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        default="path/to/EvalData",
        help="Root directory to prepend to image paths. If empty, use image_path as is.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="path/to/EvalData/eval_results/box_eval/COCO/eval.jsonl",
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--short_side_size",
        type=int,
        default=None,
        help="Resize image short side to this value before inference. Coordinates will be mapped back to original size.",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="box_eval",
        choices=["box_eval", "point_eval"],
        help="Evaluation type: box_eval or point_eval",
    )
    parser.add_argument(
        "--generation_mode",
        type=str,
        default="hybrid",
        choices=["fast", "slow", "hybrid"],
        help="Generation mode: 'fast' (MTP only), 'slow' (AR only), 'hybrid' (MTP + AR fallback).",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="Output path for visualization video (mp4 format). If None, no video will be generated.",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=2.0,
        help="FPS for output video",
    )
    
    # DDP parameters
    parser.add_argument("--world_size", type=int, default=1, help="Total number of GPUs")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="Master address")
    parser.add_argument("--master_port", type=str, default="29500", help="Master port")
    parser.add_argument("--use_tcp", action="store_true", help="Use TCP for communication")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank (set by torchrun)")
    
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        print("Not using distributed mode")
        return 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    
    # Initialize process group with longer timeout (2 hours)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(hours=2),
    )
    
    dist.barrier()
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is the main process."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank():
    """Get the rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get the total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


class LocateAnythingWorker:
    def __init__(self, model_path, device='cuda', generation_mode: str = 'hybrid'):
        self.model_id = model_path
        self.device = device
        self.generation_mode = generation_mode
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        if hasattr(self.processor, 'tokenizer'):
            try:
                self.processor.tokenizer.padding_side = "left"
            except Exception:
                pass
        self.model = self.model.to(device)
        self.model.eval()

    def build_messages(self, image, categories, question_override=None):
        """
        Build messages. categories is a list like ["person", "car"].
        """
        if question_override is not None:
            user_text = question_override + ", ".join(categories) + "."
        else:
            category_set_str = "</c>".join(categories)
            user_text = f"Locate all the instances that matches the following description: {category_set_str}."
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            }
        ], user_text

    @torch.inference_mode()
    def generate(self, image, categories, max_new_tokens=4096, question_override=None):
        """
        Generate detection results.
        Returns: (output_text, question)
        """
        messages, question = self.build_messages(image, categories, question_override=question_override)
        text_list = [apply_chat_template(self.processor, messages)]
        image_inputs, video_inputs = process_vision_info(
            self.processor,
            messages,
        )
        processor_inputs = self.processor(
            text=text_list,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )

        prepared_inputs = prepare_generation_inputs(processor_inputs, self.device)
        generate_kwargs = build_generate_kwargs(
            prepared_inputs,
            self.processor,
            generation_mode=self.generation_mode,
            max_new_tokens=max_new_tokens,
            include_eos_token=True,
        )

        raw_output = self.model.generate(**generate_kwargs)
        output_text = decode_generation_output(
            raw_output,
            prepared_inputs["input_ids"],
            self.processor,
        )

        return output_text, question

    @torch.inference_mode()
    def generate_single_category_pointing(self, image, category: str, max_new_tokens=4096):
        """
        Pointing task: query one category at a time.
        """
        category = str(category)
        # Construct text directly to avoid combining multiple categories
        question_override = "Point to: "
        outputs, _ = self.generate(
            image,
            categories=[category],
            max_new_tokens=max_new_tokens,
            question_override=question_override,
        )
        question = f"{question_override}{category}."
        return outputs, question


class LocalizationDataset(Dataset):
    """
    Dataset class for distributed inference.
    """
    def __init__(self, test_data, image_root_dir):
        self.test_data = test_data
        self.image_root_dir = image_root_dir

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        entry = self.test_data[idx]
        image_path = entry["image_path"]
        categories = entry["categories"]
        gt = entry["gt"]
        dataset_name = entry["dataset_name"]
        task_name = entry["task_name"]
        question = entry.get("question", "")

        # Construct full image path
        if self.image_root_dir:
            full_image_path = os.path.join(self.image_root_dir, image_path)
        else:
            full_image_path = image_path

        return {
            "image_path": image_path,
            "full_image_path": full_image_path,
            "categories": categories,
            "gt": gt,
            "dataset_name": dataset_name,
            "task_name": task_name,
            "question": question,
            "idx": idx,
        }


def resize_image_short_side(image, short_side_size):
    """
    Resize image short side to specified value, preserving aspect ratio.

    Args:
        image: PIL.Image object
        short_side_size: target short side size

    Returns:
        resized_image: resized image
        scale_factor: scale factor (resized / original)
    """
    w, h = image.size
    if w <= h:
        new_w = short_side_size
        scale_factor = new_w / w
        new_h = int(h * scale_factor)
    else:
        new_h = short_side_size
        scale_factor = new_h / h
        new_w = int(w * scale_factor)
    
    resized_image = image.resize((new_w, new_h), Image.BILINEAR)
    return resized_image, scale_factor


def parse_bbox_with_labels(text):
    """
    Parse <ref>category</ref><box>...</box> format.
    Supports two formats:
    1. Point coordinates: <box><x><y></box> (2 values)
    2. Box coordinates: <box><x1><y1><x2><y2></box> (4 values)
    Returns [(category, coords, is_point), ...]
    where is_point=True indicates point coordinates, False indicates box coordinates.
    """
    results = []
    # Match <ref>category</ref> followed by all <box>...</box>
    ref_pattern = r'<ref>([^<]+)</ref>((?:<box>.*?</box>)+)'
    
    # Match point coordinate format: <box><x><y></box>
    point_pattern = (
        r'<box>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*'
        r'<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>'
    )
    
    # Match box coordinate format: <box><x1><y1><x2><y2></box>
    box_pattern = (
        r'<box>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*'
        r'<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*'
        r'<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*'
        r'<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>'
    )

    ref_matches = re.findall(ref_pattern, text)
    for category, boxes_str in ref_matches:
        # Try matching point coordinates (2 values) first
        point_matches = re.findall(point_pattern, boxes_str)
        for match in point_matches:
            try:
                x, y = map(float, match)
                if 0 <= x <= 10000 and 0 <= y <= 10000:
                    results.append((category, [x, y], True))  # True indicates point coordinates
            except Exception:
                continue
        
        # Then try matching box coordinates (4 values)
        box_matches = re.findall(box_pattern, boxes_str)
        for match in box_matches:
            try:
                x1, y1, x2, y2 = map(float, match)
                if 0 <= x1 <= 10000 and 0 <= y1 <= 10000 and 0 <= x2 <= 10000 and 0 <= y2 <= 10000:
                    results.append((category, [x1, y1, x2, y2], False))  # False indicates box coordinates
            except Exception:
                continue
    return results


def convert_normalized_point_to_absolute(nor_point, img_size):
    """
    Convert normalized point coordinates (0-1000) to absolute coordinates.
    Args: nor_point = [x, y] (0-1000 range)
          img_size = (w, h)
    Returns: [x, y] (absolute coordinates)
    """
    w, h = img_size
    x, y = nor_point
    x = x * w / 1000
    y = y * h / 1000
    # Clamp coordinates to valid range
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    return [x, y]


def convert_normalized_bbox_to_absolute(nor_bbox, img_size):
    """
    Convert normalized coordinates (0-1000) to absolute coordinates.
    Args: nor_bbox = [x1, y1, x2, y2] (0-1000 range)
          img_size = (w, h)
    Returns: [x1, y1, x2, y2] (absolute coordinates, xyxy format)
    """
    w, h = img_size
    x1, y1, x2, y2 = nor_bbox
    x1 = x1 * w / 1000
    y1 = y1 * h / 1000
    x2 = x2 * w / 1000
    y2 = y2 * h / 1000
    # Clamp coordinates to valid range
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    # Ensure x1 < x2, y1 < y2
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1, x2)
    y_max = max(y1, y2)
    return [x_min, y_min, x_max, y_max]


def get_color_for_label(label):
    """
    Assign a fixed color based on category name.
    """
    colors = [
        (220, 38, 38),   # red
        (22, 163, 74),   # green
        (37, 99, 235),   # blue
        (217, 119, 6),   # orange
        (147, 51, 234),  # purple
        (8, 145, 178)    # cyan
    ]
    idx = sum(ord(c) for c in label.lower())
    return colors[idx % len(colors)]


def draw_detections_on_image(image, extracted_predictions, draw_label=True):
    """
    Draw detection boxes on image.

    Args:
        image: PIL.Image object
        extracted_predictions: dict, format {category: [[x1,y1,x2,y2], ...], ...}
        draw_label: bool, Whether to draw labels.

    Returns:
        PIL.Image object with detection boxes drawn.
    """
    img_draw = image.convert("RGBA")
    overlay = Image.new("RGBA", img_draw.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    w_img, h_img = image.size
    
    for category, bboxes in extracted_predictions.items():
        color = get_color_for_label(category)
        fill_color = color + (65,)  # Semi-transparent fill
        
        for bbox in bboxes:
            if len(bbox) == 4:
                # Box coordinates
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
                
                # Draw box
                draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=color, width=4)
                
                # Draw text label
                if draw_label and category:
                    t_box = draw.textbbox((0, 0), category, font=font)
                    draw.rectangle(
                        [
                            x1,
                            y1 - (t_box[3] - t_box[1]) - 10,
                            x1 + (t_box[2] - t_box[0]) + 15,
                            y1,
                        ],
                        fill=color,
                    )
                    draw.text(
                        (x1 + 7, y1 - (t_box[3] - t_box[1]) - 7),
                        category,
                        fill="white",
                        font=font,
                    )
            elif len(bbox) == 2:
                # Point coordinates
                x, y = bbox
                x, y = max(0, min(w_img-1, x)), max(0, min(h_img-1, y))
                radius = 5
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline=color, width=2)
    
    combined = Image.alpha_composite(img_draw, overlay).convert("RGB")
    return combined


def parse_prediction(text, w, h):
    """
    Parse model output and extract points/boxes grouped by category.

    Input format examples:
    Point: <ref>person</ref><box><100><200></box>
    Box: <ref>car</ref><box><150><250><350><450></box>

    Returns:
    {
        'person': [[x, y], ...],
        'car': [[x1,y1,x2,y2], ...],
        ...
    }
    Note: points have length 2, boxes have length 4.
    """
    result = {}

    # Parse <ref>category</ref><box>...</box> format
    parsed_items = parse_bbox_with_labels(text)

    for category, coords, is_point in parsed_items:
        if category not in result:
            result[category] = []
        
        if is_point:
            # Point: convert normalized to absolute coordinates
            abs_point = convert_normalized_point_to_absolute(coords, (w, h))
            result[category].append(abs_point)
        else:
            # Box: convert normalized to absolute coordinates
            abs_bbox = convert_normalized_bbox_to_absolute(coords, (w, h))
            result[category].append(abs_bbox)

    return result


def load_test_data(test_jsonl_path):
    """
    Load test data from JSONL file.

    Returns:
    List of test entries, each containing:
    {
        'image_name': str,
        'question': str,
        'gt': dict,
        'categories': list,
        'task_name': str,
        'dataset_name': str,
        'point': [x, y] (optional, for point-based tasks)
    }
    """
    test_data = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    test_data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")

    return test_data


def gather_predictions(local_predictions, world_size):
    """
    Gather predictions from all processes to the main process.
    """
    if world_size == 1:
        return local_predictions
    
    # Serialize predictions to JSON string
    local_data = json.dumps(local_predictions, ensure_ascii=False)
    local_data_bytes = local_data.encode('utf-8')
    
    # Get local data size
    local_size = torch.tensor([len(local_data_bytes)], dtype=torch.long, device='cuda')
    
    # Gather data sizes from all processes
    size_list = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    
    # Get maximum data size
    max_size = max([s.item() for s in size_list])
    
    # Pad local data to maximum size
    padded_data = local_data_bytes + b'\x00' * (max_size - len(local_data_bytes))
    local_tensor = torch.ByteTensor(list(padded_data)).cuda()
    
    # Gather all data
    gathered_tensors = [torch.zeros(max_size, dtype=torch.uint8, device='cuda') for _ in range(world_size)]
    dist.all_gather(gathered_tensors, local_tensor)
    
    # Only parse results on main process
    if is_main_process():
        all_predictions = []
        for i, (tensor, size) in enumerate(zip(gathered_tensors, size_list)):
            data_bytes = bytes(tensor[:size.item()].cpu().tolist())
            data_str = data_bytes.decode('utf-8')
            predictions = json.loads(data_str)
            all_predictions.extend(predictions)
        return all_predictions
    
    return []


def main():
    args = get_args()
    
    # Set up distributed environment
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if local_rank >= 0 else "cuda"
    
    if is_main_process():
        print(f"=== DDP Inference Configuration ===")
        print(f"World Size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local Rank: {local_rank}")
        print(f"Device: {device}")
        print(f"Model Path: {args.model_path}")
        print(f"Test JSONL Path: {args.test_jsonl_path}")
        print(f"Save Path: {args.save_path}")
        print(f"=====================")
    
    # Create save directory
    if is_main_process():
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    
    # Wait for main process to create directory
    if dist.is_initialized():
        dist.barrier()
    
    # Initialize model
    if is_main_process():
        print(f"Loading model from: {args.model_path}")
    if is_main_process():
        print(f"Generation Mode: {args.generation_mode}")
    worker = LocateAnythingWorker(args.model_path, device=device, generation_mode=args.generation_mode)
    
    # Load test data
    if is_main_process():
        print(f"Loading test data from: {args.test_jsonl_path}")
    test_data = load_test_data(args.test_jsonl_path)
    
    if is_main_process():
        print(f"Loaded {len(test_data)} test entries")
    
    # Create dataset and distributed sampler
    dataset = LocalizationDataset(test_data, args.image_root_dir)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    
    # Use DataLoader with batch_size=1 for single-sample inference
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        collate_fn=lambda x: x[0],
    )
    
    if is_main_process():
        print(f"Rank {rank}: Processing {len(dataloader)} samples")
    
    local_predictions = []
    
    # Progress bar (only on main process)
    iterator = tqdm(dataloader, desc=f"Rank {rank} processing", disable=not is_main_process())
    
    for sample in iterator:
        image_path = sample["image_path"]
        full_image_path = sample["full_image_path"]
        categories = normalize_categories(sample["categories"])
        gt = sample["gt"]
        dataset_name = sample["dataset_name"]
        task_name = sample["task_name"]
        
        if not os.path.exists(full_image_path):
            print(f"[Rank {rank}] Warning: Image not found: {full_image_path}")
            continue
        
        try:
            image = Image.open(full_image_path).convert("RGB")
            original_w, original_h = image.size
        except Exception as e:
            print(f"[Rank {rank}] Error loading image {full_image_path}: {e}")
            continue
        
        # Resize image if short_side_size is set
        scale_factor = 1.0
        if args.short_side_size is not None:
            image, scale_factor = resize_image_short_side(image, args.short_side_size)
            resized_w, resized_h = image.size
            if is_main_process():
                print(f"[Resize] Original: {original_w}x{original_h} -> Resized: {resized_w}x{resized_h} (scale: {scale_factor:.4f})")
        
        # Build dataset-specific prompt
        question_override = None
        per_category_raw = None

        # Per-category pointing for COCO/LVIS/VisDrone
        if args.eval_type == "point_eval" and dataset_name in POINTING_PER_CATEGORY_DATASETS:
            extracted_predictions = {}
            per_category_raw = {}
            questions = []
            merged_outputs = []
            for cat in categories:
                out_text, q = worker.generate_single_category_pointing(
                    image, cat, max_new_tokens=args.max_new_tokens
                )
                per_category_raw[cat] = out_text
                questions.append(q)
                merged_outputs.append(out_text if isinstance(out_text, str) else str(out_text))

                # Parse: force results onto the queried category regardless of <ref> output
                parsed = parse_prediction(out_text, image.size[0], image.size[1])
                # Keep only point coordinates (len==2); ignore boxes
                points = []
                for _, coords_list in parsed.items():
                    for coords in coords_list:
                        if isinstance(coords, (list, tuple)) and len(coords) == 2:
                            points.append(coords)
                if cat not in extracted_predictions:
                    extracted_predictions[cat] = []
                extracted_predictions[cat].extend(points)

            output = "\n".join(merged_outputs)
            question = " | ".join(questions)
        else:
            if args.eval_type == "point_eval":
                question_override = "Point to: "
            elif dataset_name == "HumanRef":
                question_override = "Locate all the instances that matches the following description: "
            elif dataset_name in ["RefCOCOg_test", "RefCOCOg_val"]:
                question_override = "Locate a single instance that matches the following description: "
            elif dataset_name in OCR_DATASETS:
                # OCR datasets: use OCR-specific prompt
                question_override = "Detect all the text in box format."

            output, question = worker.generate(
                image,
                categories,
                max_new_tokens=args.max_new_tokens,
                question_override=question_override,
            )
        
        if is_main_process():
            print(f"[{dataset_name}]")
            print(f"Question: {question}")
            print(f"Output: {output[:200]}...")
        
        try:
            # Skip re-parsing if per-category pointing was already done
            if not (args.eval_type == "point_eval" and dataset_name in POINTING_PER_CATEGORY_DATASETS):
                # Parse with resized dimensions, then convert back
                resized_w, resized_h = image.size
                extracted_predictions = parse_prediction(
                    output,
                    resized_w,
                    resized_h,
                )

            # Convert coordinates back to original size if resized
            if args.short_side_size is not None and scale_factor != 1.0:
                extracted_predictions = scale_predictions(extracted_predictions, scale_factor)

            prediction = {
                "image_path": image_path,
                "extracted_predictions": extracted_predictions,
                "gt": gt,
                "question": question,
                "dataset_name": dataset_name,
                "raw_response": output,
                "task_name": task_name,
            }
            if per_category_raw is not None:
                prediction["per_category_outputs"] = per_category_raw

        except Exception as e:
            print(f"[Rank {rank}] Parse failed, error is {e}")
            prediction = {
                "image_path": image_path,
                "extracted_predictions": {},
                "gt": gt,
                "question": question,
                "dataset_name": dataset_name,
                "raw_response": output,
                "task_name": task_name,
            }
            if per_category_raw is not None:
                prediction["per_category_outputs"] = per_category_raw
        
        if "gt_mask" in gt:
            prediction["gt"] = gt["gt_mask"]
        
        local_predictions.append(prediction)
    
    print(f"[Rank {rank}] Finished processing {len(local_predictions)} samples")
    
    # Each rank saves its own results independently to avoid all_gather timeout
    save_dir = os.path.dirname(args.save_path)
    base_name = os.path.basename(args.save_path)
    name_without_ext = os.path.splitext(base_name)[0]
    ext = os.path.splitext(base_name)[1] or ".jsonl"
    
    # Save to per-rank file
    rank_save_path = os.path.join(save_dir, f"{name_without_ext}_rank{rank}{ext}")
    with open(rank_save_path, "w") as f:
        for prediction in local_predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(f"[Rank {rank}] Saved {len(local_predictions)} predictions to: {rank_save_path}")
    
    # Synchronize all processes to ensure all ranks have finished saving
    if dist.is_initialized():
        dist.barrier()
    
    # Main process merges all results
    if is_main_process():
        print(f"\n🔄 Merging predictions from all ranks...")
        all_predictions = []
        for r in range(world_size):
            r_save_path = os.path.join(save_dir, f"{name_without_ext}_rank{r}{ext}")
            if os.path.exists(r_save_path):
                with open(r_save_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_predictions.append(json.loads(line))
                # Optional: remove temporary files
                # os.remove(r_save_path)
        
        print(f"💾 Saving merged predictions to: {args.save_path}")
        with open(args.save_path, "w") as f:
            for prediction in all_predictions:
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        print(f"✅ Saved {len(all_predictions)} predictions!")
        
        # Clean up temporary files
        for r in range(world_size):
            r_save_path = os.path.join(save_dir, f"{name_without_ext}_rank{r}{ext}")
            if os.path.exists(r_save_path):
                os.remove(r_save_path)
                print(f"🗑️ Cleaned up temporary file: {r_save_path}")
        
        # Generate visualization video (if enabled)
        if args.output_video and all_predictions:
            print(f"\n🎬 Generating visualization video...")
            video_dir = os.path.dirname(args.output_video)
            if video_dir and not os.path.exists(video_dir):
                os.makedirs(video_dir, exist_ok=True)
            
            # Load images from predictions for visualization
            video_frames = []
            for pred in tqdm(all_predictions, desc="Loading images for video"):
                image_path = pred["image_path"]
                extracted_predictions = pred.get("extracted_predictions", {})
                
                # Build full image path
                if args.image_root_dir:
                    full_image_path = os.path.join(args.image_root_dir, image_path)
                else:
                    full_image_path = image_path
                
                if os.path.exists(full_image_path):
                    try:
                        image = Image.open(full_image_path).convert("RGB")
                        # Using original image for visualization
                        video_frames.append({
                            "image": image,
                            "predictions": extracted_predictions
                        })
                    except Exception as e:
                        print(f"Warning: Failed to load image {full_image_path}: {e}")
                        continue
            
            if video_frames:
                # Get first frame dimensions
                first_image = video_frames[0]["image"]
                video_w, video_h = first_image.size
                
                # Create video writer (mp4v codec)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(args.output_video, fourcc, args.video_fps, (video_w, video_h))
                
                if not out.isOpened():
                    print(f"⚠️ Failed to open video writer, trying alternative codec...")
                    # Try H.264 codec as fallback
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    out = cv2.VideoWriter(args.output_video, fourcc, args.video_fps, (video_w, video_h))
                
                if out.isOpened():
                    for frame_data in tqdm(video_frames, desc="Writing video frames"):
                        vis_image = draw_detections_on_image(
                            frame_data["image"],
                            frame_data["predictions"],
                            draw_label=True
                        )
                        # Convert to OpenCV format (BGR)
                        frame_bgr = cv2.cvtColor(np.array(vis_image), cv2.COLOR_RGB2BGR)
                        out.write(frame_bgr)
                    
                    out.release()
                    print(f"✅ Video saved to: {args.output_video}")
                else:
                    print(f"⚠️ Failed to create video writer")
            else:
                print(f"⚠️ No valid frames to write to video")
    
    # Clean up distributed environment
    cleanup_distributed()


if __name__ == "__main__":
    main()
