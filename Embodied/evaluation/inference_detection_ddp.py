# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Multi-GPU DDP inference script for LocateAnything detection (COCO/LVIS).
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
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from inference_compat import (
    BenchmarkInferenceWorker,
    apply_chat_template,
    build_generate_kwargs,
    decode_generation_output,
    has_physical_atoms,
    prepare_generation_inputs,
    process_vision_info,
)

# Set longer NCCL timeout (2 hours)
os.environ["NCCL_TIMEOUT"] = "7200"


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
        "--per_category_training_prompt",
        action="store_true",
        help="Query each positive category separately with the visual-primitive "
        "training prompt, then merge the predictions per image.",
    )
    parser.add_argument(
        "--multiturn_training_prompt",
        action="store_true",
        help="Query all positive categories in one conversation with the image only "
        "in the first turn, like the multi-turn training records.",
    )
    parser.add_argument(
        "--vflash_box_prompt",
        action="store_true",
        help="Add the box and coordinate paragraphs to the training prompt.",
    )
    parser.add_argument(
        "--short_side_size",
        type=int,
        default=None,
        help="Resize image short side to this value before inference. Coordinates will be mapped back to original size.",
    )
    parser.add_argument(
        "--generation_mode",
        type=str,
        default="hybrid",
        choices=["fast", "slow", "hybrid", "ar", "vflash"],
        help="Generation mode: 'fast' (MTP only), 'slow' (AR only), 'hybrid' (MTP + AR fallback).",
    )
    parser.add_argument(
        "--draft_model_path",
        type=str,
        default=None,
        help="VFlash drafter of an Eagle Embodied target, used with --generation_mode vflash "
        "(any other mode decodes Eagle Embodied autoregressively).",
    )

    # DDP parameters
    parser.add_argument("--world_size", type=int, default=1, help="Total number of GPUs")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="Master address")
    parser.add_argument("--master_port", type=str, default="29500", help="Master port")
    parser.add_argument("--use_tcp", action="store_true", help="Use TCP for communication")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank (set by torchrun)")
    
    args = parser.parse_args()
    if args.vflash_box_prompt and not (args.per_category_training_prompt or args.multiturn_training_prompt):
        parser.error("--vflash_box_prompt requires a per-category or multiturn training prompt")
    return args


def visual_primitive_question(category, vflash_box_prompt=False):
    question = f"Locate every instance of {category}. Return only the requested visual-primitive format."
    if vflash_box_prompt:
        question += (
            "\n\nRepresent each bounding box as <box><x1><y1><x2><y2></box>, "
            "where (x1,y1) is the top-left and (x2,y2) is the bottom-right corner."
            "\n\nCoordinates are 2D image coordinates, with x and y normalized to "
            "the range [0, 1000] along the image width and height, respectively. "
            "The origin is at the top-left corner, with x increasing to the right "
            "and y increasing downward. /no_think"
        )
    return question


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

    def build_messages(self, image, categories):
        """Build messages. categories is a list like ["person", "car"]."""
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
        ]

    @torch.inference_mode()
    def generate(self, image, categories, max_new_tokens=4096):
        """Generate detection results.
        Returns: (output_text, question)
        """
        messages = self.build_messages(image, categories)
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
        )
        raw_output = self.model.generate(**generate_kwargs)
        output_text = decode_generation_output(
            raw_output,
            prepared_inputs["input_ids"],
            self.processor,
        )

        category_set_str = "</c>".join(categories)
        question = f"Locate all the instances that matches the following description: {category_set_str}."

        return output_text, question


class LocalizationDataset(Dataset):
    """Dataset class for distributed inference."""
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
            "idx": idx,
        }


def resize_image_short_side(image, short_side_size):
    """Resize image short side to specified value while maintaining aspect ratio.

    Args:
        image: PIL.Image object.
        short_side_size: Target short side size.

    Returns:
        resized_image: Resized image.
        scale_factor: Scale factor (resized / original).
    """
    w, h = image.size
    if w <= h:
        # Width is the short side
        new_w = short_side_size
        scale_factor = new_w / w
        new_h = int(h * scale_factor)
    else:
        # Height is the short side
        new_h = short_side_size
        scale_factor = new_h / h
        new_w = int(w * scale_factor)
    
    resized_image = image.resize((new_w, new_h), Image.BILINEAR)
    return resized_image, scale_factor


def parse_bbox_with_labels(text, objects=False):
    """Parse <ref>category</ref><box><x1><y1><x2><y2></box> format.
    Returns [(category, [x1, y1, x2, y2]), ...]
    ``objects`` also accepts the boxes wrapped in <objects>...</objects>.
    """
    results = []
    # Match <ref>category</ref> followed by all <box>...</box>
    ref_pattern = r'<ref>([^<]+)</ref>((?:<box>.*?</box>)+)'
    if objects:
        ref_pattern = (
            r'<ref>([^<]+)</ref>\s*(?:<objects>)?\s*'
            r'((?:<box>.*?</box>)+)\s*(?:</objects>)?'
        )
    box_pattern = (
        r'<box>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*'
        r'<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*'
        r'<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*'
        r'<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>'
    )

    ref_matches = re.findall(ref_pattern, text)
    for category, boxes_str in ref_matches:
        box_matches = re.findall(box_pattern, boxes_str)
        for match in box_matches:
            try:
                x1, y1, x2, y2 = map(float, match)
                if 0 <= x1 <= 10000 and 0 <= y1 <= 10000 and 0 <= x2 <= 10000 and 0 <= y2 <= 10000:
                    results.append((category, [x1, y1, x2, y2]))
            except Exception:
                continue
    return results


def convert_normalized_bbox_to_absolute(nor_bbox, img_size):
    """Convert normalized coordinates (0-1000) to absolute coordinates.
    Args: nor_bbox: [x1, y1, x2, y2] in range 0-1000.
          img_size: (w, h) tuple.
    Returns: [x1, y1, x2, y2] in absolute coordinates (xyxy format).
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


def parse_prediction(text, w, h, objects=False):
    """Parse model output text and extract bounding boxes grouped by category.

    Input format example:
    <ref>person</ref><box><100><200><300><400></box><box><500><600><700><800></box>
    <ref>car</ref><box><150><250><350><450></box>

    Returns:
    {
        'person': [[x1,y1,x2,y2], [x1,y1,x2,y2], ...],
        'car': [[x1,y1,x2,y2], ...],
        ...
    }
    """
    result = {}

    # Parse <ref>category</ref><box>...</box> format
    parsed_boxes = parse_bbox_with_labels(text, objects=objects)

    for category, nor_bbox in parsed_boxes:
        # Convert normalized coordinates to absolute coordinates
        abs_bbox = convert_normalized_bbox_to_absolute(nor_bbox, (w, h))

        if category not in result:
            result[category] = []
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
    """Gather predictions from all processes to the main process."""
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
        print(f"Generation Mode: {args.generation_mode}")
    if has_physical_atoms(args.model_path):
        # VFlash targets (Eagle Embodied with physical atoms) run through the
        # shared adapter; --generation_mode slow gives their AR baseline.
        bbox_hybrid = None
        if args.generation_mode == "vflash":
            from eaglevl.model.vflash.bbox_hybrid import BBoxHybridConfig

            bbox_hybrid = BBoxHybridConfig()
        worker = BenchmarkInferenceWorker(
            args.model_path,
            device=device,
            generation_mode=args.generation_mode,
            draft_model_path=args.draft_model_path,
            bbox_hybrid=bbox_hybrid,
        )
    else:
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
        categories = sample["categories"]
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
        
        # Run inference
        category_predictions = None
        if args.multiturn_training_prompt or args.per_category_training_prompt:
            # The training-prompt modes ask for one category at a time and
            # attribute each answer's boxes to the queried category.
            questions = [
                visual_primitive_question(category, args.vflash_box_prompt)
                for category in categories
            ]
            if args.multiturn_training_prompt:
                raw_outputs = worker.generate_multiturn(
                    image,
                    questions,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                raw_outputs = [
                    worker.generate(image, question_i, max_new_tokens=args.max_new_tokens)
                    for question_i in questions
                ]
            category_predictions = {}
            for category, output_i in zip(categories, raw_outputs):
                parsed_i = parse_prediction(output_i, image.size[0], image.size[1], objects=True)
                category_predictions[str(category)] = [
                    coords
                    for coords_list in parsed_i.values()
                    for coords in coords_list
                    if isinstance(coords, (list, tuple)) and len(coords) == 4
                ]
            output = "\n".join(raw_outputs)
            question = " | ".join(questions)
        else:
            output, question = worker.generate(
                image,
                categories,
                max_new_tokens=args.max_new_tokens,
            )
        
        if is_main_process():
            print(f"[{dataset_name}]")
            print(f"Question: {question}")
            print(f"Output: {output[:200]}...")
        
        try:
            # Parse using resized dimensions, then convert back to original size
            resized_w, resized_h = image.size
            extracted_predictions = category_predictions if category_predictions is not None else parse_prediction(
                    output,
                    resized_w,
                    resized_h,
                )
            
            # Convert coordinates back to original image size if resized
            if args.short_side_size is not None and scale_factor != 1.0:
                for category in extracted_predictions:
                    for i, bbox in enumerate(extracted_predictions[category]):
                        extracted_predictions[category][i] = [
                            bbox[0] / scale_factor,
                            bbox[1] / scale_factor,
                            bbox[2] / scale_factor,
                            bbox[3] / scale_factor,
                        ]
            
            prediction = {
                "image_path": image_path,
                "extracted_predictions": extracted_predictions,
                "gt": gt,
                "question": question,
                "dataset_name": dataset_name,
                "raw_response": output,
                "task_name": task_name,
            }
        
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
    
    # Synchronize all processes
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
    
    # Clean up distributed environment
    cleanup_distributed()


if __name__ == "__main__":
    main()