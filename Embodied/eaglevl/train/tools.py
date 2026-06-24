# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import logging
import os
import os.path as osp
import re
import shutil
import time
import copy
from typing import List

import torch
import torch.distributed as dist
from pynvml import (
    NVML_TEMPERATURE_GPU,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetTemperature,
    nvmlInit,
    nvmlShutdown,
)
from PIL import Image
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from .fastseek.draw_marker import DRAW_FUNCTIONS

logger = logging.getLogger(__name__)

_DETECTION_CATEGORY_RE = re.compile(
    r"(Detect all the objects in the image that belong to the category set:\s*)(?P<category>.+?)(?P<suffix>\.)"
)
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")

def get_last_checkpoint_guard(folder):
    while True:
        last_checkpoint = get_last_checkpoint(folder)
        if last_checkpoint is None:
            break
        
        world_size = dist.get_world_size()
        if len(glob.glob(os.path.join(last_checkpoint, "*.pth"))) != world_size:
            # incomplete xxx.pth
            shutil.rmtree(last_checkpoint)
        else:
            break

    return last_checkpoint

class SaveCheckpointCallback(TrainerCallback):
    def __init__(self, initial_interval_hours, save_interval_minutes):
        super().__init__()
        self.initial_interval_seconds = initial_interval_hours * 3600 - 15 * 60
        self.save_interval_seconds = save_interval_minutes * 60
        self.start_time = None
        self.first_save_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is None:
            return control

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Check if the initial interval has passed
        if self.first_save_time is None and elapsed_time >= self.initial_interval_seconds:
            self.first_save_time = current_time
            control.should_save = True
        # Check if the subsequent save interval has passed
        elif self.first_save_time is not None and (current_time - self.first_save_time) >= self.save_interval_seconds:
            self.first_save_time = current_time
            control.should_save = True

        return control


class SkipIterCallback(TrainerCallback):
    """Callback that skips training at specified iteration steps and logs loss as 1."""
    
    def __init__(self, skip_iters: List[int]):
        """
        Args:
            skip_iters: List of iteration steps to skip during training
        """
        super().__init__()
        self.skip_iters = set(skip_iters)
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Check at the beginning of each training step whether to skip it."""
        if state.global_step in self.skip_iters:
            # Skip training for this step
            control.should_training_stop = False
            control.should_epoch_stop = False
            control.should_save = False
            control.should_evaluate = False
            control.should_log = True
            
            # Manually log loss as 1
            if hasattr(state, 'log_history'):
                fake_log = {
                    'loss': 1.0,
                    'learning_rate': args.learning_rate,
                    'epoch': state.epoch,
                    'step': state.global_step
                }
                state.log_history.append(fake_log)
                
            print(f"Skipped training step {state.global_step}, logging loss as 1.0")
            
        return control
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """When logging, ensure loss is 1 if this is a skipped step."""
        if state.global_step in self.skip_iters and logs is not None:
            logs['loss'] = 1.0
            print(f"Step {state.global_step} skipped, loss set to 1.0")
        return control


class MemoryLoggerCallback(TrainerCallback):
    def __init__(self):
        nvmlInit()  
        self.rank = dist.get_rank() if torch.distributed.is_initialized() else 0
        self.device_id = torch.cuda.current_device()

    def log_gpu_info(self, step):
        
        handle = nvmlDeviceGetHandleByIndex(self.device_id)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        power_usage = nvmlDeviceGetPowerUsage(handle) / 1000
       
        print(f"[Step {step} | Rank {self.rank} / GPU {self.device_id}] "
              f"Memory: {mem_info.used / 1024**2:.2f} MB, "
              f"Temperature: {temperature}°C, "
              f"Power: {power_usage:.2f} W, ")

    def on_step_end(self, args, state, control, **kwargs):
        if self.rank % 32 == 0:
            self.log_gpu_info(state.global_step)

    def __del__(self):
        nvmlShutdown()


class MilestoneCheckpointCallback(TrainerCallback):
    """
    Callback that saves milestone checkpoints.
    In addition to the Trainer's normal checkpoints (subject to save_total_limit),
    this copies checkpoints to a milestones directory for permanent retention
    every milestone_interval steps.
    
    E.g., with milestone_interval=1000, at step 3800:
    - milestones/checkpoint-1000, checkpoint-2000, checkpoint-3000 (permanently retained)
    - checkpoint-3400, checkpoint-3600, checkpoint-3800 (subject to save_total_limit)
    """
    
    def __init__(self, milestone_interval: int = 1000):
        """
        Args:
            milestone_interval: Milestone interval in steps, default 1000
        """
        super().__init__()
        self.milestone_interval = milestone_interval
        self.rank = dist.get_rank() if torch.distributed.is_initialized() else 0
    
    def on_save(self, args, state, control, **kwargs):
        """After saving a checkpoint, check if it should be saved as a milestone."""
        current_step = state.global_step
        
        # Check if current step is a milestone step
        if current_step > 0 and current_step % self.milestone_interval == 0:
            checkpoint_folder = f"checkpoint-{current_step}"
            source_dir = os.path.join(args.output_dir, checkpoint_folder)
            
            # Create milestones directory
            milestones_dir = os.path.join(args.output_dir, "milestones")
            target_dir = os.path.join(milestones_dir, checkpoint_folder)
            
            # Only perform copy on rank 0
            if self.rank == 0:
                try:
                    if os.path.exists(source_dir):
                        os.makedirs(milestones_dir, exist_ok=True)
                        
                        # Remove target if it already exists
                        if os.path.exists(target_dir):
                            shutil.rmtree(target_dir)
                        
                        # Copy checkpoint to milestones directory
                        shutil.copytree(source_dir, target_dir)
                        print(f"[MilestoneCheckpoint] Step {current_step}: Saved milestone checkpoint to {target_dir}")
                    else:
                        print(f"[MilestoneCheckpoint] Warning: Source checkpoint {source_dir} not found")
                except Exception as e:
                    print(f"[MilestoneCheckpoint] Error saving milestone checkpoint at step {current_step}: {e}")
            
            # Synchronize all processes
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        return control 




def save_fsdp_as_hf_format(state_dict, save_directory):
    """Save FSDP state dict in Hugging Face standard format."""
    from safetensors.torch import split_torch_state_dict_into_shards
    
    os.makedirs(save_directory, exist_ok=True)
    
    # Use Hugging Face's sharding logic
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, 
    )
    
    # Save each shard
    from safetensors.torch import save_file
    for filename, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor: state_dict[tensor] for tensor in tensors}
        save_file(shard, os.path.join(save_directory, filename))
    
    # Create index file
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
    
    print(f"Model saved in Hugging Face standard format to {save_directory}")


def load_config(config_path):
    """Read a config file and return a dictionary."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def auto_thinking_prompt_handler(conversations):
    """
    Automatically process conversation data to conform to thinking mode format.
    - For GPT replies with thinking content, append /think marker after the human question.
    - For empty thinking blocks, remove them from the answer without adding /no_think marker.
    """
    empty_thinking_block = '<think>\n\n</think>\n\n'
    
    def find_next_human(conversations, start_idx):
        for i in range(start_idx, len(conversations)):
            if conversations[i]['from'] == 'human':
                return i
        return len(conversations)
    
    def get_next_all_gpt_thinking_status(conversations, start_idx):
        """
        Check the thinking status of all GPT messages from start_idx until the next human message.
        Returns: (is_thinking_mode, has_empty_thinking, gpt_idx_list)
        - is_thinking_mode: Whether the last GPT message is in thinking mode (has non-empty thinking content)
        - has_empty_thinking: Whether there are empty thinking blocks
        - gpt_idx_list: List of indices of all GPT messages
        """
        is_thinking_mode = False
        has_empty_thinking = False
        gpt_idx_list = []
        for i in range(start_idx, len(conversations)):
            if conversations[i]['from'] == 'human':
                break
            
            if conversations[i]['from'] == 'gpt':
                value = conversations[i]['value']
                # Check if it contains an empty thinking block
                if empty_thinking_block in value:
                    has_empty_thinking = True
                
                # Check for non-empty thinking content (remove empty blocks first)
                value_to_check = value.replace(empty_thinking_block, '')
                if '<think>' in value_to_check and '</think>' in value_to_check:
                    # Extract all thinking block contents and check if any are non-empty
                    thinking_pattern = r'<think>(.*?)</think>'
                    matches = re.findall(thinking_pattern, value_to_check, re.DOTALL)
                    if matches and any(match.strip() for match in matches):
                        is_thinking_mode = True
                
                gpt_idx_list.append(i)
        return is_thinking_mode, has_empty_thinking, gpt_idx_list
    
    i = 0
    while i < len(conversations):
        human_idx = find_next_human(conversations, i)
        if human_idx == len(conversations):
            break
        is_thinking_mode, has_empty_thinking, gpt_idx_list = get_next_all_gpt_thinking_status(conversations, human_idx + 1)
        
        if len(gpt_idx_list) > 0:
            # If there are empty thinking blocks, remove them
            if has_empty_thinking:
                for gpt_idx in gpt_idx_list:
                    if empty_thinking_block in conversations[gpt_idx]['value']:
                        conversations[gpt_idx]['value'] = conversations[gpt_idx]['value'].replace(empty_thinking_block, '')
            
            # If in thinking mode, append /think marker
            if is_thinking_mode:
                if '/think' not in conversations[human_idx]['value'] and '/no_think' not in conversations[human_idx]['value']:
                    conversations[human_idx]['value'] = conversations[human_idx]['value'] + ' /think'
            # If thinking blocks were empty (already removed), do not add any marker
        i = human_idx + 1
    return conversations


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _resolve_image_path(image_info, media_root):
    if not isinstance(image_info, str):
        return None
    if image_info.startswith(("http://", "https://", "file://", "data:image")):
        return image_info
    return osp.join(media_root, image_info)


def _load_visual_prompt_source_image(sample, media_root):
    raw_images = sample.get("image") if sample.get("image") is not None else sample.get("image_list")
    image_list = _as_list(raw_images)
    if not image_list:
        return None, image_list

    source = image_list[0]
    if isinstance(source, Image.Image):
        return source.convert("RGB"), image_list

    image_path = _resolve_image_path(source, media_root)
    if image_path is None or image_path.startswith(("http://", "https://", "data:image")):
        return None, image_list
    if image_path.startswith("file://"):
        image_path = image_path[7:]

    try:
        return Image.open(image_path).convert("RGB"), image_list
    except Exception as exc:
        logger.warning("Failed to load visual prompt source image %s: %s", image_path, exc)
        return None, image_list


def _extract_detection_category(text):
    match = _DETECTION_CATEGORY_RE.search(text)
    if match is None:
        return None, None

    category = match.group("category").strip()
    if not category or "," in category or "</c>" in category or "<image" in category:
        return None, None

    return match, category


def _find_boxes_for_ref(answer, category):
    ref_pattern = re.compile(rf"<ref>\s*{re.escape(category)}\s*</ref>(?P<body>.*?)(?=<ref>|$)", re.DOTALL)
    match = ref_pattern.search(answer)
    if match is None:
        return []

    body = match.group("body")
    if "<box>None</box>" in body:
        return []

    boxes = []
    for box_match in _BOX_RE.finditer(body):
        x1, y1, x2, y2 = [int(v) for v in box_match.groups()]
        boxes.append((x1, y1, x2, y2))
    return boxes


def _crop_normalized_box(image: Image.Image, box):
    width, height = image.size
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, round(x1 / 1000 * width)))
    top = max(0, min(height - 1, round(y1 / 1000 * height)))
    right = max(left + 1, min(width, round(x2 / 1000 * width)))
    bottom = max(top + 1, min(height, round(y2 / 1000 * height)))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


def apply_visual_prompt_to_sample(sample, media_root):
    """Replace positive single-category detection text with cropped visual prompts.

    The source image remains image-1. Each positive visual prompt crop is appended
    as image-2, image-3, ... and the matching category text is replaced with that
    placeholder. Negative GT answers (`<box>None</box>`) stay as text prompts.
    """
    sample = copy.deepcopy(sample)
    conversations = sample.get("conversations") or []
    if not conversations:
        return sample

    source_image, image_list = _load_visual_prompt_source_image(sample, media_root)
    if source_image is None:
        return sample

    appended_crops = []
    next_image_idx = len(image_list) + 1

    for idx, conv in enumerate(conversations[:-1]):
        if conv.get("from") != "human":
            continue

        next_conv = conversations[idx + 1]
        if next_conv.get("from") != "gpt":
            continue

        match, category = _extract_detection_category(str(conv.get("value", "")))
        if match is None:
            continue

        boxes = _find_boxes_for_ref(str(next_conv.get("value", "")), category)
        if not boxes:
            continue

        crop = _crop_normalized_box(source_image, boxes[0])
        if crop is None:
            continue

        placeholder = f"<image-{next_image_idx}>"
        conv["value"] = (
            conv["value"][:match.start("category")]
            + placeholder
            + conv["value"][match.end("category"):]
        )
        appended_crops.append(crop)
        next_image_idx += 1

    if appended_crops:
        updated_images = image_list + appended_crops
        sample["image"] = updated_images
        sample.pop("image_list", None)

    return sample


def process_multimodal_sample(
    sample,
    media_root,
    max_frames,
    target_fps,
    video_total_pixels,
    auto_thinking_handler=False,
    visual_prompt=False,
):
    """
    Process a multimodal data sample and format it into the final message list.
    This function handles media path resolution, placeholder formatting, and the conversion of dialogue content.
    
    Args:
        auto_thinking_handler: Whether to auto-process thinking mode format, default False
        visual_prompt: Whether to replace positive single-category detection text with image crops.
    """
    if visual_prompt:
        sample = apply_visual_prompt_to_sample(sample, media_root)

    conversations = sample.get('conversations', [])
    
    # If auto_thinking_handler is enabled, process conversation format
    if auto_thinking_handler and conversations:
        conversations = auto_thinking_prompt_handler(conversations)
    
    # --- Step 1: Format media file paths ---
    image_data, video_data = [], []
    
    # Extract raw media from sample (support multiple field formats)
    raw_images = sample.get("image") or sample.get("image_list")
    raw_videos = sample.get("video") or sample.get("video_list")
    
    # Also check "data" field for nested media format
    if sample.get("data"):
        for item in sample["data"]:
            item_type = item.get("type")
            if item_type == "image" and not raw_images:
                raw_images = item.get("image_list") or item.get("image")
            elif item_type == "video" and not raw_videos:
                raw_videos = item.get("video_list") or item.get("video")
    
    # Process images
    if raw_images:
        image_list = raw_images if isinstance(raw_images, list) else [raw_images]
        for img in image_list:
            if isinstance(img, str):
                image_data.append(osp.join(media_root, img))
            elif isinstance(img, Image.Image):
                image_data.append(img.convert("RGB"))
            elif isinstance(img, dict):
                if 'video' in img:  # Support the case where video frames are treated as images
                    video_data.append(osp.join(media_root, img['video']))
                else:  # Handle lmdb-format images
                    img_copy = img.copy()
                    if 'lmdb_file' in img_copy:
                        img_copy['lmdb_file'] = osp.join(media_root, img_copy['lmdb_file'])
                    image_data.append(img_copy)

    # Process videos
    if raw_videos:
        video_list = raw_videos if isinstance(raw_videos, list) else [raw_videos]
        for vid in video_list:
            if isinstance(vid, str):
                video_data.append(dict(video=osp.join(media_root, vid)))

    # If there is no media and no conversation, return empty
    if not conversations and not image_data and not video_data:
        return []

    # --- Step 2: Check and supplement placeholders ---
    media_map = {'image': image_data, 'video': video_data}
    
    for data_type, media_list in media_map.items():
        if not media_list:
            continue
        
        # Concatenate all conversation texts for placeholder checking
        all_texts = ''.join(conv['value'] for conv in conversations)
        
        # Count generic placeholders like <image> or <video>
        generic_placeholder = f'<{data_type}>'
        generic_count = all_texts.count(generic_placeholder)
        
        # If generic placeholders exist, replace them with numbered ones sequentially
        if generic_count > 0:
            counter = [0]  # Use list to allow modification in nested function
            def replace_with_number(match):
                counter[0] += 1
                return f'<{data_type}-{counter[0]}>'
            
            pattern = re.compile(re.escape(generic_placeholder))
            for conv in conversations:
                conv['value'] = pattern.sub(replace_with_number, conv['value'])
            # Update all_texts for further checks
            all_texts = ''.join(conv['value'] for conv in conversations)

        # Check if each media file has a corresponding numbered placeholder. If not, add them at the beginning.
        placeholders_to_add = ''
        for i in range(len(media_list)):
            if f'<{data_type}-{i+1}>' not in all_texts:
                placeholders_to_add += f'<{data_type}-{i+1}>'
        
        if placeholders_to_add and conversations:
            conversations[0]['value'] = placeholders_to_add + conversations[0]['value']

    # --- Step 3: Construct the final message format ---
    # If there is no media, treat as plain text format
    if not image_data and not video_data:
        return [
            {
                "role": 'user' if conv['from'] == 'human' else 'assistant',
                "content": [{"type": "text", "text": conv['value']}] if conv['from'] == 'human' else conv['value']
            }
            for conv in conversations
        ]

    # Handle as multimodal format
    new_messages = []
    placeholder_pattern = re.compile(r"<(image|video)-(\d+)>")
    
    for conv in conversations:
        role = 'user' if conv['from'] == 'human' else 'assistant'
        value = conv['value']

        if role == 'assistant':
            new_messages.append({"role": role, "content": value})
            continue

        # Handle user message
        content_list = [{"type": "text", "text": value}]
        matches = placeholder_pattern.findall(value)
        
        for media_type, num_str in matches:
            index = int(num_str) - 1
            try:
                path_info = media_map[media_type][index]
                if media_type == 'image':
                    content_list.append({"type": "image", "image": path_info})
                elif media_type == 'video':
                    video_path = path_info['video'] if isinstance(path_info, dict) else path_info
                    content_list.append({
                        "type": "video", "video": video_path,
                        'max_frames': max_frames, "fps": target_fps,
                        "video_total_pixels": video_total_pixels
                    })
            except (IndexError, KeyError):
                # Ignore the placeholder if index or type does not exist
                pass
        
        new_messages.append({"role": role, "content": content_list})
        
    return new_messages

def draw_mark(image_inputs: list[dict], metadata: dict):
    if "type" in metadata and metadata["type"] in DRAW_FUNCTIONS:
        draw_fn = DRAW_FUNCTIONS[metadata['type']]
        if len(image_inputs) == 1:
            draw_fn(image_inputs[0], metadata)
        else:
            draw_fn(image_inputs, metadata)

    return [img for img in image_inputs if img is not None]
