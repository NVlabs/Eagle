"""
Eagle3-VL MTP Finetuning with Stream (Online) Packing
Combines Multi-Token Prediction (MTP) with efficient stream packing

Key Features:
1. [MTP] Multi-token prediction with block-based attention pattern
2. [Stream Packing] Online sample packing for efficient GPU utilization
3. [Attention] Proper attention masking for both packing and MTP blocks
4. [Resume] Perfect stateful resume support

"""

import os
import os.path as osp
import copy
import logging
import random
import sys
import warnings
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field

import json
import shutil
import time
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
import torch.distributed as dist
import transformers
import traceback
import socket
from eaglevl.dist_utils import init_dist

import packaging.version as version
from eaglevl.model.moon_vit.modeling_vit import MoonVitPretrainedModel
from eaglevl.patch import (
    replace_liger_fused_ops,
    replace_train_dataloader,
    replace_train_sampler
)
from eaglevl.model.locany.modeling_locateanything import LocateAnythingForConditionalGeneration
from eaglevl.model.locany.configuration_locateanything import LocateAnythingConfig
from eaglevl.utils.locany.processing_locateanything import LocateAnythingProcessor
from eaglevl.utils.locany.image_processing_locateanything import LocateAnythingImageProcessor
from eaglevl.sp_utils import set_pg_manager, get_pg_manager
from eaglevl.train.constants import (
    special_tokens_list, IMG_CONTEXT_TOKEN, TEXT_MASK_TOKEN,
    NULL_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN,
    REF_START_TOKEN, REF_END_TOKEN, number_tokens_list
)
from eaglevl.train.arguments import ModelArguments, DataTrainingArguments
from eaglevl.train.trainer_monkey_patch import replace_create_optimizer_with_various_lr
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed, AutoProcessor)
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
from transformers import TrainerCallback
from eaglevl.train.tools import (SaveCheckpointCallback, MemoryLoggerCallback, 
                                  MilestoneCheckpointCallback, get_last_checkpoint_guard, 
                                  load_config, process_multimodal_sample)
from eaglevl.train.augmentation import apply_resize_augmentation
from dotenv import load_dotenv
load_dotenv()
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


if version.parse(torch.__version__) >= version.parse("2.4.0"):
    torch.serialization.add_safe_globals(
        [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))])


# ============ Patch ============
replace_liger_fused_ops()

# Patch HF PreTrainedModel to accept "magi" as attn_implementation.
# HF validates attn_implementation in __init__ and only allows
# eager/sdpa/flash_attention_2/flash_attention_3. We keep "magi" as-is so
# the custom Qwen2 stack can dispatch to the MagiAttention path explicitly.
import transformers.modeling_utils as _hf_modeling_utils
_orig_check_and_adjust = _hf_modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation

def _patched_check_and_adjust(self, attn_implementation, is_init_check=False):
    if attn_implementation == "magi":
        return "magi"
    return _orig_check_and_adjust(self, attn_implementation, is_init_check=is_init_check)

_hf_modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation = _patched_check_and_adjust
# ========================

# ============ for loading large images ============
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
# ==================================================

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class LazyJsonlLoader:
    """Lazy loader for JSONL files with fast index-based access."""
    
    def __init__(self, paths: List[str]):
        if isinstance(paths, str):
            paths = [paths]
        self.paths = paths
        self.offsets = []
        self._file_handles = {}
        self._build_index()

    def _build_index(self):
        for file_idx, path in enumerate(self.paths):
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}")
                continue
            with open(path, 'rb') as f:
                offset = 0
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        self.offsets.append((file_idx, offset))
                    offset = f.tell()
        logger.info(f"Indexed {len(self.offsets)} lines from {len(self.paths)} files.")

    def __len__(self):
        return len(self.offsets)

    def _get_file_handle(self, file_idx: int):
        import threading
        thread_id = threading.get_ident()
        key = (thread_id, file_idx)
        if key not in self._file_handles:
            self._file_handles[key] = open(self.paths[file_idx], 'r', encoding='utf-8')
        return self._file_handles[key]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError("Index out of range")
        file_idx, offset = self.offsets[idx]
        f = self._get_file_handle(file_idx)
        f.seek(offset)
        line = f.readline()
        return json.loads(line)

    def __del__(self):
        for f in self._file_handles.values():
            try:
                f.close()
            except:
                pass
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file_handles'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file_handles = {}


class LazySupervisedDatasetMTP(Dataset):
    """Lazy-loading dataset with MTP (Multi-Token Prediction) support."""
    
    def __init__(self,
                 ds_name: str,
                 meta: dict,
                 processor, 
                 block_size: int = 6,
                 repeat_time: float = 1,
                 max_frames: int = 16,
                 target_fps: int = 2,
                 video_total_pixels: int = 32000 * 28 * 28 * 0.9):
        super().__init__()
        self.ds_name = ds_name
        self.processor = processor
        self.max_length = self.processor.tokenizer.model_max_length
        self.repeat_time = repeat_time
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.video_total_pixels = video_total_pixels
        self.block_size = block_size
        self.data_augment = meta.get("data_augment", False)
        self.visual_prompt = bool(meta.get("visual_prompt", False))

        ann_paths = meta["annotation"]
        if not isinstance(ann_paths, (list, tuple)):
            ann_paths = [ann_paths]
        self.root = meta.get("root", "")
        
        logger.info(f"[Dataset] {self.ds_name} Indexing JSONL files...")
        start_time = time.time()
        self.lazy_loader = LazyJsonlLoader(ann_paths)
        logger.info(f"[Dataset] {self.ds_name} Indexing done in {time.time() - start_time:.2f}s.")
        
        original_num_rows = len(self.lazy_loader)
        logger.info(
            f"[Dataset] {self.ds_name} Found {original_num_rows} samples. "
            f"visual_prompt={self.visual_prompt}"
        )
        self.active_indices = list(range(original_num_rows))
        
        if repeat_time < 1:
            if original_num_rows > 0:
                partial_len = int(original_num_rows * repeat_time)
                if partial_len > 0:
                    rnd = random.Random(10086)
                    sampled_indices = set(rnd.sample(range(original_num_rows), partial_len))
                    self.active_indices = [i for i in range(original_num_rows) if i in sampled_indices]
                    logger.info(f"[Dataset] {self.ds_name} Downsampled to {len(self.active_indices)} samples.")
                else:
                    self.active_indices = []
        
        self._length = len(self.active_indices)

    def __len__(self):
        return self._length

    def get_targets_flag_with_mtp(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create MTP (Multi-Token Prediction) blocks with proper labels."""
        tokenizer = self.processor.tokenizer
        targets_flag = torch.zeros_like(input_ids)
        
        box_end_id = tokenizer.convert_tokens_to_ids("</box>")
        ref_end_id = tokenizer.convert_tokens_to_ids("</ref>")
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        null_id = tokenizer.convert_tokens_to_ids("<null>")
        mask_id = tokenizer.convert_tokens_to_ids("<text_mask>")
        
        start_header_idxs = torch.where(
            input_ids == tokenizer.convert_tokens_to_ids("<|im_start|>")
        )[0]
        assistant_idxs = torch.where(
            input_ids == tokenizer.convert_tokens_to_ids("assistant")
        )[0]
        eot_idxs = torch.where(input_ids == eos_id)[0]
        
        # Identify assistant response positions for mask block generation
        # 同时记录每个 assistant 回复内 label 区间 [label_start, label_end]（含）
        resp_start_position_ids = []
        resp_end_position_ids = []
        resp_label_ranges = []
        
        for assistant_idx in assistant_idxs:
            sets = list(set(start_header_idxs + 1))
            sets = [each.item() for each in sets]
            if assistant_idx.item() in sets:
                st = assistant_idx + 1
                for eot_idx in eot_idxs:
                    if eot_idx > st:
                        # st+1 到 eot_idx（含）是需要监督的 token 区间
                        targets_flag[st+1: eot_idx + 1] = 1
                        resp_start_position_ids.append((st).item())
                        resp_end_position_ids.append((eot_idx + 1).item())
                        label_start = (st + 1).item()
                        label_end = eot_idx.item()
                        resp_label_ranges.append((label_start, label_end))
                        break
        
        targets = input_ids.clone()
        assert targets_flag.sum() > 0, f"No valid labels for training, skip sample in {self.ds_name}"
        targets[targets_flag == 0] = IGNORE_TOKEN_ID
        
        input_ids_np = input_ids.squeeze(0).cpu().numpy()
        targets_np = targets.squeeze(0).cpu().numpy()
        len_input_ids = len(input_ids_np)
        
        # ========= 分支 1：无检测序列标记（</box>、</ref>），采用随机 block 切分 =========
        has_box = (input_ids == box_end_id).any().item()
        has_ref = (input_ids == ref_end_id).any().item()
        if not (has_box or has_ref):
            all_mask_input_ids = []
            all_mask_targets = []
            all_mask_positions = []

            # 仅在单个 assistant 回复内部随机切分
            for label_start, label_end in resp_label_ranges:
                # 该回复内部的有效 supervision 位置（去掉 IGNORE_TOKEN_ID）
                resp_valid_positions = [
                    i for i in range(label_start, label_end + 1)
                    if targets_np[i] != IGNORE_TOKEN_ID
                ]
                num_valid = len(resp_valid_positions)
                if num_valid == 0:
                    continue

                num_blocks = num_valid // self.block_size
                if num_blocks == 0:
                    continue

                # 为了保证可复现性，随机数种子在外部 __getitem__ 中已设置
                used_tokens = num_blocks * self.block_size
                remaining = num_valid - used_tokens  # 0 <= remaining < self.block_size
                # 在 [0, remaining] 中随机选择一个 offset，使得切分起点是随机的
                offset = np.random.randint(0, remaining + 1) if remaining > 0 else 0

                for block_idx in range(num_blocks):
                    start_pos = offset + block_idx * self.block_size
                    end_pos = start_pos + self.block_size
                    block_valid = resp_valid_positions[start_pos:end_pos]
                    if len(block_valid) == 0:
                        continue

                    first_token_idx = int(block_valid[0])
                    anchor_idx = max(first_token_idx - 1, 0)

                    # 构建 target block：长度为 block_size，默认全 IGNORE_TOKEN_ID
                    target_block = np.full(self.block_size, IGNORE_TOKEN_ID, dtype=targets_np.dtype)
                    candidate_tokens = input_ids_np[block_valid]

                    for i, tok in enumerate(candidate_tokens):
                        if i >= self.block_size:
                            break
                        target_block[i] = tok
                        if tok == eos_id:
                            # EOS 之后保持 IGNORE_TOKEN_ID，不再计算 loss
                            break

                    # 构建输入 block：第一个位置是 anchor，其余为 <text_mask>
                    mask_input_block = np.full(self.block_size, mask_id, dtype=input_ids_np.dtype)
                    mask_input_block[0] = input_ids_np[anchor_idx]

                    # 位置 id 与原序列对齐，从 anchor 开始连续增长
                    pos_block = np.arange(anchor_idx, anchor_idx + self.block_size, dtype=np.int32)

                    all_mask_input_ids.append(mask_input_block)
                    all_mask_targets.append(target_block)
                    all_mask_positions.append(pos_block)

            if len(all_mask_input_ids) > 0:
                final_mask_ids = np.concatenate(all_mask_input_ids)
                final_mask_targets = np.concatenate(all_mask_targets)
                final_mask_positions = np.concatenate(all_mask_positions)

                pad_id = tokenizer.pad_token_id
                bridge_ignore = np.array([IGNORE_TOKEN_ID], dtype=targets_np.dtype)
                pad_token = np.array([pad_id], dtype=input_ids_np.dtype)

                input_ids_out = np.concatenate([input_ids_np, final_mask_ids, pad_token])
                targets_out = np.concatenate([targets_np, bridge_ignore, final_mask_targets])

                orig_pos = np.arange(len_input_ids, dtype=np.int32)
                pad_pos = np.array([final_mask_positions[-1] + 1], dtype=np.int32)
                position_ids_out = np.concatenate([orig_pos, final_mask_positions, pad_pos])
            else:
                input_ids_out = input_ids_np
                targets_out = targets_np
                position_ids_out = np.arange(len_input_ids, dtype=np.int32)

            input_ids = torch.tensor(input_ids_out, dtype=torch.long)
            targets = torch.tensor(targets_out, dtype=torch.long)
            position_ids = torch.tensor(position_ids_out, dtype=torch.long)

            return dict(
                input_ids=input_ids,
                labels=targets,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),
                position_ids=position_ids,
            )

        # ========= 分支 2：检测序列标记存在，使用原有的 box/ref-aware 逻辑 =========
        all_mask_input_ids = []
        all_mask_targets = []
        all_mask_positions = []
        
        for start, end in zip(resp_start_position_ids, resp_end_position_ids):
            curr = start
            
            while curr < end:
                anchor_token = input_ids_np[curr]
                if anchor_token == eos_id:
                    break
                
                pred_start = curr + 1
                if pred_start > end:
                    break
                
                candidates = input_ids_np[pred_start : min(pred_start + self.block_size, end + 1)]
                if len(candidates) == 0:
                    break

                valid_len = len(candidates)
                
                eos_indices = np.where(candidates == eos_id)[0]
                if len(eos_indices) > 0:
                    first_eos_idx = eos_indices[0]
                    if first_eos_idx == 0:
                        valid_len = 1
                    else:
                        valid_len = first_eos_idx
                
                if valid_len > 1 or (len(eos_indices) > 0 and eos_indices[0] != 0):
                    ref_indices = np.where(candidates[:valid_len] == ref_end_id)[0]
                    if len(ref_indices) > 0:
                        valid_len = min(valid_len, ref_indices[0] + 1)
                    
                    box_indices = np.where(candidates[:valid_len] == box_end_id)[0]
                    if len(box_indices) > 0:
                        valid_len = min(valid_len, box_indices[0] + 1)

                target_block = np.full(self.block_size, null_id, dtype=input_ids_np.dtype)
                target_block[:valid_len] = candidates[:valid_len]
                
                mask_input_block = np.full(self.block_size, mask_id, dtype=input_ids_np.dtype)
                mask_input_block[0] = anchor_token 
                
                pos_block = np.arange(curr, curr + self.block_size, dtype=np.int32)
                
                all_mask_input_ids.append(mask_input_block)
                all_mask_targets.append(target_block)
                all_mask_positions.append(pos_block)
                
                curr += valid_len
                
        if len(all_mask_input_ids) > 0:
            final_mask_ids = np.concatenate(all_mask_input_ids)
            final_mask_targets = np.concatenate(all_mask_targets)
            final_mask_positions = np.concatenate(all_mask_positions)
            
            pad_id = tokenizer.pad_token_id
            bridge_ignore = np.array([IGNORE_TOKEN_ID], dtype=targets_np.dtype)
            pad_token = np.array([pad_id], dtype=input_ids_np.dtype)
            
            input_ids_out = np.concatenate([input_ids_np, final_mask_ids, pad_token])
            targets_out = np.concatenate([targets_np, bridge_ignore, final_mask_targets])
            
            orig_pos = np.arange(len_input_ids, dtype=np.int32)
            pad_pos = np.array([final_mask_positions[-1] + 1], dtype=np.int32)
            position_ids_out = np.concatenate([orig_pos, final_mask_positions, pad_pos])
        else:
            input_ids_out = input_ids_np
            targets_out = targets_np
            position_ids_out = np.arange(len_input_ids, dtype=np.int32)
        
        input_ids = torch.tensor(input_ids_out, dtype=torch.long)
        targets = torch.tensor(targets_out, dtype=torch.long)
        position_ids = torch.tensor(position_ids_out, dtype=torch.long)
        
        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            position_ids=position_ids,
        )

    def _validate_image_token_alignment(self, input_ids: torch.Tensor, pixel_values, image_grid_hws) -> None:
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            image_token = getattr(self.processor, "image_token", IMG_CONTEXT_TOKEN)
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)

        if isinstance(image_grid_hws, torch.Tensor):
            grid_array = image_grid_hws.detach().cpu().numpy()
        else:
            grid_array = np.asarray(image_grid_hws)

        merge_kernel = getattr(self.processor.image_processor, "merge_kernel_size", [2, 2])
        expected_context_tokens = int(
            sum(int(h) * int(w) // (int(merge_kernel[0]) * int(merge_kernel[1])) for h, w in grid_array)
        )
        actual_context_tokens = int((input_ids == image_token_id).sum().item())
        if actual_context_tokens != expected_context_tokens:
            raise ValueError(
                f"[{self.ds_name}] image token mismatch: actual={actual_context_tokens}, "
                f"expected={expected_context_tokens}, num_images={len(grid_array)}, "
                f"grid_hws={grid_array.tolist()}"
            )

        expected_patches = int(sum(int(h) * int(w) for h, w in grid_array))
        actual_patches = int(pixel_values.shape[0])
        if actual_patches != expected_patches:
            raise ValueError(
                f"[{self.ds_name}] pixel patch mismatch: actual={actual_patches}, "
                f"expected={expected_patches}, num_images={len(grid_array)}, "
                f"grid_hws={grid_array.tolist()}"
            )

    def multi_modal_get_item(self, messages: list) -> Dict[str, torch.Tensor]:
        message_text = self.processor.py_apply_chat_template(messages, tokenize=False)
        image_inputs, video_inputs = self.processor.process_vision_info(messages)
        
        if image_inputs is not None:
            image_inputs = [
                apply_resize_augmentation(
                    img, data_augment=self.data_augment,
                    min_long_edge=640, max_long_edge=2560, augment_prob=0.5
                ) for img in image_inputs
            ]
        
        inputs = self.processor(
            text=message_text, images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=False, truncation=True
        )
        input_ids = inputs["input_ids"][0]

        if "pixel_values" not in inputs:
            pixel_values = torch.zeros((4, 3, 14, 14), dtype=torch.float32)
            image_flags = torch.tensor([0], dtype=torch.long)
            image_grid_hws = np.array([[2, 2]])
        else:
            pixel_values = inputs["pixel_values"]
            image_grid_hws = inputs["image_grid_hws"]
            image_flags = torch.tensor([len(inputs["image_grid_hws"])], dtype=torch.long)
            self._validate_image_token_alignment(input_ids, pixel_values, image_grid_hws)

        labels_dict = self.get_targets_flag_with_mtp(input_ids)
        
        return dict(
            input_ids=labels_dict["input_ids"],
            labels=labels_dict["labels"],
            position_ids=labels_dict["position_ids"],
            attention_mask=labels_dict["attention_mask"],
            image_flags=image_flags,
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        retry_count = 0
        current_idx = idx
        
        seed = int(idx + 10086)
        random.seed(seed)
        np.random.seed(seed)
        
        while retry_count <= 10:
            real_idx = self.active_indices[(current_idx + retry_count) % self._length]
            try:
                data_item = self.lazy_loader[real_idx]
                data_item = process_multimodal_sample(
                    data_item, self.root, self.max_frames, 
                    self.target_fps, self.video_total_pixels,
                    visual_prompt=self.visual_prompt,
                )
                return self.multi_modal_get_item(data_item)
            except Exception as e:
                tb = traceback.format_exc()
                logger.warning(f"[{self.ds_name}] idx {real_idx} failed: {e}\n{tb}")
                retry_count += 1
        
        raise RuntimeError(f"[{self.ds_name}] Failed after 10 retries")
    
    def get_sample_at_global_idx(self, global_idx: int, seed: int) -> Dict[str, torch.Tensor]:
        """Get a sample by global index (used for resume)."""
        ds_len = self._length
        if ds_len == 0:
            raise ValueError("Dataset is empty")
        
        epoch = global_idx // ds_len
        pos = global_idx % ds_len
        
        shuffle_seed = seed + epoch * 999983
        rng = random.Random(shuffle_seed)
        indices = list(range(ds_len))
        rng.shuffle(indices)
        
        real_idx = indices[pos]
        return self[real_idx]


@dataclass
class IteratorState:
    """Iterator state."""
    seed: int
    global_idx: int
    
    def to_dict(self) -> dict:
        return {'seed': self.seed, 'global_idx': self.global_idx}
    
    @classmethod
    def from_dict(cls, d: dict) -> 'IteratorState':
        return cls(seed=d['seed'], global_idx=d['global_idx'])


class DeterministicIterator:
    """Deterministic dataset iterator."""
    
    def __init__(self, dataset: LazySupervisedDatasetMTP, seed: int, start_global_idx: int = 0):
        self.dataset = dataset
        self.seed = seed
        self.ds_len = len(dataset)
        self.ds_name = getattr(dataset, 'ds_name', 'unknown')
        self.global_idx = start_global_idx
        
        self._cached_epoch = -1
        self._cached_indices = None
    
    def _get_epoch_indices(self, epoch: int) -> list:
        if self._cached_epoch == epoch and self._cached_indices is not None:
            return self._cached_indices
        
        shuffle_seed = self.seed + epoch * 999983
        rng = random.Random(shuffle_seed)
        indices = list(range(self.ds_len))
        rng.shuffle(indices)
        
        self._cached_epoch = epoch
        self._cached_indices = indices
        return indices
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[dict, int]:
        if self.ds_len == 0:
            raise StopIteration
        
        current_global_idx = self.global_idx
        epoch = current_global_idx // self.ds_len
        pos = current_global_idx % self.ds_len
        indices = self._get_epoch_indices(epoch)
        
        real_idx = indices[pos]
        sample = self.dataset[real_idx]
        self.global_idx += 1
        
        return sample, current_global_idx
    
    def peek_global_idx(self) -> int:
        return self.global_idx
    
    def state_dict(self) -> dict:
        return IteratorState(seed=self.seed, global_idx=self.global_idx).to_dict()
    
    @classmethod
    def from_state_dict(cls, dataset: LazySupervisedDatasetMTP, state: dict) -> 'DeterministicIterator':
        return cls(dataset=dataset, seed=state['seed'], start_global_idx=state['global_idx'])


@dataclass
class WorkerState:
    """Complete state of a worker, including buffer state."""
    iterator_states: List[dict]
    sample_rng_state: tuple
    samples_produced: int
    batches_produced: int
    current_batch_locations: List[Tuple[int, int]]
    buffer_locations: List[Tuple[int, int]]
    
    def to_dict(self) -> dict:
        return {
            'iterator_states': self.iterator_states,
            'sample_rng_state': self.sample_rng_state,
            'samples_produced': self.samples_produced,
            'batches_produced': self.batches_produced,
            'current_batch_locations': self.current_batch_locations,
            'buffer_locations': self.buffer_locations,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'WorkerState':
        raw_rng_state = d['sample_rng_state']
        if isinstance(raw_rng_state, (list, tuple)) and len(raw_rng_state) == 3:
            version, internal_state, gauss_next = raw_rng_state
            if isinstance(internal_state, list):
                internal_state = tuple(internal_state)
            sample_rng_state = (version, internal_state, gauss_next)
        else:
            sample_rng_state = tuple(raw_rng_state) if isinstance(raw_rng_state, list) else raw_rng_state
        
        return cls(
            iterator_states=d['iterator_states'],
            sample_rng_state=sample_rng_state,
            samples_produced=d.get('samples_produced', 0),
            batches_produced=d.get('batches_produced', 0),
            current_batch_locations=d.get('current_batch_locations', []),
            buffer_locations=d.get('buffer_locations', []),
        )


class StreamPackedDatasetMTP(IterableDataset):
    """Online packing IterableDataset with MTP support and perfect stateful resume."""
    
    def __init__(
        self,
        tokenizer,
        data_rank: int,
        data_world_size: int,
        datasets: List[LazySupervisedDatasetMTP],
        dataset_weight: List[float] = None,
        max_num_tokens_per_sample: int = 16384,
        max_num_tokens: int = 36864,
        log_freq: int = 10000,
        base_seed: int = 42,
        buffer_size: int = 32,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.datasets = datasets
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.max_num_tokens = max_num_tokens
        self.log_freq = log_freq
        self.base_seed = base_seed
        self.buffer_size = buffer_size

        if dataset_weight is None:
            dataset_weight = [1] * len(datasets)
        total_weight = sum(dataset_weight)
        self.dataset_weight = [w / total_weight for w in dataset_weight]
        
        self._worker_states: Dict[str, dict] = {}
        self._resume_states: Dict[str, dict] = {}
        
        if get_rank() == 0:
            ds_info = '\n'.join([f'  {ds.ds_name}: weight={w*100:.2f}%, len={len(ds)}' 
                                 for ds, w in zip(self.datasets, self.dataset_weight)])
            logger.info(f'StreamPackedDatasetMTP initialized:\n'
                       f'  max_num_tokens_per_sample={max_num_tokens_per_sample}\n'
                       f'  max_num_tokens={max_num_tokens}\n'
                       f'  buffer_size={buffer_size}\n'
                       f'  base_seed={base_seed}\n'
                       f'  data_rank={data_rank}, data_world_size={data_world_size}\n'
                       f'Datasets:\n{ds_info}')

    def state_dict(self) -> dict:
        return {
            'worker_states': copy.deepcopy(self._worker_states),
            'base_seed': self.base_seed,
            'version': 4,
        }
    
    def load_state_dict(self, state: dict):
        version = state.get('version', 1)
        if version < 3:
            logger.warning(f"Loading old state version {version}, perfect resume not available.")
        
        if 'worker_states' in state:
            self._resume_states = copy.deepcopy(state['worker_states'])
            if get_rank() == 0:
                logger.info(f"Loaded resume states for {len(self._resume_states)} workers")

    def _get_sample_length(self, sample: Optional[dict]) -> int:
        if sample is None:
            return 0
        return sample['input_ids'].size(0)

    def _merge_samples(self, batch: Optional[dict], sample: dict) -> dict:
        """Merge a sample into the batch, tracking sample lengths."""
        sample_len = sample['input_ids'].size(0)
        
        if batch is None:
            result = copy.copy(sample)
            result['_sample_lengths'] = [sample_len]
            return result
        
        result = {}
        for k in batch:
            if k == '_sample_lengths':
                result[k] = batch[k] + [sample_len]
            elif k == 'image_grid_hws':
                if isinstance(batch[k], np.ndarray) and isinstance(sample[k], np.ndarray):
                    result[k] = np.concatenate([batch[k], sample[k]], axis=0)
                else:
                    result[k] = torch.cat([batch[k], sample[k]])
            elif k == 'pixel_values':
                result[k] = torch.cat([batch[k], sample[k]])
            elif isinstance(batch[k], torch.Tensor):
                result[k] = torch.cat([batch[k], sample[k]])
            else:
                result[k] = batch[k]
        
        return result

    def _finalize_batch(self, batch: dict) -> dict:
        """Finalize batch by computing sub_sample_lengths."""
        sample_lengths = batch.pop('_sample_lengths', [batch['input_ids'].size(0)])
        sub_sample_lengths = torch.tensor(sample_lengths, dtype=torch.long)
        
        batch['sub_sample_lengths'] = sub_sample_lengths
        # attention_mask is not needed here; model will use sub_sample_lengths to generate data_index
        
        return batch

    def __iter__(self):
        from torch.utils.data import get_worker_info
        
        worker_info = get_worker_info()
        local_worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        
        global_worker_id = num_workers * self.data_rank + local_worker_id
        worker_key = f'worker_{global_worker_id}'
        is_main_log = (global_worker_id == 0)
        
        if is_main_log:
            logger.info(f'[{worker_key}] Starting iteration with MTP Buffer Strategy...')
        
        # Initialize state
        worker_seed = self.base_seed + global_worker_id
        sample_rng = random.Random(worker_seed)
        
        iterators: List[DeterministicIterator] = []
        iterator_seeds: List[int] = []
        for ds_idx, ds in enumerate(self.datasets):
            iter_seed = self.base_seed + global_worker_id * 10000 + ds_idx
            iterator_seeds.append(iter_seed)
            iterators.append(DeterministicIterator(ds, seed=iter_seed, start_global_idx=0))
        
        samples_produced = 0
        batches_produced = 0
        skipped_count = 0
        
        current_batch = None
        current_batch_locations: List[Tuple[int, int]] = []
        buffer: List[Tuple[dict, int, int]] = []
        
        # Resume handling
        if worker_key in self._resume_states:
            saved = self._resume_states.pop(worker_key)
            try:
                ws = WorkerState.from_dict(saved)
                
                if is_main_log:
                    logger.info(f'[{worker_key}] Resuming...')
                
                for ds_idx, iter_state in enumerate(ws.iterator_states):
                    iterators[ds_idx] = DeterministicIterator.from_state_dict(
                        self.datasets[ds_idx], iter_state
                    )
                
                sample_rng.setstate(ws.sample_rng_state)
                samples_produced = ws.samples_produced
                batches_produced = ws.batches_produced
                
                if ws.current_batch_locations:
                    if is_main_log:
                        logger.info(f'[{worker_key}] Rebuilding current_batch ({len(ws.current_batch_locations)} samples)...')
                    for loc in ws.current_batch_locations:
                        ds_idx, global_idx = loc
                        try:
                            sample = self.datasets[ds_idx].get_sample_at_global_idx(global_idx, iterator_seeds[ds_idx])
                            if self._get_sample_length(sample) <= self.max_num_tokens_per_sample:
                                current_batch = self._merge_samples(current_batch, sample)
                                current_batch_locations.append(loc)
                        except Exception as e:
                            logger.warning(f'[{worker_key}] Failed to restore batch sample {loc}: {e}')

                if ws.buffer_locations:
                    if is_main_log:
                        logger.info(f'[{worker_key}] Rebuilding buffer ({len(ws.buffer_locations)} samples)...')
                    for loc in ws.buffer_locations:
                        ds_idx, global_idx = loc
                        try:
                            sample = self.datasets[ds_idx].get_sample_at_global_idx(global_idx, iterator_seeds[ds_idx])
                            if self._get_sample_length(sample) <= self.max_num_tokens_per_sample:
                                buffer.append((sample, ds_idx, global_idx))
                        except Exception as e:
                            logger.warning(f'[{worker_key}] Failed to restore buffer sample {loc}: {e}')
                
                if is_main_log:
                    logger.info(f'[{worker_key}] Resume complete. Buffer size: {len(buffer)}')
                    
            except Exception as e:
                logger.error(f'[{worker_key}] Failed to resume: {e}')
                traceback.print_exc()
                current_batch = None
                current_batch_locations = []
                buffer = []

        # Helper functions
        def build_state_snapshot() -> dict:
            return WorkerState(
                iterator_states=[it.state_dict() for it in iterators],
                sample_rng_state=sample_rng.getstate(),
                samples_produced=samples_produced,
                batches_produced=batches_produced,
                current_batch_locations=list(current_batch_locations),
                buffer_locations=[(b[1], b[2]) for b in buffer],
            ).to_dict()

        def fetch_next_sample() -> Tuple[dict, int, int]:
            nonlocal samples_produced, skipped_count
            while True:
                ds_idx = sample_rng.choices(range(len(self.datasets)), weights=self.dataset_weight)[0]
                try:
                    sample, global_idx = next(iterators[ds_idx])
                    samples_produced += 1
                    if self._get_sample_length(sample) > self.max_num_tokens_per_sample:
                        skipped_count += 1
                        continue
                    return sample, ds_idx, global_idx
                except StopIteration:
                    continue

        # Main loop
        while True:
            # Try to fill current_batch from buffer (Best-Fit Strategy)
            current_len = self._get_sample_length(current_batch)
            remaining_space = self.max_num_tokens - current_len
            
            best_fit_idx = -1
            max_fit_len = -1
            
            for i, (buf_sample, _, _) in enumerate(buffer):
                s_len = self._get_sample_length(buf_sample)
                if s_len <= remaining_space:
                    if s_len > max_fit_len:
                        max_fit_len = s_len
                        best_fit_idx = i
            
            if best_fit_idx != -1:
                sample, ds_idx, global_idx = buffer.pop(best_fit_idx)
                current_batch = self._merge_samples(current_batch, sample)
                current_batch_locations.append((ds_idx, global_idx))
                continue

            if len(buffer) < self.buffer_size:
                new_sample, ds_idx, global_idx = fetch_next_sample()
                new_len = self._get_sample_length(new_sample)
                
                if new_len <= remaining_space:
                    current_batch = self._merge_samples(current_batch, new_sample)
                    current_batch_locations.append((ds_idx, global_idx))
                    continue
                else:
                    buffer.append((new_sample, ds_idx, global_idx))
            
            # Yield Logic
            if current_batch is not None:
                batches_produced += 1
                output_batch = self._finalize_batch(current_batch)
                
                current_batch = None
                current_batch_locations = []
                
                # Start new batch with largest sample from buffer (Big Rocks First)
                if len(buffer) > 0:
                    buffer.sort(key=lambda x: self._get_sample_length(x[0]), reverse=True)
                    sample, ds_idx, global_idx = buffer.pop(0)
                    current_batch = self._merge_samples(None, sample)
                    current_batch_locations = [(ds_idx, global_idx)]
                
                state_snapshot = build_state_snapshot()
                
                output_batch['_worker_key'] = worker_key
                output_batch['_batch_idx'] = batches_produced
                output_batch['_state_snapshot'] = state_snapshot
                
                yield output_batch
                
                if is_main_log and batches_produced % self.log_freq == 0:
                    packing_efficiency = current_len / self.max_num_tokens * 100
                    logger.info(f'batches={batches_produced}, samples={samples_produced}, '
                               f'buffer_len={len(buffer)}, packing_eff={packing_efficiency:.1f}%')
            else:
                if len(buffer) == 0:
                    continue
                else:
                    buffer.sort(key=lambda x: self._get_sample_length(x[0]), reverse=True)
                    sample, ds_idx, global_idx = buffer.pop(0)
                    current_batch = self._merge_samples(None, sample)
                    current_batch_locations = [(ds_idx, global_idx)]


def packed_collate_fn_mtp(features: List[dict], dataset: Optional[StreamPackedDatasetMTP] = None) -> dict:
    """Collator for MTP packing: processes batch and preserves state metadata."""
    assert len(features) == 1, f"Expected batch_size=1 for packing, got {len(features)}"
    
    feat = features[0]
    input_len = int(feat['input_ids'].shape[0])
    label_len = int(feat['labels'].shape[0])
    pos_len = int(feat['position_ids'].shape[-1])
    sub_sample_lengths = feat['sub_sample_lengths']
    packed_len = int(sub_sample_lengths.sum().item()) if isinstance(sub_sample_lengths, torch.Tensor) else int(sum(sub_sample_lengths))
    non_ignore_labels = int(feat['labels'][1:].ne(IGNORE_TOKEN_ID).sum().item()) if feat['labels'].numel() > 1 else 0

    if not (input_len == label_len == pos_len == packed_len):
        raise ValueError(
            f"Packed feature length mismatch: input_ids={input_len}, labels={label_len}, "
            f"position_ids={pos_len}, sub_sample_lengths_sum={packed_len}"
        )
    if non_ignore_labels == 0:
        raise ValueError(
            f"Packed feature has no valid shifted labels: input_ids={input_len}, "
            f"labels_non_ignore_after_shift={non_ignore_labels}, "
            f"sub_sample_lengths={sub_sample_lengths.tolist() if isinstance(sub_sample_lengths, torch.Tensor) else sub_sample_lengths}"
        )
    
    worker_key = feat.get('_worker_key', None)
    state_snapshot = feat.get('_state_snapshot', None)
    
    image_flags = feat['image_flags']
    if not isinstance(image_flags, torch.Tensor):
        image_flags = torch.tensor(image_flags)

    pos = feat['position_ids'].unsqueeze(0)  # [L] -> [1, L]

    result = dict(
        input_ids=feat['input_ids'].unsqueeze(0),
        labels=feat['labels'].unsqueeze(0),
        attention_mask=None,
        position_ids=pos,
        pixel_values=feat['pixel_values'],
        image_flags=image_flags,
        sub_sample_lengths=[sub_sample_lengths],
    )

    if 'image_grid_hws' in feat:
        grid = feat['image_grid_hws']
        if isinstance(grid, np.ndarray):
            grid = torch.from_numpy(grid)
        result['image_grid_hws'] = grid
    
    if worker_key is not None:
        result['_worker_key'] = worker_key
    if state_snapshot is not None:
        result['_state_snapshot'] = state_snapshot
        
    return result


class PackedCollatorMTP:
    """Pickle-able collator class for MTP packing."""
    
    def __init__(self, pad_id: int = 0, dataset: StreamPackedDatasetMTP = None):
        self.pad_id = pad_id
        self.dataset = dataset
    
    def __call__(self, features):
        return packed_collate_fn_mtp(features, dataset=self.dataset)


class StateAwareDataLoader:
    """Wrapper around DataLoader to capture state snapshots from worker processes."""
    def __init__(self, dataloader, dataset: StreamPackedDatasetMTP):
        self.dataloader = dataloader
        self.dataset = dataset

    def __iter__(self):
        for batch in self.dataloader:
            if '_worker_key' in batch and '_state_snapshot' in batch:
                worker_key = batch.pop('_worker_key')
                state_snapshot = batch.pop('_state_snapshot')
                
                if self.dataset is not None:
                    self.dataset._worker_states[worker_key] = state_snapshot

            batch.pop('_batch_idx', None)
            
            yield batch

    def __len__(self):
        return len(self.dataloader)


class DataloaderStateCallback(TrainerCallback):
    """Callback to save dataloader state."""
    
    def __init__(self, train_dataset: StreamPackedDatasetMTP):
        self.train_dataset = train_dataset
    
    def on_save(self, args, state, control, **kwargs):
        if not hasattr(self.train_dataset, 'state_dict'):
            return control
        
        checkpoint_folder = f"checkpoint-{state.global_step}"
        output_dir = os.path.join(args.output_dir, checkpoint_folder)
        rank = get_rank()
        state_path = os.path.join(output_dir, f"dataloader_state_rank{rank}.pt")
        temp_path = state_path + ".tmp"
        
        try:
            ds_state = self.train_dataset.state_dict()
            
            if rank == 0:
                total_batches = sum(
                    ws.get('batches_produced', 0) 
                    for ws in ds_state.get('worker_states', {}).values()
                )
                total_samples = sum(
                    ws.get('samples_produced', 0) 
                    for ws in ds_state.get('worker_states', {}).values()
                )
                logger.info(f"Saving dataloader state: total_batches={total_batches}")
            
            torch.save(ds_state, temp_path)
            os.replace(temp_path, state_path)
            
            if rank == 0:
                logger.info(f"Saved dataloader state to {state_path}")
                
        except Exception as e:
            logger.error(f"Rank {rank}: Failed to save dataloader state: {e}")
            traceback.print_exc()
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
        
        return control


class StreamPackingMTPTrainer(Trainer):
    """Trainer with StreamPackedDatasetMTP support."""
    
    def __init__(self, *args, sample_log_interval: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_samples = 0
        self._sample_log_interval = sample_log_interval
        self._start_step = None  # 记录开始的step，用于resume时正确计算平均值
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        # 记录开始的step（用于resume时正确计算平均值）
        if self._start_step is None:
            self._start_step = self.state.global_step
        
        # Count samples in current batch (通过 sub_sample_lengths 获取样本数)
        if 'sub_sample_lengths' in inputs:
            sub_sample_lengths = inputs['sub_sample_lengths']
            if isinstance(sub_sample_lengths, list):
                # sub_sample_lengths 是 list of tensors，每个tensor的长度是样本数
                num_samples = sum(len(ssl) for ssl in sub_sample_lengths)
            elif isinstance(sub_sample_lengths, torch.Tensor):
                num_samples = sub_sample_lengths.size(0)
            else:
                num_samples = 1
            self._total_samples += int(num_samples)
        
        # Log sample count every N steps (rank 0 only)
        if self.state.global_step > 0 and self.state.global_step % self._sample_log_interval == 0:
            if get_rank() == 0:
                steps_since_start = self.state.global_step - self._start_step
                if steps_since_start > 0:
                    avg_samples_per_step = self._total_samples / steps_since_start
                    logger.info(f"[SampleStats] Step {self.state.global_step}: "
                               f"Total samples (this run) = {self._total_samples}, "
                               f"Avg samples/step = {avg_samples_per_step:.2f}")
        
        return super().training_step(model, inputs, num_items_in_batch)
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        from torch.utils.data import DataLoader
        
        pad_id = 0
        if self.processing_class is not None and hasattr(self.processing_class, 'tokenizer'):
            pad_id = self.processing_class.tokenizer.pad_token_id or 0
        
        collate_fn = PackedCollatorMTP(pad_id=pad_id, dataset=self.train_dataset)
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=collate_fn,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=self.args.dataloader_prefetch_factor if self.args.dataloader_num_workers > 0 else None,
        )
        
        return StateAwareDataLoader(dataloader, self.train_dataset)


def build_stream_packed_dataset_mtp(
    model_args,
    data_args, 
    processor, 
    base_seed: int = 42
) -> StreamPackedDatasetMTP:
    """Build StreamPackedDatasetMTP."""
    ds_collections = json.loads(open(data_args.meta_path).read())
    
    datasets = []
    dataset_weights = []
    
    for ds_name, meta in ds_collections.items():
        repeat_time = meta.get('repeat_time', 1)
        try:
            ds = LazySupervisedDatasetMTP(
                ds_name, meta, processor,
                block_size=model_args.block_size,
                repeat_time=repeat_time,
                target_fps=data_args.target_fps,
                max_frames=data_args.max_frames,
                video_total_pixels=data_args.video_total_pixels,
            )
            
            if len(ds) == 0:
                logger.warning(f'Dataset {ds_name} is empty, skipping.')
                continue
            
            datasets.append(ds)
            
            weight = repeat_time * len(ds) if repeat_time >= 1 else len(ds)
            dataset_weights.append(weight)
            
            logger.info(f'Added dataset: {ds_name}, length={len(ds)}, '
                       f'repeat_time={repeat_time}, weight={weight:.0f}')
            
        except Exception as e:
            traceback.print_exc()
            logger.error(f'Error loading dataset {ds_name}: {e}')
            raise

    if len(datasets) == 0:
        raise ValueError("No valid datasets found!")

    buffer_size = getattr(data_args, 'packing_buffer_size', 32)

    return StreamPackedDatasetMTP(
        tokenizer=processor.tokenizer,
        data_rank=get_rank(),
        data_world_size=get_world_size(),
        datasets=datasets,
        dataset_weight=dataset_weights,
        max_num_tokens_per_sample=getattr(data_args, 'max_num_tokens_per_sample', 16384),
        max_num_tokens=getattr(data_args, 'max_num_tokens', data_args.max_seq_length),
        log_freq=10000,
        base_seed=base_seed,
        buffer_size=buffer_size,
    )


def main():
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if os.path.exists(osp.join(training_args.output_dir, 'done.txt')):
        logger.info("Training done (done.txt exists), exiting!")
        return
    
    # Patches - Note: NOT applying patch_packing_attention since we use custom attention
    # patch_packing_attention()  # Disabled - using MTP-specific attention
    replace_train_dataloader()
    replace_train_sampler()
    
    # Logging setup
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, '
        f'n_gpu: {training_args.n_gpu}, distributed: {bool(training_args.local_rank != -1)}, '
        f'fp16: {training_args.fp16}'
    )
    logger.info(f'Training parameters: {training_args}')

    # Checkpoint detection
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint_guard(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f'Checkpoint detected at {last_checkpoint}, resuming training.')
            
    set_seed(training_args.seed)
    
    # Load model and tokenizer
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    num_new_tokens = tokenizer.add_tokens(special_tokens_list + number_tokens_list, special_tokens=True)
    
    if len(tokenizer.encode("assistant")) > 1:
        tokenizer.add_tokens(["assistant"], special_tokens=False)
        num_new_tokens += 1
        
    image_token_index = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    text_mask_token_id = tokenizer.convert_tokens_to_ids(TEXT_MASK_TOKEN)
    null_token_id = tokenizer.convert_tokens_to_ids(NULL_TOKEN)
    box_start_token_id = tokenizer.convert_tokens_to_ids(BOX_START_TOKEN)
    box_end_token_id = tokenizer.convert_tokens_to_ids(BOX_END_TOKEN)
    ref_start_token_id = tokenizer.convert_tokens_to_ids(REF_START_TOKEN)
    ref_end_token_id = tokenizer.convert_tokens_to_ids(REF_END_TOKEN)
    coord_start_token_id = tokenizer.convert_tokens_to_ids(number_tokens_list[0])
    coord_end_token_id = tokenizer.convert_tokens_to_ids(number_tokens_list[-1])
    none_token_ids = tokenizer.encode("none", add_special_tokens=False)
    none_token_id = none_token_ids[0] if len(none_token_ids) == 1 else 4064
    
    if model_args.model_name_or_path is not None:
        # ===== LocateAnything (MoonVit + Qwen2/Qwen3) Loading Path =====
        logger.info('Loading LocateAnythingForConditionalGeneration...')
        config = LocateAnythingConfig.from_pretrained(model_args.model_name_or_path)
        config._attn_implementation = model_args.attn_implementation
        config._attn_implementation_autoset = False
        config.text_config._attn_implementation = model_args.attn_implementation
        config.text_config._attn_implementation_autoset = False
        config.vision_config._attn_implementation = 'flash_attention_2'
        config.vision_config._attn_implementation_autoset = False
        logger.info(f'Text attn: {model_args.attn_implementation}, Vision attn: flash_attention_2')

        config.image_token_index = image_token_index
        config.text_config.block_size = int(model_args.block_size)
        config.text_config.causal_attn = model_args.causal_attn
        config.text_config.text_mask_token_id = text_mask_token_id
        config.text_config.null_token_id = null_token_id
        config.box_start_token_id = box_start_token_id
        config.box_end_token_id = box_end_token_id
        config.coord_start_token_id = coord_start_token_id
        config.coord_end_token_id = coord_end_token_id
        config.ref_start_token_id = ref_start_token_id
        config.ref_end_token_id = ref_end_token_id
        config.none_token_id = none_token_id

        model = LocateAnythingForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, 
            torch_dtype=torch.bfloat16, config=config, 
            attn_implementation=model_args.attn_implementation
        )
            
        model.text_mask_token_id = text_mask_token_id
        model.language_model.block_size = int(model_args.block_size)
        model.language_model.causal_attn = model_args.causal_attn
        model.language_model.training = True
        
        try:
            processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=True)
            processor.tokenizer = tokenizer
        except Exception as e:
            logger.warning(f'AutoProcessor failed ({e}), building processor from local configs...')
            chat_template_data = load_config(model_args.chat_template_path)
            processor_config = load_config(model_args.processor_config_path)
            preprocessor_config = load_config(model_args.preprocessor_config_path)
            image_processor = LocateAnythingImageProcessor(**preprocessor_config)
            processor_config["chat_template"] = chat_template_data["chat_template"]
            processor = LocateAnythingProcessor(tokenizer=tokenizer, image_processor=image_processor, **processor_config)
    else:
        logger.info(f"Loading vision backbone from {model_args.vision_path}")
        vision_config = AutoConfig.from_pretrained(model_args.vision_path, trust_remote_code=True)

        if vision_config.model_type == 'moonvit':
            logger.info('Loading MoonVit...')
            vision_config._attn_implementation = 'flash_attention_2'
            vision_model = MoonVitPretrainedModel.from_pretrained(
                model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
        else:
            raise ValueError(f"Unsupported vision model type: {vision_config.model_type}")
            
        logger.info('Loading LLM...')
        text_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        text_config._attn_implementation = 'magi'

        llm = AutoModelForCausalLM.from_pretrained(
            model_args.llm_path, torch_dtype=torch.bfloat16,
            config=text_config, trust_remote_code=True)
        
        locateanything_config = LocateAnythingConfig(
            vision_config.to_dict(), text_config.to_dict(), 
            image_token_index=image_token_index, 
            mlp_connector_layers=model_args.mlp_connector_layers)
        locateanything_config._attn_implementation = 'magi'
        model = LocateAnythingForConditionalGeneration(locateanything_config, vision_model, llm)

        chat_template_data = load_config(model_args.chat_template_path)
        processor_config = load_config(model_args.processor_config_path)
        preprocessor_config = load_config(model_args.preprocessor_config_path)
        image_processor = LocateAnythingImageProcessor(**preprocessor_config)
        processor_config["chat_template"] = chat_template_data["chat_template"]
        processor = LocateAnythingProcessor(tokenizer=tokenizer, image_processor=image_processor, **processor_config)
        
    model.neftune_alpha = data_args.neftune_alpha
    
    # Enable packing mode for stream packing (works for both pretrained and scratch models)
    model.language_model.model.is_packing_mode = True
    
    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        model.config.text_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
        dist.barrier()

    model.language_model.config.use_cache = False

    if model_args.grad_checkpoint:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    logger.info("model init done")
    
    # Sequence parallelism
    if data_args.sequence_parallel_degree > 1:
        set_pg_manager(model, data_args.sequence_parallel_degree)
        logger.info(f'Sequence parallelism enabled: SP={data_args.sequence_parallel_degree}')

    # Multi-node info
    hostnames = [None] * dist.get_world_size()
    if get_pg_manager() is not None:
        local_info = f"sp_rank: {get_pg_manager().sequence_parallel_rank}, host: {socket.gethostname()}"
    else:
        local_info = f"sp_rank: None, host: {socket.gethostname()}"
    dist.all_gather_object(hostnames, local_info)
    
    if dist.get_rank() == 0:
        for i, info in enumerate(hostnames):
            logger.info(f"global rank[{i}]: {info}")

    # Build dataset
    logger.info("Building stream packed MTP dataset...")
    t_start = time.time()
    train_dataset = build_stream_packed_dataset_mtp(model_args, data_args, processor, base_seed=training_args.seed)
    logger.info(f"Dataset built in {time.time() - t_start:.2f}s")

    # Freeze params
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # Verify parameter order consistency across all ranks (critical for ZeRO-3)
    param_names = [name for name, param in model.named_parameters()]
    param_names_list = [None] * dist.get_world_size()
    dist.all_gather_object(param_names_list, param_names)
    
    if dist.get_rank() == 0:
        logger.info("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"  {name}")
        
        # Verify all ranks have the same parameter names
        for rank, names in enumerate(param_names_list):
            if names != param_names:
                logger.warning(f"Rank {rank} has different parameter order! This may cause ZeRO-3 errors.")
                logger.warning(f"Rank 0 has {len(param_names)} params, Rank {rank} has {len(names)} params")
    
    # Critical: Synchronize all ranks before DeepSpeed initialization
    # This ensures parameter order consistency across ranks for ZeRO-3
    dist.barrier()
    logger.info(f"Rank {dist.get_rank()}: Model initialization synchronized across all ranks")

    set_seed(training_args.seed)

    if model_args.lr_scale is not None:
        training_args.lr_scale = model_args.lr_scale
        replace_create_optimizer_with_various_lr()

    # Callbacks
    my_callbacks = []
    if model_args.save_every_n_hours > 0:
        my_callbacks.append(SaveCheckpointCallback(
            initial_interval_hours=model_args.save_every_n_hours, save_interval_minutes=5))
    my_callbacks.append(MemoryLoggerCallback())
    my_callbacks.append(DataloaderStateCallback(train_dataset))
    my_callbacks.append(MilestoneCheckpointCallback(milestone_interval=2000))
    
    CustomTrainer = StreamPackingMTPTrainer

    assert processor is not None, "Processor is required"
    
    collate_fn = PackedCollatorMTP(pad_id=processor.tokenizer.pad_token_id, dataset=train_dataset)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        data_collator=collate_fn,
        callbacks=my_callbacks,
        processing_class=processor,
        sample_log_interval=getattr(data_args, 'sample_log_interval', 100),
    )

    # Training
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        
        if checkpoint is not None:
            training_args.ignore_data_skip = True
            logger.info("Enabled ignore_data_skip=True for stateful dataloader resume.")
            
            rank = get_rank()
            dataloader_state_path = os.path.join(checkpoint, f"dataloader_state_rank{rank}.pt")
            
            if os.path.exists(dataloader_state_path):
                try:
                    dataloader_state = torch.load(dataloader_state_path, weights_only=False)
                    train_dataset.load_state_dict(dataloader_state)
                    logger.info(f"Rank {rank}: Loaded dataloader state from {dataloader_state_path}")
                except Exception as e:
                    logger.warning(f"Rank {rank}: Failed to load dataloader state: {e}")
                    traceback.print_exc()
            else:
                logger.warning(f"Rank {rank}: No dataloader state found at {dataloader_state_path}")

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        if get_rank() == 0:
            output_dir = training_args.output_dir

            locany_utils_src = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'utils', 'locany')
            skip_files = {'config.json', 'README.md', '__init__.py', '__pycache__'}
            if osp.isdir(locany_utils_src):
                for file in os.listdir(locany_utils_src):
                    if file in skip_files or file.startswith('__'):
                        continue
                    src_file = osp.join(locany_utils_src, file)
                    dst_file = osp.join(output_dir, file)
                    if osp.isfile(src_file):
                        shutil.copy2(src_file, dst_file)
                logger.info(f"Copied inference files from {locany_utils_src} to {output_dir}")

            config_path = osp.join(output_dir, 'config.json')
            if osp.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                config_data['auto_map'] = {
                    "AutoConfig": "configuration_locateanything.LocateAnythingConfig",
                    "AutoModel": "modeling_locateanything.LocateAnythingForConditionalGeneration",
                    "AutoModelForCausalLM": "modeling_locateanything.LocateAnythingForConditionalGeneration",
                    "AutoImageProcessor": "image_processing_locateanything.LocateAnythingImageProcessor",
                    "AutoProcessor": "processing_locateanything.LocateAnythingProcessor",
                }
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                    f.write('\n')
                logger.info("Updated config.json with auto_map")

        metrics = train_result.metrics
        metrics['train_samples'] = 'streaming'

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
        
    with open(osp.join(training_args.output_dir, 'done.txt'), 'w') as f:
        f.write('done: ' + time.ctime())


if __name__ == '__main__':
    main()
