# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# This file is adopted from the InternVL project
# (https://github.com/OpenGVLab/InternVL), licensed under the MIT License.
#
# Modifications © 2026 NVIDIA CORPORATION & AFFILIATES, licensed under
# the Apache License, Version 2.0.
#
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License
# --------------------------------------------------------

"""
Eagle Embodied Finetuning with Stream (Online) Packing - Perfect Resume Version
With High-Efficiency Buffer Strategy (Best-Fit & Big-Rocks-First)

Key Improvements:
1. [Buffer Strategy] Added a buffer to store samples that don't fit in current batch.
2. [Best-Fit] Try to fill remaining space in current batch using best-fitting samples from buffer.
3. [Big-Rocks-First] When starting a new batch, pick the largest sample from buffer first.
4. [Resume] State includes buffer content, ensuring bit-wise identical resume.

"""

import os
import os.path as osp
import copy
import hashlib
import logging
import math
import random
import sys
import warnings
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field

import json
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
    patch_packing_attention,
    replace_train_dataloader,
    replace_train_sampler
)
from eaglevl.model.eagle_embodied.modeling_eagle_embodied import EagleEmbodiedForConditionalGeneration
from eaglevl.model.eagle_embodied.configuration_eagle_embodied import EagleEmbodiedConfig
# The VFlash drafter binds the Hugging Face Qwen3 MLP/RMSNorm classes when it is
# imported, so it must be imported before replace_liger_fused_ops() installs
# the Liger kernels for the target.
from eaglevl.model.vflash.modeling_dflash import (
    VFLASH_NUM_DRAFT_LAYERS,
    DFlashDraftModel,
    _compiled_create_block_mask,
    _compiled_flex_attention,
)
from eaglevl.model.vflash.binding import (
    VFLASH_BINDING_VERSION,
    assert_hf_native_drafter,
    build_vflash_draft_config,
    stable_json_sha256,
    tokenizer_fingerprint,
    validate_vflash_draft_binding,
)
from eaglevl.model.vflash.physical_atoms import (
    TEXT_VFLASH_BLOCK_SIZE,
    physical_atom_config,
    validate_no_object_text,
    validate_oof_loss_scale,
    validate_physical_atom_sequence,
    validate_physical_atom_tokenizer,
)
from eaglevl.utils.moonvit.processing_eagle_embodied import EagleEmbodiedProcessor
from eaglevl.utils.moonvit.image_processing_eagle_embodied import EagleEmbodiedImageProcessor
from eaglevl.sp_utils import set_pg_manager, get_pg_manager
from eaglevl.train.constants import (eagle_embodied_special_tokens_list, IMG_CONTEXT_TOKEN, number_tokens_list,
                                    vflash_special_tokens_list)
from eaglevl.train.arguments import ModelArguments, DataTrainingArguments
from eaglevl.train.long_context import configure_text_config_for_long_context
from eaglevl.train.jsonl_index import (
    IndexedJsonlReader,
    MappedIndexSequence,
    build_active_index,
    prepare_jsonl_indices,
)
from eaglevl.train.packing_balance import aligned_quantile_order, pack_balance_key
from eaglevl.train.sequence_parallel import IGNORE_INDEX
from eaglevl.train.trainer_monkey_patch import replace_create_optimizer_with_various_lr
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer,
                          TrainingArguments as HFTrainingArguments,
                          set_seed, AutoProcessor)
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
from eaglevl.train.tools import (SaveCheckpointCallback, MemoryLoggerCallback,
                                  get_last_checkpoint_guard, load_config,
                                  process_multimodal_sample, draw_mark,
                                  build_local_processor)
from eaglevl.train.checkpoint_export import export_eagle_embodied_checkpoint
from transformers import TrainerCallback
from eaglevl.train.augmentation import apply_resize_augmentation, VisualAugmenter
from eaglevl.train.vflash.drafter_packing import (
    VFlashPackedCollator,
    labeled_response_ranges,
    resolve_vflash_sample_schedule,
)
from eaglevl.train.vflash.drafter_training import (
    EagleEmbodiedVFlashTrainingModel,
    build_vflash_trainer,
    distributed_target_weight_fingerprint,
)
from eaglevl.train.vflash.loss_weighting import apply_oof_loss_scale, build_atom_label_weights
from dotenv import load_dotenv
load_dotenv()
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class TrainingArguments(HFTrainingArguments):
    """Expose options hidden by some Transformers releases to HfArgumentParser."""

    overwrite_output_dir: bool = field(
        default=False,
        metadata={'help': 'Overwrite the content of the output directory.'},
    )
    group_by_length: bool = field(
        default=False,
        metadata={'help': 'Group samples of roughly the same length when sampling.'},
    )


if version.parse(torch.__version__) >= version.parse("2.4.0"):
    torch.serialization.add_safe_globals(
        [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))])


# ============ Patch ============
replace_liger_fused_ops()
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


class LazySupervisedDataset(Dataset):
    """Lazy-loading dataset with index-based access.

    The VFlash options are off by default:
    - ``physical_tokens`` validates the physical atoms and no-object text of
      every assistant response.
    - ``emit_label_weights`` adds FP32 per-token loss weights from the row's
      ``atom_loss_masks``; with ``oof_loss_scale`` set, ``<oof>`` labels are
      scaled by it as well.
    - ``strict_data`` aborts on any data error instead of skipping the row.
    - ``visual_aug`` applies the photometric augmentation that the recipe
      entry's ``visual_aug`` settings describe.
    """

    def __init__(self,
                 ds_name: str,
                 meta: dict,
                 processor,
                 repeat_time: float = 1,
                 max_frames: int = 16,
                 target_fps: int = 2,
                 video_total_pixels: int = 32000 * 28 * 28 * 0.9,
                 auto_thinking_handler: bool = False,
                 jsonl_index_paths: Optional[List[str]] = None,
                 jsonl_row_counts: Optional[List[int]] = None,
                 active_index_path: Optional[str] = None,
                 active_index_dtype: Optional[np.dtype] = None,
                 physical_tokens=None,
                 emit_label_weights: bool = False,
                 oof_loss_scale: Optional[float] = None,
                 strict_data: bool = False,
                 visual_aug: bool = False):
        super().__init__()
        self.ds_name = ds_name
        self.processor = processor
        self.max_length = self.processor.tokenizer.model_max_length
        self.repeat_time = repeat_time
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.video_total_pixels = video_total_pixels
        self.data_augment = meta.get("data_augment", False)
        self.auto_thinking_handler = auto_thinking_handler
        self.video_reader_backend = meta.get("video_reader_backend", None)

        ann_paths = meta["annotation"]
        if not isinstance(ann_paths, (list, tuple)):
            ann_paths = [ann_paths]
        self.root = meta.get("root", "")

        should_log_dataset = get_rank() == 0
        if should_log_dataset:
            logger.info(f"[Dataset] {self.ds_name} Opening JSONL indexes...")
        start_time = time.time()
        if jsonl_index_paths is None:
            raise ValueError(f"Missing prepared JSONL indexes for dataset {self.ds_name}")
        self.lazy_loader = IndexedJsonlReader(
            ann_paths, jsonl_index_paths, row_counts=jsonl_row_counts
        )
        if should_log_dataset:
            logger.info(
                f"[Dataset] {self.ds_name} JSONL indexes opened in "
                f"{time.time() - start_time:.2f}s."
            )

        original_num_rows = len(self.lazy_loader)
        if should_log_dataset:
            logger.info(f"[Dataset] {self.ds_name} Found {original_num_rows} samples.")
        self.active_indices = range(original_num_rows)

        if repeat_time < 1:
            if original_num_rows > 0:
                partial_len = int(original_num_rows * repeat_time)
                if partial_len > 0:
                    if active_index_path is not None and active_index_dtype is not None:
                        self.active_indices = MappedIndexSequence(
                            active_index_path, active_index_dtype, length=partial_len
                        )
                    else:
                        sampled_indices = random.Random(10086).sample(
                            range(original_num_rows), partial_len
                        )
                        sampled_indices.sort()
                        dtype = np.uint32 if original_num_rows <= np.iinfo(np.uint32).max else np.uint64
                        self.active_indices = np.asarray(sampled_indices, dtype=dtype)
                    if should_log_dataset:
                        logger.info(
                            f"[Dataset] {self.ds_name} Downsampled to "
                            f"{len(self.active_indices)} samples."
                        )
                else:
                    self.active_indices = range(0)

        self._length = len(self.active_indices)
        self.visual_augmenter = VisualAugmenter.from_recipe(meta) if visual_aug else None
        self.physical_tokens = physical_tokens
        self.emit_label_weights = bool(emit_label_weights)
        self.oof_loss_scale = oof_loss_scale
        self.strict_data = bool(strict_data)
        if should_log_dataset and self.visual_augmenter is not None:
            logger.info('[Dataset] %s visual_aug=%s', self.ds_name, self.visual_augmenter.cfg)

        self.bad_indices = set()
        self.bad_sample_reasons: Dict[int, str] = {}
        self.failure_counts: Dict[int, int] = {}

    def __len__(self):
        return self._length

    def _sample_reference(self, raw_data_item: Optional[dict]) -> str:
        if not isinstance(raw_data_item, dict):
            return "<unknown>"
        for key in ("video", "image", "image_url", "id", "sample_id", "question_id"):
            value = raw_data_item.get(key)
            if value:
                return str(value)
        return "<unknown>"

    def _is_transient_system_error(self, exc: Exception, tb: str) -> bool:
        """Transient system-level resource errors — the sample itself is not bad."""
        error_text = f"{type(exc).__name__}: {exc}\n{tb}"
        transient_signals = (
            "Resource temporarily unavailable",
            "BlockingIOError",
            "Errno 11",
            "Cannot allocate memory",
            "Too many open files",
        )
        return any(sig in error_text for sig in transient_signals)

    def _is_permanent_sample_error(self, exc: Exception, tb: str) -> bool:
        if self._is_transient_system_error(exc, tb):
            return False
        if isinstance(exc, (TimeoutError, MemoryError)):
            return True
        error_text = f"{type(exc).__name__}: {exc}\n{tb}"
        permanent_error_signals = (
            "All video backends failed",
            "Invalid data found when processing input",
            "Error splitting the input into NAL units",
            "Invalid NAL unit",
            "partial file",
            "avcodec_send_packet",
            "cannot identify image file",
            "No valid labels for training",
            "Video reading timed out",
            "Video file too large",
            "moov atom not found",
            "Video has 0 decodable frames",
            "Video has 0 frames after sampling",
            "exceeding",
            "Unable to allocate",
        )
        return any(signal in error_text for signal in permanent_error_signals)

    def _mark_bad_sample(self, real_idx: int, raw_data_item: Optional[dict], reason: str):
        if real_idx in self.bad_indices:
            return
        self.bad_indices.add(real_idx)
        self.bad_sample_reasons[real_idx] = reason
        logger.warning(
            f"[{self.ds_name}] permanently skip bad sample idx {real_idx}: "
            f"{self._sample_reference(raw_data_item)}; reason={reason}"
        )

    def get_targets_flag(self, input_ids: torch.Tensor) -> torch.Tensor:
        tokenizer = self.processor.tokenizer
        targets_flag = torch.zeros_like(input_ids)
        start_header_idxs = torch.where(
            input_ids == tokenizer.convert_tokens_to_ids("<|im_start|>")
        )[0]
        assistant_idxs = torch.where(
            input_ids == tokenizer.convert_tokens_to_ids("assistant")
        )[0]
        eot_idxs = torch.where(
            input_ids == tokenizer.convert_tokens_to_ids("<|im_end|>")
        )[0]

        for assistant_idx in assistant_idxs:
            sets = list(set(start_header_idxs + 1))
            sets = [each.item() for each in sets]
            if assistant_idx.item() in sets:
                st = assistant_idx + 1  # assistant\n
                # Find the first eot_idx after the assistant token
                found_eot = False
                for eot_idx in eot_idxs:
                    if eot_idx > st:
                        targets_flag[st + 1: eot_idx + 1] = 1
                        found_eot = True
                        break
                # No eot_idx (conversation truncated): mark everything after the assistant token as targets
                if not found_eot:
                    targets_flag[st + 1:] = 1
        targets = input_ids.clone()

        assert targets_flag.sum() > 0, f"No valid labels for training, skip sample in {self.ds_name}"
        targets[targets_flag == 0] = IGNORE_TOKEN_ID
        return targets

    def multi_modal_get_item(self, messages: list, raw_data_item: dict) -> Dict[str, torch.Tensor]:
        if self.physical_tokens is not None:
            for message in messages:
                role = str(message.get("role", message.get("from", ""))).lower()
                content = message.get("content", message.get("value"))
                if role in {"assistant", "gpt"} and isinstance(content, str):
                    validate_no_object_text(content)
        message_text = self.processor.py_apply_chat_template(messages, tokenize=False)
        image_inputs, video_inputs = self.processor.process_vision_info(messages, video_reader_backend=self.video_reader_backend)

        if image_inputs is not None:
            image_inputs = draw_mark(image_inputs, raw_data_item)
            image_inputs = [
                apply_resize_augmentation(
                    img, data_augment=self.data_augment,
                    min_long_edge=640, max_long_edge=2048, augment_prob=0.5
                ) for img in image_inputs
            ]
        if self.visual_augmenter is not None:
            # Draws from the global random state seeded per sample in __getitem__.
            aug_plan = self.visual_augmenter.draw(random)
            if aug_plan is not None:
                if image_inputs is not None:
                    image_inputs = self.visual_augmenter.apply_images(image_inputs, aug_plan)
                if video_inputs is not None:
                    video_inputs = self.visual_augmenter.apply_videos(video_inputs, aug_plan)
        inputs = self.processor(
            text=message_text, images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=False, truncation=True,
            video_total_pixels=self.video_total_pixels
        )
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"]
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[0]

        if "pixel_values" not in inputs:
            pixel_values = torch.zeros((4, 3, 14, 14), dtype=torch.float32)
            image_flags = torch.tensor([0], dtype=torch.long)
            image_grid_hws = np.array([[2, 2]])
        else:
            pixel_values = inputs["pixel_values"]
            image_grid_hws = inputs["image_grid_hws"]
            image_flags = torch.tensor([len(inputs["image_grid_hws"])], dtype=torch.long)

        labels = self.get_targets_flag(input_ids)
        sample = dict(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask,
            image_flags=image_flags, pixel_values=pixel_values, image_grid_hws=image_grid_hws
        )
        if self.physical_tokens is not None:
            for response_start, response_end in labeled_response_ranges(labels):
                validate_physical_atom_sequence(
                    input_ids,
                    response_start,
                    response_end,
                    self.physical_tokens,
                )
        if self.emit_label_weights:
            # Dense trajectories mark occluded cells with 0 in atom_loss_masks.
            label_weights = build_atom_label_weights(
                input_ids,
                labels,
                raw_data_item.get("atom_loss_masks") if isinstance(raw_data_item, dict) else None,
                self.physical_tokens,
            )
            if self.oof_loss_scale is not None:
                label_weights = apply_oof_loss_scale(
                    label_weights,
                    labels,
                    oof_token_id=self.physical_tokens.oof_token_id,
                    oof_loss_scale=self.oof_loss_scale,
                )
            sample['label_weights'] = label_weights
        return sample

    def _make_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a minimal sample with all labels masked — contributes zero loss."""
        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id or 0
        dummy_len = 4
        input_ids = torch.full((dummy_len,), pad_id, dtype=torch.long)
        labels = torch.full((dummy_len,), IGNORE_TOKEN_ID, dtype=torch.long)
        attention_mask = torch.ones(dummy_len, dtype=torch.long)
        pixel_values = torch.zeros((4, 3, 14, 14), dtype=torch.float32)
        image_flags = torch.tensor([0], dtype=torch.long)
        image_grid_hws = np.array([[2, 2]])
        sample = dict(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask,
            image_flags=image_flags, pixel_values=pixel_values, image_grid_hws=image_grid_hws,
        )
        if self.emit_label_weights:
            sample['label_weights'] = torch.ones((dummy_len,), dtype=torch.float32)
        return sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._length == 0:
            raise RuntimeError(f"[{self.ds_name}] Dataset is empty")

        retry_count = 0
        current_idx = idx % self._length
        max_attempts = min(self._length, 1024)
        transient_budget = 10

        seed = int(idx + 10086)
        random.seed(seed)
        np.random.seed(seed)

        while retry_count < max_attempts:
            real_idx = int(self.active_indices[(current_idx + retry_count) % self._length])
            retry_count += 1
            if real_idx in self.bad_indices:
                continue

            raw_data_item = None
            try:
                raw_data_item = self.lazy_loader[real_idx]
                data_item = process_multimodal_sample(
                    raw_data_item, self.root, self.max_frames,
                    self.target_fps, self.video_total_pixels,
                    auto_thinking_handler=self.auto_thinking_handler
                )
                return self.multi_modal_get_item(data_item, raw_data_item)
            except Exception as e:
                tb = traceback.format_exc()
                sample_ref = self._sample_reference(raw_data_item)

                if self._is_transient_system_error(e, tb):
                    transient_budget -= 1
                    logger.warning(
                        f"[{self.ds_name}] Transient system error at idx {real_idx} "
                        f"(sample={sample_ref}, budget={transient_budget}): {e}"
                    )
                    if transient_budget > 0:
                        time.sleep(2.0)
                        retry_count -= 1
                    elif self.strict_data:
                        raise RuntimeError(
                            f'[{self.ds_name}] transient data error retry budget exhausted: {e}'
                        ) from e
                    continue

                if self.strict_data:
                    raise RuntimeError(
                        f'[{self.ds_name}] strict data check failed at row {real_idx}: {e}'
                    ) from e

                self.failure_counts[real_idx] = self.failure_counts.get(real_idx, 0) + 1
                logger.warning(
                    f"[{self.ds_name}] idx {real_idx} failed "
                    f"(sample={sample_ref}, failure_count={self.failure_counts[real_idx]}): {e}\n{tb}"
                )
                if self.failure_counts[real_idx] >= 2 or self._is_permanent_sample_error(e, tb):
                    self._mark_bad_sample(real_idx, raw_data_item, str(e))

        if len(self.bad_indices) >= self._length:
            raise RuntimeError(f"[{self.ds_name}] All samples are marked bad; cannot continue")

        logger.error(
            f"[{self.ds_name}] Failed to find a valid sample after {max_attempts} attempts "
            f"(bad_samples={len(self.bad_indices)}); returning dummy sample to keep training alive"
        )
        return self._make_dummy_sample()

    def get_sample_at_global_idx(self, global_idx: int, seed: int) -> Dict[str, torch.Tensor]:
        """Get a sample by global index (used to rebuild current_batch/buffer during resume)."""
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

    def __init__(self, dataset: LazySupervisedDataset, seed: int, start_global_idx: int = 0):
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
    def from_state_dict(cls, dataset: LazySupervisedDataset, state: dict) -> 'DeterministicIterator':
        return cls(dataset=dataset, seed=state['seed'], start_global_idx=state['global_idx'])


@dataclass
class WorkerState:
    """Complete state of a worker, including buffer state."""
    iterator_states: List[dict]           # State of each dataset iterator
    sample_rng_state: tuple               # State of the dataset-selection RNG
    samples_produced: int                 # Number of samples produced
    batches_produced: int                 # Number of batches produced
    current_batch_locations: List[Tuple[int, int]]  # Locations of the samples in the current batch
    buffer_locations: List[Tuple[int, int]]         # [NEW] Locations of the samples in the buffer
    ready_pack_locations: List[List[Tuple[int, int]]]
    packs_generated: int
    balance_windows_built: int

    def to_dict(self) -> dict:
        return {
            'iterator_states': self.iterator_states,
            'sample_rng_state': self.sample_rng_state,
            'samples_produced': self.samples_produced,
            'batches_produced': self.batches_produced,
            'current_batch_locations': self.current_batch_locations,
            'buffer_locations': self.buffer_locations, # [NEW]
            'ready_pack_locations': self.ready_pack_locations,
            'packs_generated': self.packs_generated,
            'balance_windows_built': self.balance_windows_built,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'WorkerState':
        # Reconstruct RNG state: random.Random.getstate() returns (version, tuple_of_625_ints, gauss_next)
        # torch.save/load converts tuples to lists, so we need to restore the proper structure
        raw_rng_state = d['sample_rng_state']
        if isinstance(raw_rng_state, (list, tuple)) and len(raw_rng_state) == 3:
            version, internal_state, gauss_next = raw_rng_state
            # The internal state must be a tuple of integers
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
            buffer_locations=d.get('buffer_locations', []), # [NEW]
            ready_pack_locations=d.get('ready_pack_locations', []),
            packs_generated=d.get('packs_generated', d.get('batches_produced', 0)),
            balance_windows_built=d.get('balance_windows_built', 0),
        )


class StreamPackedDataset(IterableDataset):
    """
    Online packing IterableDataset with Buffer and perfect stateful resume support.

    Optional VFlash extensions, all off by default:
    - ``max_samples_per_pack`` closes every pack at exactly that many samples
      (the token budget must then fit them).
    - ``extra_packing_config`` records objective settings that a resume must
      match, next to the packing configuration.
    - ``emit_pack_seed`` attaches ``_pack_seed`` to every pack, a deterministic
      seed derived from the worker and the pack's sample locations.
    - ``strict_data`` aborts on any data error instead of skipping the sample.
    """

    def __init__(
        self,
        tokenizer,
        data_rank: int,
        data_world_size: int,
        datasets: List[LazySupervisedDataset],
        dataset_weight: List[float] = None,
        max_num_tokens_per_sample: int = 16384,
        max_num_tokens: int = 36864,
        log_freq: int = 10000,
        base_seed: int = 42,
        buffer_size: int = 32, # [NEW] Buffer capacity
        balance_media_tokens: bool = False,
        balance_window_size: int = 8,
        balance_seed: Optional[int] = None,
        dataset_fingerprint: Optional[str] = None,
        max_samples_per_pack: Optional[int] = None,
        extra_packing_config: Optional[dict] = None,
        epoch_sample_count: Optional[int] = None,
        emit_pack_seed: bool = False,
        strict_data: bool = False,
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
        self.buffer_size = buffer_size # [NEW]
        self.balance_media_tokens = bool(balance_media_tokens)
        self.balance_window_size = int(balance_window_size)
        if self.balance_window_size < 1:
            raise ValueError('balance_window_size must be >= 1')
        self.balance_seed = int(base_seed if balance_seed is None else balance_seed)
        self.media_token_id = int(tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN))
        self.dataset_fingerprint = dataset_fingerprint
        self.max_samples_per_pack = (
            None if max_samples_per_pack is None else int(max_samples_per_pack)
        )
        if self.max_samples_per_pack is not None and self.max_samples_per_pack < 1:
            raise ValueError('max_samples_per_pack must be >= 1 when provided')
        self.extra_packing_config = copy.deepcopy(extra_packing_config)
        self.epoch_sample_count = (
            None if epoch_sample_count is None else int(epoch_sample_count)
        )
        self.emit_pack_seed = bool(emit_pack_seed)
        self.strict_data = bool(strict_data)

        if dataset_weight is None:
            dataset_weight = [1] * len(datasets)
        total_weight = sum(dataset_weight)
        self.dataset_weight = [w / total_weight for w in dataset_weight]

        self._worker_states: Dict[str, dict] = {}
        self._resume_states: Dict[str, dict] = {}
        self._resume_state_version = 0
        self._runtime_num_workers: Optional[int] = None
        self._resume_num_workers: Optional[int] = None

        if get_rank() == 0:
            ds_info = '\n'.join([f'  {ds.ds_name}: weight={w*100:.2f}%, len={len(ds)}'
                                 for ds, w in zip(self.datasets, self.dataset_weight)])
            logger.info(f'StreamPackedDataset initialized:\n'
                       f'  max_num_tokens_per_sample={max_num_tokens_per_sample}\n'
                       f'  max_num_tokens={max_num_tokens}\n'
                       f'  buffer_size={buffer_size}\n'
                       f'  balance_media_tokens={self.balance_media_tokens}\n'
                       f'  balance_window_size={self.balance_window_size}\n'
                       f'  balance_seed={self.balance_seed}\n'
                       f'  base_seed={base_seed}\n'
                       f'  data_rank={data_rank}, data_world_size={data_world_size}\n'
                       f'Datasets:\n{ds_info}')

    def state_dict(self) -> dict:
        return {
            'worker_states': copy.deepcopy(self._worker_states),
            'base_seed': self.base_seed,
            'packing_config': self._packing_state_config(),
            'num_workers': self._runtime_num_workers,
            'version': 6,
        }

    def _packing_state_config(self) -> dict:
        config = {
            'dataset_fingerprint': self.dataset_fingerprint,
            'datasets': [(dataset.ds_name, len(dataset)) for dataset in self.datasets],
            'dataset_weight': self.dataset_weight,
            'data_rank': self.data_rank,
            'data_world_size': self.data_world_size,
            'max_num_tokens_per_sample': self.max_num_tokens_per_sample,
            'max_num_tokens': self.max_num_tokens,
            'buffer_size': self.buffer_size,
            'base_seed': self.base_seed,
            'balance_media_tokens': self.balance_media_tokens,
            'balance_window_size': self.balance_window_size,
            'balance_seed': self.balance_seed,
        }
        if (
            self.extra_packing_config is not None
            or self.max_samples_per_pack is not None
            or self.emit_pack_seed
        ):
            config.update(
                extra_packing_config=self.extra_packing_config,
                max_samples_per_pack=self.max_samples_per_pack,
                epoch_sample_count=self.epoch_sample_count,
                emit_pack_seed=self.emit_pack_seed,
            )
        return config

    def load_state_dict(self, state: dict):
        version = state.get('version', 1)
        self._resume_state_version = int(version)
        self._resume_num_workers = state.get('num_workers')
        if version < 3:
            logger.warning(f"Loading old state version {version}, perfect resume not available.")

        saved_config = state.get('packing_config')
        if saved_config is None:
            if version >= 6:
                raise ValueError('Dataloader state v6 is missing packing_config.')
            logger.warning(
                f'Loading dataloader state v{version} without a packing configuration; '
                'resume compatibility cannot be verified.'
            )
        elif saved_config != self._packing_state_config():
            raise ValueError(
                'Dataloader packing configuration changed since the checkpoint; refusing '
                f'a non-exact resume. saved={saved_config}, current={self._packing_state_config()}'
            )

        if 'worker_states' in state:
            self._resume_states = copy.deepcopy(state['worker_states'])
            # Keep every worker's last consumed baseline in the main process.
            # A checkpoint taken shortly after resume may not have yielded all
            # workers yet; their prior states must not disappear.
            self._worker_states = copy.deepcopy(state['worker_states'])
            if get_rank() == 0:
                logger.info(f"Loaded resume states for {len(self._resume_states)} workers")

    def _get_sample_length(self, sample: Optional[dict]) -> int:
        if sample is None:
            return 0
        return sample['input_ids'].size(0)

    def _merge_samples(self, batch: Optional[dict], sample: dict) -> dict:
        if batch is None:
            result = copy.copy(sample)
            result['data_index'] = torch.zeros_like(sample['input_ids'])
            result['_sample_count'] = 1
            return result

        if (
            self.max_samples_per_pack is not None
            and int(batch.get('_sample_count', 0)) >= self.max_samples_per_pack
        ):
            raise RuntimeError(
                'Packing attempted to exceed max_samples_per_pack='
                f'{self.max_samples_per_pack}'
            )

        new_data_index = torch.full_like(
            sample['input_ids'],
            fill_value=batch['data_index'][-1].item() + 1
        )

        result = {}
        for k in batch:
            if k == '_sample_count':
                result[k] = batch[k] + 1
            elif k == 'data_index':
                result[k] = torch.cat([batch[k], new_data_index])
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
        data_index = batch['data_index']
        unique_indices = data_index.unique(sorted=True)
        sub_sample_lengths = torch.tensor(
            [(data_index == idx).sum().item() for idx in unique_indices],
            dtype=torch.long
        )

        batch['sub_sample_lengths'] = sub_sample_lengths
        batch['attention_mask'] = data_index.clone()

        batch.pop('data_index', None)
        batch.pop('_sample_count', None)

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
            logger.info(f'[{worker_key}] Starting iteration with Buffer Strategy...')

        # ========== Initialize state ==========
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
        packs_generated = 0
        balance_windows_built = 0
        skipped_count = 0

        # Current batch state
        current_batch = None
        current_batch_locations: List[Tuple[int, int]] = []

        # Buffer state: List of (sample_dict, ds_idx, global_idx)
        buffer: List[Tuple[dict, int, int]] = []
        # Completed packs waiting to be yielded after media-cost alignment.
        ready_packs: List[Tuple[dict, List[Tuple[int, int]]]] = []

        # ========== Resume: Restore state ==========
        if worker_key in self._resume_states:
            saved = self._resume_states.pop(worker_key)
            try:
                ws = WorkerState.from_dict(saved)

                if is_main_log:
                    logger.info(f'[{worker_key}] Resuming...')

                # 1. Restore iterators
                for ds_idx, iter_state in enumerate(ws.iterator_states):
                    iterators[ds_idx] = DeterministicIterator.from_state_dict(
                        self.datasets[ds_idx], iter_state
                    )

                # 2. Restore RNG
                sample_rng.setstate(ws.sample_rng_state)
                samples_produced = ws.samples_produced
                batches_produced = ws.batches_produced
                packs_generated = ws.packs_generated
                balance_windows_built = ws.balance_windows_built

                # 3. Rebuild current_batch
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
                            if self._resume_state_version >= 5:
                                raise RuntimeError(
                                    f'Failed to restore current-pack sample {loc} exactly.'
                                ) from e
                            logger.warning(f'[{worker_key}] Failed to restore batch sample {loc}: {e}')

                # 4. Rebuild Buffer [NEW]
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
                            if self._resume_state_version >= 5:
                                raise RuntimeError(
                                    f'Failed to restore buffered sample {loc} exactly.'
                                ) from e
                            logger.warning(f'[{worker_key}] Failed to restore buffer sample {loc}: {e}')

                # 5. Rebuild completed packs that were generated during lookahead
                # but had not yet been yielded when the checkpoint was taken.
                for pack_locations in ws.ready_pack_locations:
                    restored_pack = None
                    restored_locations = []
                    for loc in pack_locations:
                        ds_idx, global_idx = loc
                        try:
                            sample = self.datasets[ds_idx].get_sample_at_global_idx(
                                global_idx, iterator_seeds[ds_idx]
                            )
                            restored_pack = self._merge_samples(restored_pack, sample)
                            restored_locations.append(loc)
                        except Exception as e:
                            if self._resume_state_version >= 5:
                                raise RuntimeError(
                                    f'Failed to restore ready-pack sample {loc} exactly.'
                                ) from e
                            logger.warning(f'[{worker_key}] Failed to restore ready-pack sample {loc}: {e}')
                    if restored_pack is not None:
                        ready_packs.append((self._finalize_batch(restored_pack), restored_locations))

                if is_main_log:
                    logger.info(f'[{worker_key}] Resume complete. Buffer size: {len(buffer)}')

            except Exception as e:
                logger.error(f'[{worker_key}] Failed to resume: {e}')
                traceback.print_exc()
                if self._resume_state_version >= 5:
                    raise RuntimeError(
                        f'[{worker_key}] Exact dataloader resume failed; refusing to restart the stream.'
                    ) from e
                current_batch = None
                current_batch_locations = []
                buffer = []
                ready_packs = []
                packs_generated = batches_produced
                balance_windows_built = 0

        # ========== Helper Functions ==========
        def build_state_snapshot() -> dict:
            return WorkerState(
                iterator_states=[it.state_dict() for it in iterators],
                sample_rng_state=sample_rng.getstate(),
                samples_produced=samples_produced,
                batches_produced=batches_produced,
                current_batch_locations=list(current_batch_locations),
                buffer_locations=[(b[1], b[2]) for b in buffer], # Save buffer locs
                ready_pack_locations=[list(locations) for _, locations in ready_packs],
                packs_generated=packs_generated,
                balance_windows_built=balance_windows_built,
            ).to_dict()

        def fetch_next_sample() -> Tuple[dict, int, int]:
            nonlocal samples_produced, skipped_count
            while True:
                ds_idx = sample_rng.choices(range(len(self.datasets)), weights=self.dataset_weight)[0]
                try:
                    sample, global_idx = next(iterators[ds_idx])
                    samples_produced += 1
                    if self._get_sample_length(sample) > self.max_num_tokens_per_sample:
                        if self.strict_data:
                            raise ValueError(
                                f'{self.datasets[ds_idx].ds_name}: sample length '
                                f'{self._get_sample_length(sample)} exceeds '
                                f'{self.max_num_tokens_per_sample}'
                            )
                        skipped_count += 1
                        continue
                    return sample, ds_idx, global_idx
                except StopIteration:
                    continue
                except Exception as e:
                    skipped_count += 1
                    logger.warning(f'[{worker_key}] Failed to fetch sample from {self.datasets[ds_idx].ds_name}: {e}')
                    if self.strict_data:
                        raise
                    continue

        def generate_next_pack() -> Tuple[dict, List[Tuple[int, int]]]:
            """Generate one completed pack and retain exact resume locations."""
            nonlocal current_batch, current_batch_locations, packs_generated

            while True:
                if (
                    current_batch is not None
                    and self.max_samples_per_pack is not None
                    and int(current_batch.get('_sample_count', 0))
                    >= self.max_samples_per_pack
                ):
                    # Close a full fixed-size pack before considering the budget.
                    output_locations = list(current_batch_locations)
                    output_batch = self._finalize_batch(current_batch)
                    current_batch = None
                    current_batch_locations = []

                    if buffer:
                        buffer.sort(
                            key=lambda item: self._get_sample_length(item[0]),
                            reverse=True,
                        )
                        sample, ds_idx, global_idx = buffer.pop(0)
                        current_batch = self._merge_samples(None, sample)
                        current_batch_locations = [(ds_idx, global_idx)]

                    packs_generated += 1
                    return output_batch, output_locations

                current_len = self._get_sample_length(current_batch)
                remaining_space = self.max_num_tokens - current_len

                best_fit_idx = -1
                max_fit_len = -1
                for i, (buf_sample, _, _) in enumerate(buffer):
                    sample_len = self._get_sample_length(buf_sample)
                    if sample_len <= remaining_space and sample_len > max_fit_len:
                        max_fit_len = sample_len
                        best_fit_idx = i

                if best_fit_idx != -1:
                    sample, ds_idx, global_idx = buffer.pop(best_fit_idx)
                    current_batch = self._merge_samples(current_batch, sample)
                    current_batch_locations.append((ds_idx, global_idx))
                    continue

                if len(buffer) < self.buffer_size:
                    new_sample, ds_idx, global_idx = fetch_next_sample()
                    if self._get_sample_length(new_sample) <= remaining_space:
                        current_batch = self._merge_samples(current_batch, new_sample)
                        current_batch_locations.append((ds_idx, global_idx))
                        continue
                    buffer.append((new_sample, ds_idx, global_idx))

                if current_batch is None:
                    if not buffer:
                        continue
                    buffer.sort(key=lambda item: self._get_sample_length(item[0]), reverse=True)
                    sample, ds_idx, global_idx = buffer.pop(0)
                    current_batch = self._merge_samples(None, sample)
                    current_batch_locations = [(ds_idx, global_idx)]
                    continue

                output_locations = list(current_batch_locations)
                output_batch = self._finalize_batch(current_batch)
                current_batch = None
                current_batch_locations = []

                if buffer:
                    buffer.sort(key=lambda item: self._get_sample_length(item[0]), reverse=True)
                    sample, ds_idx, global_idx = buffer.pop(0)
                    current_batch = self._merge_samples(None, sample)
                    current_batch_locations = [(ds_idx, global_idx)]

                packs_generated += 1
                return output_batch, output_locations

        # ========== Main Loop ==========
        while True:
            if not ready_packs:
                window_size = self.balance_window_size if self.balance_media_tokens else 1
                generated = [generate_next_pack() for _ in range(window_size)]

                if self.balance_media_tokens:
                    costs = [pack_balance_key(pack, self.media_token_id) for pack, _ in generated]
                    window_seed = (
                        self.balance_seed
                        + local_worker_id * 1000003
                        + balance_windows_built * 999983
                    )
                    order = aligned_quantile_order(costs, seed=window_seed)
                    ready_packs = [generated[index] for index in order]
                    balance_windows_built += 1
                else:
                    ready_packs = generated

            output_batch, output_locations = ready_packs.pop(0)
            batches_produced += 1

            # Iterator/buffer state already points beyond the complete lookahead
            # window. Persist the not-yet-yielded packs to make resume exact.
            state_snapshot = build_state_snapshot()
            if self.emit_pack_seed:
                # Depends only on the worker and the pack's sample locations, so
                # an exactly resumed stream reproduces it.
                pack_seed = (
                    int(self.base_seed) * 1000003
                    + int(global_worker_id) * 32452843
                ) & ((1 << 63) - 1)
                for dataset_index, global_index in output_locations:
                    pack_seed = (
                        pack_seed * 1000003
                        + int(dataset_index) * 9176
                        + int(global_index) * 6361
                        + 1447
                    ) & ((1 << 63) - 1)
                output_batch['_pack_seed'] = pack_seed
            output_batch['_worker_key'] = worker_key
            output_batch['_batch_idx'] = batches_produced
            output_batch['_state_snapshot'] = state_snapshot

            yield output_batch

            if is_main_log and batches_produced % self.log_freq == 0:
                output_len = self._get_sample_length(output_batch)
                media_tokens = pack_balance_key(output_batch, self.media_token_id)[0]
                packing_efficiency = output_len / self.max_num_tokens * 100
                logger.info(
                    f'batches={batches_produced}, samples={samples_produced}, '
                    f'buffer_len={len(buffer)}, ready_packs={len(ready_packs)}, '
                    f'media_tokens={media_tokens}, packing_eff={packing_efficiency:.1f}%'
                )


def packed_collate_fn(features: List[dict], dataset: Optional[StreamPackedDataset] = None) -> dict:
    """Collator: Process batch and update dataset state."""
    assert len(features) == 1, f"Expected batch_size=1 for packing, got {len(features)}"

    feat = features[0]

    # IMPORTANT: Do not remove metadata here. Pass it to the main process via the batch dict.
    # The StateAwareDataLoader will handle the state update in the main process.
    worker_key = feat.get('_worker_key', None)
    state_snapshot = feat.get('_state_snapshot', None)

    image_flags = feat['image_flags']
    if not isinstance(image_flags, torch.Tensor):
        image_flags = torch.tensor(image_flags)

    image_grid_hws = feat['image_grid_hws']
    if isinstance(image_grid_hws, np.ndarray):
        image_grid_hws = torch.from_numpy(image_grid_hws)

    result = dict(
        input_ids=feat['input_ids'],
        labels=feat['labels'],
        attention_mask=feat['attention_mask'].unsqueeze(0),
        pixel_values=feat['pixel_values'],
        image_flags=image_flags,
        image_grid_hws=image_grid_hws,
        sub_sample_lengths=[feat['sub_sample_lengths']],
    )
    if feat.get('label_weights') is not None:
        result['label_weights'] = feat['label_weights']

    if worker_key is not None:
        result['_worker_key'] = worker_key
    if state_snapshot is not None:
        result['_state_snapshot'] = state_snapshot

    return result


class PackedCollator:
    """Pickle-able collator class."""

    def __init__(self, pad_id: int = 0, dataset: StreamPackedDataset = None):
        self.pad_id = pad_id
        self.dataset = dataset

    def __call__(self, features):
        return packed_collate_fn(features, dataset=self.dataset)


class StateAwareDataLoader:
    """
    Wrapper around DataLoader to capture state snapshots from worker processes
    and update the dataset in the main process.
    """
    def __init__(self, dataloader, dataset: StreamPackedDataset):
        self.dataloader = dataloader
        self.dataset = dataset
        self._resume_offset_consumed = False
        if dataset is not None:
            saved_num_workers = dataset._resume_num_workers
            if saved_num_workers is not None and int(saved_num_workers) != dataloader.num_workers:
                raise ValueError(
                    f'Dataloader worker count changed across resume: saved={saved_num_workers}, '
                    f'current={dataloader.num_workers}.'
                )
            dataset._runtime_num_workers = dataloader.num_workers

    def _consume_batch_state(self, batch):
        if '_worker_key' in batch and '_state_snapshot' in batch:
            worker_key = batch.pop('_worker_key')
            state_snapshot = batch.pop('_state_snapshot')
            if self.dataset is not None:
                self.dataset._worker_states[worker_key] = state_snapshot
        return batch

    def __iter__(self):
        iterator = iter(self.dataloader)
        num_workers = self.dataloader.num_workers
        resume_offset = 0
        if (
            not self._resume_offset_consumed
            and num_workers > 1
            and self.dataset is not None
            and self.dataset._resume_states
        ):
            consumed_batches = sum(
                int(state.get('batches_produced', 0))
                for state in self.dataset._resume_states.values()
            )
            resume_offset = consumed_batches % num_workers
            if resume_offset:
                logger.info(
                    f'Rotating restored DataLoader worker order by {resume_offset} to preserve '
                    'the uninterrupted global batch sequence.'
                )
        self._resume_offset_consumed = True

        if resume_offset == 0:
            for batch in iterator:
                yield self._consume_batch_state(batch)
            return

        # PyTorch starts every fresh iterator at worker 0. Each restored worker
        # already points at its own next unseen pack, so rotate every raw worker
        # cycle to resume from the worker that followed the last consumed batch.
        while True:
            cycle = []
            try:
                for _ in range(num_workers):
                    cycle.append(next(iterator))
            except StopIteration:
                for batch in cycle:
                    yield self._consume_batch_state(batch)
                return
            for batch in cycle[resume_offset:] + cycle[:resume_offset]:
                yield self._consume_batch_state(batch)

    def __len__(self):
        return len(self.dataloader)


class DataloaderStateCallback(TrainerCallback):
    """Callback to save dataloader state."""

    def __init__(self, train_dataset: StreamPackedDataset):
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
                pending_samples = sum(
                    len(ws.get('current_batch_locations', []))
                    for ws in ds_state.get('worker_states', {}).values()
                )
                buffer_samples = sum(
                    len(ws.get('buffer_locations', []))
                    for ws in ds_state.get('worker_states', {}).values()
                )
                logger.info(f"Saving dataloader state: total_batches={total_batches}, "
                           f"pending_in_batch={pending_samples}, buffer_samples={buffer_samples}")

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
            raise RuntimeError(
                f'Rank {rank}: refusing to keep a checkpoint without exact dataloader state.'
            ) from e

        return control


class StreamPackingTrainer(Trainer):
    """Trainer with StreamPackedDataset support.

    With ``export_layout`` set (VFlash target training), every saved checkpoint
    also receives that remote-code layout from ``eaglevl.train.checkpoint_export``.
    A subclass with ``use_data_collator = True`` (the VFlash drafter trainer)
    collates packs with its ``data_collator`` instead of ``PackedCollator``.
    """

    use_data_collator = False

    def __init__(
        self,
        *args,
        sample_log_interval: int = 100,
        debug_assert_sp_batch_sync: bool = False,
        export_layout: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._total_samples = 0
        self._sample_log_interval = sample_log_interval
        self._start_step = None  # Step this run started at; keeps averages correct after resume
        self._debug_assert_sp_batch_sync = debug_assert_sp_batch_sync
        self._sp_batches_checked = 0
        self._export_layout = export_layout

    def training_step(self, model, inputs, num_items_in_batch=None):
        pg_manager = get_pg_manager()
        if self._debug_assert_sp_batch_sync and pg_manager is not None:
            hasher = hashlib.sha256()

            def update_tensor_digest(name, value, *, sampled=False):
                if value is None:
                    hasher.update(f'{name}:none'.encode())
                    return
                tensor = torch.as_tensor(value).detach().cpu().contiguous()
                hasher.update(f'{name}:{tuple(tensor.shape)}:{tensor.dtype}'.encode())
                if sampled and tensor.numel() > 1024:
                    indices = torch.linspace(
                        0,
                        tensor.numel() - 1,
                        steps=1024,
                        dtype=torch.long,
                    )
                    tensor = tensor.reshape(-1).index_select(0, indices).contiguous()
                hasher.update(tensor.numpy().tobytes())

            for key in ('input_ids', 'labels', 'attention_mask', 'image_flags', 'image_grid_hws'):
                update_tensor_digest(key, inputs.get(key))
            for index, lengths in enumerate(inputs.get('sub_sample_lengths', [])):
                update_tensor_digest(f'sub_sample_lengths[{index}]', lengths)
            update_tensor_digest('pixel_values_sample', inputs.get('pixel_values'), sampled=True)

            local_digest = torch.tensor(
                list(hasher.digest()),
                dtype=torch.uint8,
                device=self.args.device,
            )
            signatures = [torch.empty_like(local_digest) for _ in range(
                pg_manager.sequence_parallel_world_size
            )]
            dist.all_gather(
                signatures,
                local_digest,
                group=pg_manager.sequence_parallel_group,
            )
            if any(not torch.equal(signatures[0], signature) for signature in signatures[1:]):
                raise RuntimeError(
                    'Sequence-parallel ranks received different packed batches at local '
                    f'training step {self._sp_batches_checked}: '
                    f'{[bytes(signature.cpu().tolist()).hex() for signature in signatures]}'
                )
            if self._sp_batches_checked == 0 and dist.get_rank() == 0:
                labels = torch.as_tensor(inputs['labels'])
                lengths = inputs.get('sub_sample_lengths', [])
                logger.info(
                    'SP batch synchronization passed: tokens=%d, segments=%d, valid_labels=%d.',
                    labels.numel(),
                    sum(len(per_batch) for per_batch in lengths),
                    int((labels != IGNORE_INDEX).sum().item()),
                )
            self._sp_batches_checked += 1

        # Record the starting step (keeps averages correct after resume)
        if self._start_step is None:
            self._start_step = self.state.global_step

        # Count samples in current batch (from sub_sample_lengths)
        if 'sub_sample_lengths' in inputs:
            sub_sample_lengths = inputs['sub_sample_lengths']
            if isinstance(sub_sample_lengths, list):
                # sub_sample_lengths is a list of tensors; each length is a sample count
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

        if self.use_data_collator:
            collate_fn = self.data_collator
        else:
            collate_fn = PackedCollator(pad_id=pad_id, dataset=self.train_dataset)

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=collate_fn,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=self.args.dataloader_prefetch_factor if self.args.dataloader_num_workers > 0 else None,
        )

        # Wrap the dataloader to handle state synchronization from workers to main process
        return StateAwareDataLoader(dataloader, self.train_dataset)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        super()._save(output_dir, state_dict=state_dict)
        if self._export_layout is not None and self.args.should_save:
            copied = export_eagle_embodied_checkpoint(output_dir, self._export_layout)
            logger.info("Exported %s remote code: %s", self._export_layout, ", ".join(copied))


def _annotation_paths(meta: dict) -> List[str]:
    paths = meta['annotation']
    if isinstance(paths, str):
        return [paths]
    return list(paths)


def _balanced_rank_shard(items: List[Any], weights: Dict[Any, int]) -> List[Any]:
    """Greedily spread index-building work across ranks by byte/row cost."""
    world_size = get_world_size()
    rank = get_rank()
    if world_size == 1:
        return list(items)

    assignments = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for item in sorted(items, key=lambda value: (-weights[value], str(value))):
        target_rank = min(range(world_size), key=lambda index: (loads[index], index))
        assignments[target_rank].append(item)
        loads[target_rank] += weights[item]
    return assignments[rank]


def _gather_rank_dict(local_items: Dict[Any, Any]) -> Dict[Any, Any]:
    """Gather disjoint build results from every distributed rank."""
    if not is_dist_avail_and_initialized():
        return dict(local_items)

    gathered: List[Optional[Dict[Any, Any]]] = [None] * get_world_size()
    dist.all_gather_object(gathered, local_items)
    merged: Dict[Any, Any] = {}
    for rank_items in gathered:
        if rank_items is None:
            raise RuntimeError('A rank returned no JSONL index build results')
        duplicates = merged.keys() & rank_items.keys()
        if duplicates:
            raise RuntimeError(
                f'Duplicate JSONL index build results: {sorted(map(str, duplicates))}'
            )
        merged.update(rank_items)
    return merged


def _active_index_identity(index_paths: List[str]) -> str:
    return '|'.join(os.path.basename(path) for path in index_paths)


def build_stream_packed_dataset(
    data_args,
    processor,
    base_seed: int = 42,
    dataset_kwargs: Optional[dict] = None,
    **stream_kwargs,
) -> StreamPackedDataset:
    """Build StreamPackedDataset.

    ``dataset_kwargs`` are extra ``LazySupervisedDataset`` options and
    ``stream_kwargs`` the optional ``StreamPackedDataset`` extensions (VFlash).
    A recipe entry with ``"enabled": false`` is skipped.
    """
    dataset_kwargs = dict(dataset_kwargs or {})
    with open(data_args.meta_path, 'rb') as meta_handle:
        meta_bytes = meta_handle.read()
    dataset_fingerprint = hashlib.sha256(meta_bytes).hexdigest()
    ds_collections = json.loads(meta_bytes)
    for ds_name, meta in ds_collections.items():
        if not isinstance(meta.get('enabled', True), bool):
            raise ValueError(
                f'Recipe entry {ds_name!r} field "enabled" must be a JSON boolean'
            )
    ds_collections = {
        ds_name: meta for ds_name, meta in ds_collections.items()
        if meta.get('enabled', True)
    }

    index_dir = getattr(data_args, 'jsonl_index_dir', None)
    if not index_dir:
        index_dir = os.environ.get('EAGLE_JSONL_INDEX_DIR')
    if not index_dir:
        index_dir = os.path.join(
            os.path.dirname(os.path.abspath(data_args.meta_path)),
            '.jsonl_index',
        )
    index_dir = os.path.abspath(os.path.expanduser(index_dir))
    index_workers = int(getattr(data_args, 'jsonl_index_workers', 4))
    chunk_size_mb = int(getattr(data_args, 'jsonl_index_chunk_size_mb', 4))
    if index_workers < 1:
        raise ValueError('jsonl_index_workers must be positive')
    if chunk_size_mb < 1:
        raise ValueError('jsonl_index_chunk_size_mb must be positive')

    source_expected_rows: Dict[str, Optional[int]] = {}
    for meta in ds_collections.values():
        paths = _annotation_paths(meta)
        expected_rows = meta.get('length') if len(paths) == 1 else None
        for path in paths:
            previous = source_expected_rows.get(path)
            if previous is not None and expected_rows is not None and previous != expected_rows:
                raise ValueError(
                    f'Conflicting recipe lengths for {path}: {previous} and {expected_rows}'
                )
            if path not in source_expected_rows or previous is None:
                source_expected_rows[path] = expected_rows

    # Rebuild every source on every launch. Each source is assigned to one rank,
    # while atomic files and locks keep concurrent jobs from publishing partial indexes.
    source_weights = {path: os.path.getsize(path) for path in source_expected_rows}
    local_sources = _balanced_rank_shard(list(source_expected_rows), source_weights)
    index_started = time.monotonic()
    local_index_info = prepare_jsonl_indices(
        {path: source_expected_rows[path] for path in local_sources},
        index_dir=index_dir,
        workers=index_workers,
        chunk_size=chunk_size_mb * 1024 * 1024,
    )
    local_rebuilt = len(local_index_info)
    logger.info(
        '[JSONL index] rank=%d files=%d rebuilt=%d index_dir=%s elapsed=%.2fs',
        get_rank(), len(local_index_info), local_rebuilt, index_dir,
        time.monotonic() - index_started,
    )
    index_info = _gather_rank_dict(local_index_info)
    missing_sources = source_expected_rows.keys() - index_info.keys()
    if missing_sources:
        raise RuntimeError(
            f'Missing JSONL index build results: {sorted(missing_sources)}'
        )
    if get_rank() == 0:
        mismatches = [
            (path, expected, index_info[path].rows)
            for path, expected in source_expected_rows.items()
            if expected is not None and expected != index_info[path].rows
        ]
        if mismatches:
            preview = ', '.join(
                f'{os.path.basename(path)}: recipe={expected}, actual={actual}'
                for path, expected, actual in mismatches[:10]
            )
            logger.warning(
                '[JSONL index] %d recipe length mismatches (first 10: %s)',
                len(mismatches), preview,
            )
        logger.info(
            '[JSONL index] ready: files=%d rows=%d compact_size=%.1f MiB',
            len(index_info),
            sum(info.rows for info in index_info.values()),
            sum(info.rows for info in index_info.values()) * 8 / (1024 ** 2),
        )

    # Preserve the old random.Random(10086).sample(...), sorted selection exactly,
    # but write it once so spawned DataLoader workers only remap a compact uint array.
    active_tasks = {}
    for ds_name, meta in ds_collections.items():
        repeat_time = meta.get('repeat_time', 1)
        if repeat_time >= 1:
            continue
        paths = _annotation_paths(meta)
        jsonl_index_paths = [index_info[path].index_path for path in paths]
        total_rows = sum(index_info[path].rows for path in paths)
        sample_rows = int(total_rows * repeat_time)
        if sample_rows > 0:
            active_tasks[ds_name] = (
                _active_index_identity(jsonl_index_paths), total_rows, sample_rows
            )

    active_weights = {name: task[2] for name, task in active_tasks.items()}
    local_active_tasks = _balanced_rank_shard(list(active_tasks), active_weights)
    local_active_info = {}
    for ds_name in local_active_tasks:
        identity, total_rows, sample_rows = active_tasks[ds_name]
        active_path, active_dtype = build_active_index(
            index_dir,
            identity,
            total_rows,
            sample_rows,
            seed=10086,
        )
        local_active_info[ds_name] = (active_path, active_dtype.str)
    logger.info(
        '[JSONL active rows] rank=%d datasets=%d rebuilt=%d',
        get_rank(), len(local_active_tasks), len(local_active_info),
    )
    active_index_info = _gather_rank_dict(local_active_info)
    missing_active = active_tasks.keys() - active_index_info.keys()
    if missing_active:
        raise RuntimeError(
            f'Missing active-row index build results: {sorted(missing_active)}'
        )

    datasets = []
    dataset_weights = []

    for ds_name, meta in ds_collections.items():
        repeat_time = meta.get('repeat_time', 1)
        try:
            paths = _annotation_paths(meta)
            jsonl_index_paths = [index_info[path].index_path for path in paths]
            jsonl_row_counts = [index_info[path].rows for path in paths]
            active_index_path = None
            active_index_dtype = None
            if ds_name in active_tasks:
                active_index_path, dtype_string = active_index_info[ds_name]
                active_index_dtype = np.dtype(dtype_string)
            ds = LazySupervisedDataset(
                ds_name, meta, processor,
                repeat_time=repeat_time,
                target_fps=data_args.target_fps,
                max_frames=data_args.max_frames,
                video_total_pixels=data_args.video_total_pixels,
                auto_thinking_handler=getattr(data_args, 'auto_thinking_handler', False),
                jsonl_index_paths=jsonl_index_paths,
                jsonl_row_counts=jsonl_row_counts,
                active_index_path=active_index_path,
                active_index_dtype=active_index_dtype,
                **dataset_kwargs,
            )

            if len(ds) == 0:
                if get_rank() == 0:
                    logger.warning(f'Dataset {ds_name} is empty, skipping.')
                continue

            datasets.append(ds)

            weight = repeat_time * len(ds) if repeat_time >= 1 else len(ds)
            dataset_weights.append(weight)

            if get_rank() == 0:
                logger.info(f'Added dataset: {ds_name}, length={len(ds)}, '
                           f'repeat_time={repeat_time}, weight={weight:.0f}')

        except Exception as e:
            traceback.print_exc()
            logger.error(f'Error loading dataset {ds_name}: {e}')
            raise

    if len(datasets) == 0:
        raise ValueError("No valid datasets found!")

    # Check for buffer size in data_args, default to 32
    buffer_size = getattr(data_args, 'packing_buffer_size', 32)

    pg_manager = get_pg_manager()
    if pg_manager is None:
        data_rank = get_rank()
        data_world_size = get_world_size()
    else:
        # Sequence-parallel ranks must consume the exact same packed sample;
        # only data-parallel groups advance independent sample streams.
        data_rank = pg_manager.data_parallel_rank
        data_world_size = pg_manager.data_parallel_world_size

    balance_media_tokens = getattr(data_args, 'balance_media_tokens', False)
    if balance_media_tokens and data_world_size == 1:
        logger.info(
            'Media-token balancing requested but data_parallel_world_size=1; disabling bounded '
            'lookahead because there are no data-parallel stragglers to align.'
        )
        balance_media_tokens = False

    return StreamPackedDataset(
        tokenizer=processor.tokenizer,
        data_rank=data_rank,
        data_world_size=data_world_size,
        datasets=datasets,
        dataset_weight=dataset_weights,
        max_num_tokens_per_sample=getattr(data_args, 'max_num_tokens_per_sample', 16384),
        max_num_tokens=getattr(data_args, 'max_num_tokens', data_args.max_seq_length),
        log_freq=int(getattr(data_args, 'packing_log_interval', 10000)),
        base_seed=base_seed,
        buffer_size=buffer_size,
        balance_media_tokens=balance_media_tokens,
        balance_window_size=getattr(data_args, 'balance_window_size', 8),
        balance_seed=getattr(data_args, 'balance_seed', None),
        dataset_fingerprint=dataset_fingerprint,
        epoch_sample_count=int(round(sum(dataset_weights))),
        **stream_kwargs,
    )


def _validate_vflash_arguments(model_args, data_args, training_args):
    """Reject unsupported VFlash options."""
    if model_args.training_mode not in ('ar', 'vflash'):
        raise ValueError(
            f'--training_mode must be ar or vflash, got {model_args.training_mode!r}'
        )
    validate_oof_loss_scale(model_args.oof_loss_scale)
    if data_args.vflash_train_sample_epochs is not None:
        sample_epochs = float(data_args.vflash_train_sample_epochs)
        if model_args.training_mode != 'vflash':
            raise ValueError('--vflash_train_sample_epochs is only valid with --training_mode vflash.')
        if not math.isfinite(sample_epochs) or sample_epochs <= 0:
            raise ValueError('--vflash_train_sample_epochs must be finite and positive.')
    if model_args.training_mode != 'vflash':
        return
    if not model_args.model_name_or_path:
        raise ValueError(
            'VFlash training requires --model_name_or_path to point to one complete '
            'exported VFlash target.'
        )
    if training_args.label_smoothing_factor:
        raise ValueError('VFlash uses its direct weighted loss and requires label_smoothing_factor=0.')
    if data_args.sequence_parallel_degree != 1:
        raise ValueError(
            'VFlash training requires sequence_parallel_degree=1; the separate '
            'draft sparse mask is not Ulysses-sharded.'
        )
    if data_args.vflash_max_samples_per_pack < 1:
        raise ValueError('--vflash_max_samples_per_pack must be at least 1.')
    torch_version = version.parse(torch.__version__.split('+', 1)[0])
    # NVIDIA containers may ship a 2.8 development build such as
    # 2.8.0a0+...nv25.06, which PEP 440 orders before 2.8.0; gate on the
    # release major/minor and on the compiled FlexAttention check below.
    if torch_version.release[:2] != (2, 8):
        raise RuntimeError(
            'VFlash training requires PyTorch 2.8.x for the compiled '
            f'FlexAttention path; found torch {torch.__version__}.'
        )
    if _compiled_create_block_mask is None or _compiled_flex_attention is None:
        raise RuntimeError(
            'VFlash training requires compiled create_block_mask and '
            'FlexAttention; the eager dense fallback is intentionally disabled.'
        )


def _vflash_objective_config(model_args, data_args, serialized_physical_atoms):
    """Return the drafter objective a VFlash data-stream resume must match."""
    return {
        'version': VFLASH_BINDING_VERSION,
        'allow_variable_trajectory_lengths': True,
        'ordinary_anchors_per_sequence': model_args.vflash_anchors_per_sequence,
        'ordinary_text_loss_gamma': model_args.vflash_loss_gamma,
        'physical_atom_loss_weights': 'uniform',
        'oof_loss_scale': model_args.oof_loss_scale,
        'text_block_size': TEXT_VFLASH_BLOCK_SIZE,
        'num_draft_layers': VFLASH_NUM_DRAFT_LAYERS,
        'max_samples_per_pack': data_args.vflash_max_samples_per_pack,
        'train_sample_epochs': data_args.vflash_train_sample_epochs,
        'physical_atom_sha256': stable_json_sha256(serialized_physical_atoms),
        'dispatch': {
            'point_bbox': 'fixed_atom',
            'robot_driving': 'one_complete_primitive',
            'dense_3d': 'one_block_per_trajectory',
        },
    }


def main():
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_mode = model_args.training_mode
    if training_mode is not None:
        _validate_vflash_arguments(model_args, data_args, training_args)

    if model_args.attn_implementation != 'flash_attention_2':
        raise ValueError(
            'Stream packing requires attn_implementation=flash_attention_2 so integer segment '
            'IDs are handled by the patched varlen attention kernel without cross-sample attention.'
        )
    if data_args.max_num_tokens_per_sample > data_args.max_seq_length:
        raise ValueError(
            f'max_num_tokens_per_sample ({data_args.max_num_tokens_per_sample}) cannot exceed '
            f'max_seq_length ({data_args.max_seq_length}), because tokenization truncates first.'
        )
    if data_args.max_num_tokens < data_args.max_num_tokens_per_sample:
        raise ValueError(
            f'max_num_tokens ({data_args.max_num_tokens}) must be at least '
            f'max_num_tokens_per_sample ({data_args.max_num_tokens_per_sample}).'
        )
    if data_args.packing_log_interval < 1:
        raise ValueError('--packing_log_interval must be at least 1.')
    if (
        data_args.sequence_parallel_degree > 1
        and (model_args.rope_scaling_type or '').lower() == 'dynamic'
    ):
        raise ValueError(
            'Dynamic RoPE scaling is not supported with sequence parallelism because rank-local '
            'position ranges can update rotary frequencies inconsistently. Use yarn or linear.'
        )

    if os.path.exists(osp.join(training_args.output_dir, 'done.txt')):
        logger.info("Training done (done.txt exists), exiting!")
        return

    # Patches
    patch_packing_attention()
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
    # VFlash appends its object and physical-atom tokens after the Eagle Embodied list.
    added_special_tokens = eagle_embodied_special_tokens_list if training_mode is None else vflash_special_tokens_list
    num_new_tokens = tokenizer.add_tokens(
        added_special_tokens + number_tokens_list, special_tokens=True
    )

    if len(tokenizer.encode("assistant")) > 1:
        tokenizer.add_tokens(["assistant"], special_tokens=False)
        num_new_tokens += 1

    physical_token_ids = None
    serialized_physical_atoms = None
    if training_mode is not None:
        physical_token_ids = validate_physical_atom_tokenizer(tokenizer)
        serialized_physical_atoms = physical_atom_config(physical_token_ids)
    if training_mode == 'vflash' and num_new_tokens:
        raise ValueError(
            'The VFlash recipe cannot add tokenizer rows to its frozen target. '
            'Train the VFlash target with --training_mode ar first.'
        )

    image_token_index = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    if model_args.model_name_or_path is not None:
        logger.info('Loading EagleEmbodiedForConditionalGeneration...')
        config = EagleEmbodiedConfig.from_pretrained(model_args.model_name_or_path)
        rope_scaling = configure_text_config_for_long_context(
            config.text_config,
            data_args.max_num_tokens_per_sample,
            rope_scaling_type=model_args.rope_scaling_type,
            rope_scaling_factor=model_args.rope_scaling_factor,
            original_max_position_embeddings=model_args.rope_original_max_position_embeddings,
        )
        logger.info(
            f'Long-context config: max_position_embeddings='
            f'{config.text_config.max_position_embeddings}, rope_scaling={rope_scaling}'
        )
        config._attn_implementation = model_args.attn_implementation
        config.text_config._attn_implementation = model_args.attn_implementation
        config.image_token_index = image_token_index
        model = EagleEmbodiedForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16, config=config,
            attn_implementation=model_args.attn_implementation
        )
        if training_mode is not None:
            # VFlash rows reference {video, frame_index} images that only the
            # repository processor decodes, so no checkpoint processor code runs.
            processor = build_local_processor(model_args.model_name_or_path, tokenizer)
            logger.info('Using the repository EagleEmbodiedProcessor with checkpoint assets.')
        else:
            try:
                processor = AutoProcessor.from_pretrained(
                    model_args.model_name_or_path, trust_remote_code=True, use_fast=True)
                processor.tokenizer = tokenizer
            except Exception:
                processor_dir = model_args.model_name_or_path
                logger.exception(f"Failed to load processor from {processor_dir}")
                for filename in [
                    "config.json",
                    "processor_config.json",
                    "preprocessor_config.json",
                    "tokenizer_config.json",
                    "chat_template.json",
                    "chat_template.jinja",
                    "processing_eagle_embodied.py",
                    "image_processing_eagle_embodied.py",
                    "configuration_eagle_embodied.py",
                ]:
                    file_path = osp.join(processor_dir, filename)
                    logger.error(f"processor debug: {filename} exists={osp.exists(file_path)}")

                logger.warning("Falling back to local EagleEmbodiedProcessor construction.")
                processor = build_local_processor(processor_dir, tokenizer)
    else:
        logger.info(f"Loading vision backbone from {model_args.vision_path}")
        vision_config = AutoConfig.from_pretrained(model_args.vision_path, trust_remote_code=True)

        if vision_config.model_type == 'moonvit':
            logger.info('Loading MoonVit...')
            vision_config._attn_implementation = 'flash_attention_2'
            vision_model = MoonVitPretrainedModel.from_pretrained(
                model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config,
                use_final_layernorm=False)
        else:
            raise ValueError(f"Unsupported vision model type: {vision_config.model_type}")

        logger.info('Loading LLM...')
        text_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        rope_scaling = configure_text_config_for_long_context(
            text_config,
            data_args.max_num_tokens_per_sample,
            rope_scaling_type=model_args.rope_scaling_type,
            rope_scaling_factor=model_args.rope_scaling_factor,
            original_max_position_embeddings=model_args.rope_original_max_position_embeddings,
        )
        logger.info(
            f'Long-context config: max_position_embeddings='
            f'{text_config.max_position_embeddings}, rope_scaling={rope_scaling}'
        )
        text_config._attn_implementation = model_args.attn_implementation

        llm = AutoModelForCausalLM.from_pretrained(
            model_args.llm_path, torch_dtype=torch.bfloat16,
            config=text_config, trust_remote_code=True,
            attn_implementation=model_args.attn_implementation)

        eagle_embodied_config = EagleEmbodiedConfig(
            vision_config.to_dict(), text_config.to_dict(),
            image_token_index=image_token_index,
            mlp_connector_layers=model_args.mlp_connector_layers)
        eagle_embodied_config._attn_implementation = model_args.attn_implementation
        model = EagleEmbodiedForConditionalGeneration(eagle_embodied_config, vision_model, llm)

        chat_template_data = load_config(model_args.chat_template_path)
        processor_config = load_config(model_args.processor_config_path)
        preprocessor_config = load_config(model_args.preprocessor_config_path)
        image_processor = EagleEmbodiedImageProcessor(**preprocessor_config)
        processor_config["chat_template"] = chat_template_data["chat_template"]
        processor = EagleEmbodiedProcessor(tokenizer=tokenizer, image_processor=image_processor, **processor_config)

    model.neftune_alpha = data_args.neftune_alpha
    if training_mode is not None:
        model.config.physical_atom_config = serialized_physical_atoms
        model.config.oof_loss_scale = float(model_args.oof_loss_scale)
        model.config.training_mode = training_mode
        model.config.physical_atom_length_mode = 'atomic_decoding'
        logger.info(
            'Physical <oof> label training-signal scale: %s (%s mode)',
            model.config.oof_loss_scale,
            training_mode,
        )

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)

    deepspeed_config = training_args.deepspeed
    if isinstance(deepspeed_config, str):
        with open(deepspeed_config, 'r', encoding='utf-8') as handle:
            deepspeed_config = json.load(handle)
    zero_stage = (
        deepspeed_config.get('zero_optimization', {}).get('stage', 0)
        if isinstance(deepspeed_config, dict)
        else 0
    )
    deepspeed_module = None
    if int(zero_stage) == 3:
        import deepspeed as deepspeed_module
    if training_mode == 'vflash' and int(zero_stage) == 3:
        raise ValueError(
            'The standalone-drafter save path supports DeepSpeed ZeRO-2, not ZeRO-3. '
            'Use deepspeed_configs/zero_stage2_config.json.'
        )

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_parameter = model.language_model.get_output_embeddings().weight

        def initialize_new_output_tokens():
            output_embeddings = output_parameter.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0,
                keepdim=True,
            )
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        if deepspeed_module is not None:
            # HF may construct parameters partitioned when ZeRO-3 is configured
            # before from_pretrained. Gather the resized head before initializing
            # new rows, update it once, then let DeepSpeed repartition/broadcast.
            with deepspeed_module.zero.GatheredParameters(
                [output_parameter],
                modifier_rank=0,
            ):
                if dist.get_rank() == 0:
                    initialize_new_output_tokens()
        else:
            initialize_new_output_tokens()
        model.config.text_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    if deepspeed_module is not None:
        # Forward consumes the lm_head weight directly in fused linear CE
        # instead of calling the child module. Tell ZeRO-3 that the parameter
        # is external to the Eagle wrapper so it is gathered for this forward.
        deepspeed_module.zero.register_external_parameter(
            model,
            model.language_model.lm_head.weight,
        )
        logger.info('Registered lm_head.weight as a ZeRO-3 external parameter.')

    model.language_model.config.use_cache = False

    if model_args.grad_checkpoint:
        checkpoint_kwargs = {'use_reentrant': model_args.gradient_checkpointing_use_reentrant}
        try:
            model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=checkpoint_kwargs
            )
        except TypeError:
            model.language_model.gradient_checkpointing_enable()
        except AttributeError:
            model.language_model._set_gradient_checkpointing()

        vision_checkpointing_enabled = False
        try:
            model.vision_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=checkpoint_kwargs
            )
            vision_checkpointing_enabled = True
        except (AttributeError, TypeError):
            pass
        for module_path in ('encoder', 'vision_model'):
            module = getattr(model.vision_model, module_path, None)
            if module is not None and hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
                vision_checkpointing_enabled = True
        logger.info(
            f'Gradient checkpointing enabled: use_reentrant='
            f'{model_args.gradient_checkpointing_use_reentrant}, '
            f'vision={vision_checkpointing_enabled}'
        )
    logger.info("model init done")

    # Sequence parallelism
    if data_args.sequence_parallel_degree == 0:
        world_size = dist.get_world_size()
        num_heads = model.language_model.config.num_attention_heads
        valid_degrees = [
            degree
            for degree in range(1, world_size + 1)
            if world_size % degree == 0 and num_heads % degree == 0
        ]
        data_args.sequence_parallel_degree = max(valid_degrees)
        logger.info(
            f'Auto-selected sequence_parallel_degree={data_args.sequence_parallel_degree} '
            f'for world_size={world_size}, num_attention_heads={num_heads}.'
        )
    if (
        data_args.sequence_parallel_degree > 1
        and (model_args.rope_scaling_type or '').lower() == 'dynamic'
    ):
        raise ValueError(
            'Dynamic RoPE scaling is not supported with sequence parallelism because rank-local '
            'position ranges can update rotary frequencies inconsistently. Use yarn or linear.'
        )
    if data_args.sequence_parallel_degree > 1:
        # Packed Eagle Embodied attention implements Ulysses only. The shared
        # set_pg_manager would fall back to ring attention for these settings.
        num_heads = model.language_model.config.num_attention_heads
        if data_args.ring_sequence_parallel_degree != 1:
            raise ValueError(
                'Packed Eagle Embodied training currently supports Ulysses sequence parallelism only; '
                'set ring_sequence_parallel_degree=1.'
            )
        if num_heads % data_args.sequence_parallel_degree != 0:
            raise ValueError(
                f'num_attention_heads ({num_heads}) must be divisible by sequence_parallel_degree '
                f'({data_args.sequence_parallel_degree}) for Ulysses all-to-all; '
                'pass sequence_parallel_degree=0 to auto-select.'
            )
        set_pg_manager(
            model,
            data_args.sequence_parallel_degree,
            ring_sequence_parallel_size=data_args.ring_sequence_parallel_degree,
        )
        logger.info(
            f'Ulysses sequence parallelism enabled: SP={data_args.sequence_parallel_degree}, '
            f'DP={get_pg_manager().data_parallel_world_size}'
        )

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
    logger.info("Building stream packed dataset...")
    t_start = time.time()
    dataset_kwargs = dict(strict_data=data_args.strict_data)
    stream_kwargs = dict(strict_data=data_args.strict_data)
    if training_mode is not None:
        dataset_kwargs.update(
            physical_tokens=physical_token_ids,
            emit_label_weights=True,
            # The target loss expects <oof> labels already scaled; the drafter
            # loss scales them itself against its unscaled denominator.
            oof_loss_scale=model_args.oof_loss_scale if training_mode == 'ar' else None,
            visual_aug=True,
        )
    if training_mode == 'vflash':
        stream_kwargs.update(
            max_samples_per_pack=data_args.vflash_max_samples_per_pack,
            extra_packing_config=_vflash_objective_config(
                model_args, data_args, serialized_physical_atoms
            ),
            emit_pack_seed=True,
        )
    train_dataset = build_stream_packed_dataset(
        data_args,
        processor,
        base_seed=training_args.seed,
        dataset_kwargs=dataset_kwargs,
        **stream_kwargs,
    )
    logger.info(f"Dataset built in {time.time() - t_start:.2f}s")

    raw_sample_schedule = None
    if training_mode == 'vflash' and data_args.vflash_train_sample_epochs is not None:
        samples_per_pack = int(data_args.vflash_max_samples_per_pack)
        # Every pack must reach its sample count; a token-budget flush would
        # make max_steps an inexact sample budget.
        required_pack_capacity = samples_per_pack * int(data_args.max_num_tokens_per_sample)
        if int(data_args.max_num_tokens) < required_pack_capacity:
            raise ValueError(
                '--vflash_train_sample_epochs requires fixed-size packs: '
                f'max_num_tokens={data_args.max_num_tokens} must be at least '
                f'vflash_max_samples_per_pack * max_num_tokens_per_sample '
                f'({required_pack_capacity}).'
            )
        if int(training_args.per_device_train_batch_size) != 1:
            raise ValueError(
                'VFlash raw-sample scheduling requires per_device_train_batch_size=1.'
            )
        raw_sample_schedule = resolve_vflash_sample_schedule(
            epoch_samples=int(train_dataset.epoch_sample_count),
            sample_epochs=float(data_args.vflash_train_sample_epochs),
            samples_per_pack=samples_per_pack,
            data_parallel_world_size=dist.get_world_size(),
            gradient_accumulation_steps=int(training_args.gradient_accumulation_steps),
        )
        training_args.max_steps = raw_sample_schedule['max_steps']
        logger.info(
            'VFlash raw-sample schedule: epoch_samples=%d epochs=%s, '
            'samples/update=%d, max_steps=%d, trained_samples=%d, '
            'distributed_tail_padding=%d.',
            raw_sample_schedule['epoch_samples'],
            data_args.vflash_train_sample_epochs,
            raw_sample_schedule['samples_per_optimizer_step'],
            raw_sample_schedule['max_steps'],
            raw_sample_schedule['trained_raw_samples'],
            raw_sample_schedule['distributed_tail_padding_samples'],
        )

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
        for parameter in model.language_model.lm_head.parameters():
            parameter.requires_grad_(True)

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

    if training_mode == 'vflash':
        if (
            model_args.use_llm_lora
            or model_args.use_backbone_lora
            or getattr(model, 'use_llm_lora', False)
            or getattr(model.config, 'use_backbone_lora', 0)
        ):
            raise ValueError(
                'VFlash requires an exported, fixed Eagle target; LoRA adapters '
                'must be merged into the AR checkpoint before drafter training.'
            )
        if model_args.unfreeze_lm_head or model_args.unfreeze_vit_layers:
            raise ValueError('VFlash freezes the complete target, including LM head and vision.')
        logger.info('Fingerprinting the complete frozen Eagle target on rank 0...')
        target_weights_sha256 = distributed_target_weight_fingerprint(model)
        target_tokenizer_sha256 = tokenizer_fingerprint(tokenizer)
        logger.info('Frozen Eagle target fingerprint: %s', target_weights_sha256)
        draft_config = build_vflash_draft_config(
            model.config.text_config,
            mask_token_id=physical_token_ids.mask_token_id,
            physical_atom_config=serialized_physical_atoms,
            target_model_id=str(model_args.model_name_or_path),
            target_weights_sha256=target_weights_sha256,
            target_tokenizer_sha256=target_tokenizer_sha256,
            target_revision=getattr(model.config, '_commit_hash', None),
            oof_loss_scale=model_args.oof_loss_scale,
        )
        resume_checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        if resume_checkpoint:
            # A VFlash checkpoint holds only the drafter, bound to this target.
            validate_vflash_draft_binding(
                AutoConfig.from_pretrained(resume_checkpoint, trust_remote_code=True),
                target_text_config=model.config.text_config,
                target_weights_sha256=target_weights_sha256,
                target_tokenizer_sha256=target_tokenizer_sha256,
                physical_atom_config=serialized_physical_atoms,
                oof_loss_scale=model_args.oof_loss_scale,
            )
            draft_model = DFlashDraftModel.from_pretrained(
                resume_checkpoint,
                torch_dtype=torch.bfloat16,
            )
            logger.info('Loaded validated standalone VFlash drafter from %s', resume_checkpoint)
        else:
            draft_model = DFlashDraftModel(draft_config).to(dtype=torch.bfloat16)
        assert_hf_native_drafter(draft_model)
        if model_args.grad_checkpoint:
            draft_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': model_args.gradient_checkpointing_use_reentrant}
            )
        model = EagleEmbodiedVFlashTrainingModel(model, draft_model)
        logger.info(
            'VFlash enabled: frozen Eagle target + %d-layer full-width drafter; '
            'clean AR loss disabled.',
            len(model.draft_model.layers),
        )

    if dist.get_rank() == 0:
        logger.info("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"  {name}")

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

    if training_mode == 'vflash':
        trainer_class = build_vflash_trainer(StreamPackingTrainer)
    else:
        trainer_class = StreamPackingTrainer
    CustomTrainer = trainer_class

    assert processor is not None, "Processor is required"

    if training_mode == 'vflash':
        collate_fn = VFlashPackedCollator(
            physical_tokens=physical_token_ids,
            packed_collate_fn=packed_collate_fn,
            ordinary_anchors_per_sequence=model_args.vflash_anchors_per_sequence,
            gamma=model_args.vflash_loss_gamma,
        )
    else:
        collate_fn = PackedCollator(pad_id=processor.tokenizer.pad_token_id, dataset=train_dataset)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        data_collator=collate_fn,
        callbacks=my_callbacks,
        processing_class=processor,
        sample_log_interval=getattr(data_args, 'sample_log_interval', 100),
        debug_assert_sp_batch_sync=getattr(data_args, 'debug_assert_sp_batch_sync', False),
        # VFlash target checkpoints carry the remote code from every save on.
        export_layout='moonvit' if training_mode == 'ar' else None,
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
                    logger.error(f"Rank {rank}: Failed to load dataloader state: {e}")
                    traceback.print_exc()
                    raise RuntimeError(
                        f'Rank {rank}: refusing a non-exact resume from {checkpoint}.'
                    ) from e
            else:
                raise FileNotFoundError(
                    f'Rank {rank}: no dataloader state found at {dataloader_state_path}; '
                    'refusing to resume model/optimizer from the beginning of the data stream.'
                )

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        if training_mode is None and get_rank() == 0:
            copied = export_eagle_embodied_checkpoint(training_args.output_dir, "moonvit")
            logger.info("Exported Eagle Embodied remote code: %s", ", ".join(copied))

        metrics = train_result.metrics
        metrics['train_samples'] = 'streaming'
        if raw_sample_schedule is not None:
            metrics.update(raw_sample_schedule)

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()

    with open(osp.join(training_args.output_dir, 'done.txt'), 'w') as f:
        f.write('done: ' + time.ctime())


if __name__ == '__main__':
    main()
