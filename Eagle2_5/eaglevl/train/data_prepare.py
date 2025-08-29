import gc
import logging
import os
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional
import os.path as osp
# import orjson as json
import json
import re
import av
import bz2
import cv2
import torch
import torch.distributed as dist
import transformers
from eaglevl.dist_utils import init_dist
from eaglevl.patch import replace_train_sampler
from eaglevl.train.constants import special_tokens_list
from eaglevl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    dynamic_preprocess, preprocess, find_closest_aspect_ratio_v2, PROCESS_FUNCTIONS)
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed)
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
from typing import Tuple
from torchcodec.decoders import VideoDecoder, AudioDecoder
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import time
import pandas as pd
import pyarrow.parquet as pq
import hashlib
import pyarrow as pa
from transformers.trainer_pt_utils import LabelSmoother
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from eaglevl.train.arguments import ModelArguments
import lmdb
import io
import pickle
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
replace_train_sampler()
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def calculate_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=224,
        metadata={'help': 'Set the desired size for the image. Default is 224.'},
    )
    down_sample_ratio: Optional[float] = field(
        default=1.0,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 1.0.'},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='qwen2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic image size.'},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image.'},
    )
    min_dynamic_tiles: Optional[int] = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic tiles. Default is 1.'},
    )
    max_dynamic_tiles: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic tiles. Default is 12.'},
    )
    neftune_alpha: Optional[float] = field(
        default=None,
        metadata={'help': 'The noise_alpha value for NEFTune. Default is None.'},
    )
    normalize_type: Optional[str] = field(
        default='imagenet',
        metadata={'help': 'The normalize type for the image. Default is imagenet.'},
    )
    save_root: Optional[str] = field(
        default='playground/sft_recipe/cambrian_7m'
    )
    batch_size: Optional[int] = field(
        default=1024,
        metadata={'help': 'The batch size for packing data. Default is 100.'},
    )
    sample_frame_strategy: Optional[str] = field(
        default='fixed_32',
        metadata={'help': 'The sample frame strategy. Default is fixed_32.'},
    )
    sample_length_div: int = field(
        default=1,
        metadata={'help': 'Specify the division of sample length. Default is 1.'}
    )
    parquet_cache_file: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the parquet cache file. Default is None.'},
    )
    tile_downsample_size: Optional[str] = field(
        default=None,
        metadata={'help': 'The tile pooling size. Default is None.'},
    )
    max_token_per_sample: Optional[int] = field(
        default=16384,
        metadata={'help': 'The maximum number of tokens per sample. Default is 16384.'},
    )
    default_fps: Optional[int] = field(
        default=2,
        metadata={'help': 'The default fps for the video. Default is 30.'},
    )
    max_num_frames: Optional[int] = field(
        default=128,
        metadata={'help': 'The maximum number of frames for the video. Default is 128.'},
    )
    min_num_frames: Optional[int] = field(
        default=8,
        metadata={'help': 'The minimum number of frames for the video. Default is 8.'},
    )

def read_img_from_lmdb_v2(lmdb_data):
    # special case for AgiBotWorld
    lmdb_file, lmdb_key = lmdb_data['lmdb_file'], lmdb_data['lmdb_key']
    key = lmdb_key.encode('ascii')
    env = lmdb.open(lmdb_file, max_readers=10240, readonly=True, lock=False, readahead=False, meminit=False)
    txn = env.begin()
    value = txn.get(key)
    if value is None:
        print(f"Warning: Key {key} not found.")
        return None
    record = pickle.loads(value)
    image_bgr = cv2.imdecode(np.frombuffer(record['image'], dtype=np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
        
    return image

def read_image_from_lmdb(image_data):
    # special case for AgiBotWorld
    if 'AgiBotWorld' in image_data['lmdb_file']:
        return read_img_from_lmdb_v2(image_data)
    
    try:
        env = lmdb.open(image_data['lmdb_file'], readonly=True, lock=False, max_readers=10240)
    except Exception as e:
        print(f"Failed to open lmdb file {image_data['lmdb_file']}. Error message: {e}", flush=True)
        return None
        
    with env.begin(write=False) as txn:
        try:
            image_bin = txn.get(image_data['lmdb_key'].encode('ascii'))
            buf = io.BytesIO(image_bin)
        except Exception as e:
            print(f"Failed to get image from lmdb file {image_data['lmdb_file']}. Error message: {e}", flush=True)
            return None
    try:
        image = Image.open(buf)
    except Exception as e:
        image_np = np.frombuffer(image_bin, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
    return image

def parse_image_data(image_data):
    if isinstance(image_data, str) and osp.exists(image_data):
        image = Image.open(image_data)
    elif isinstance(image_data, dict) and 'lmdb_file' in image_data and 'lmdb_key' in image_data:
        # lmdb case 
        # load image from lmdb
        image = read_image_from_lmdb(image_data)
        if image is None:
            raise ValueError(f'Failed to read image from lmdb file {image_data["lmdb_file"]}. Error message: {e}')
            
    else:
        raise ValueError(f'Invalid image format: {type(image_data)}')
    return image

def fast_dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio_v2(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    num = target_aspect_ratio[0] * target_aspect_ratio[1]
    if use_thumbnail: 
        if num == 1:
            return 1, target_aspect_ratio
        else:
            return num+1, target_aspect_ratio
    else: 
        return num, target_aspect_ratio



def get_seq_frames_v2(total_num_frames, desired_num_frames=-1, stride=-1, temporal_jitter=False, corner_align=False):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.
    stride (int): Stride for sampling frames.
    temporal_jitter (bool): Whether to apply temporal jitter.
    corner_align (bool): Whether to strictly align start and end frames.

    Returns:
    list: List of indices of frames to extract.
    """

    # calculate desired_num_frames
    assert (desired_num_frames > 0 or stride > 0) and not (desired_num_frames > 0 and stride > 0), \
        "Either desired_num_frames or stride should be positive, but not both."

    if stride > 0:
        desired_num_frames = len(list(range(0, total_num_frames, stride)))

    # add assertion, ensure the number of frames is greater than 2
    assert desired_num_frames > 2, "desired_num_frames must be greater than 2."

    if corner_align:
        seq = np.linspace(0, total_num_frames - 1, desired_num_frames).round().astype(np.int32).tolist()
    else:
        seg_size = float(total_num_frames - 1) / desired_num_frames
        i = np.arange(desired_num_frames)
        starts = np.round(seg_size * i).astype(np.int32)
        ends = np.round(seg_size * (i + 1)).astype(np.int32)
        seq = ((starts + ends) // 2).tolist()

    if temporal_jitter:
        seg_size = float(total_num_frames - 1) / desired_num_frames
        shift_base = int(seg_size / 2)
        random_shifts = np.random.randint(-shift_base, shift_base + 1, size=len(seq))
        seq = np.array(seq) + random_shifts
        seq = np.clip(seq, 0, total_num_frames - 1).tolist()

    return seq



def get_video_properties_by_pyav(video_path):
    try:
        container = av.open(video_path)
        width = height = fps = duration = total_frames = has_audio = 0
        # Get video stream
        video_stream = None
        for stream in container.streams:
            if stream.type == 'video':
                video_stream = stream
                break
        if video_stream:
            video_stream.thread_type = 3
            width = video_stream.width
            height = video_stream.height
            # Frame rate might be None, need to handle this case
            if video_stream.average_rate:
                fps = float(video_stream.average_rate)
            else:
                fps = 0
            # Calculate duration correctly (in seconds)
            duration = float(video_stream.duration * video_stream.time_base)
            # Total frames
            total_frames = video_stream.frames
            if total_frames == 0:
                total_frames = len([p for p in container.demux(video=0)])
        else:
            raise ValueError(f"No video stream found in {video_path}")
        
        has_audio = any(stream.type == 'audio' for stream in container.streams)
        container.close()
    except av.FileNotFoundError as e:
        print(f"Failed to open file {video_path}. Error message: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    
    assert height > 0 and width > 0 and fps > 0 and duration > 0 and total_frames > 0
    return height, width, fps, duration, total_frames, has_audio



def get_video_properties_by_torchcodec(video_path: str) -> Tuple[int, int, float, float, int, bool]:
    """
    Returns: (height, width, fps, duration_seconds, total_frames, has_audio)

    Strategy:
      1) Try fast metadata retrieval with seek_mode="approximate".
      2) If any key field is missing/zero, fall back once to seek_mode="exact"
         (triggers a scan/indexing, but still does not decode frames).
      3) Detect audio by attempting to construct an AudioDecoder; if it succeeds,
         we consider that the file has an audio stream.
    """

    def _read_metadata(seek_mode: str):
        # num_ffmpeg_threads=0 lets FFmpeg choose an optimal thread count.
        vdec = VideoDecoder(video_path, seek_mode=seek_mode, num_ffmpeg_threads=0)
        m = vdec.metadata  # VideoStreamMetadata (public attribute)

        # Width/height straight from metadata; default to 0 if missing.
        width  = int(m.width or 0)
        height = int(m.height or 0)

        # FPS: prefer computed average_fps (after scan), else header value.
        fps = float(m.average_fps if m.average_fps is not None else (m.average_fps_from_header or 0.0))

        # Duration: prefer computed duration_seconds (after scan), else header value.
        duration = float(m.duration_seconds if m.duration_seconds is not None else (m.duration_seconds_from_header or 0.0))

        # Total frames: prefer computed num_frames, else header value,
        # otherwise estimate from duration * fps.
        if m.num_frames is not None:
            total_frames = int(m.num_frames)
        elif m.num_frames_from_header is not None:
            total_frames = int(m.num_frames_from_header)
        elif duration > 0 and fps > 0:
            total_frames = max(1, int(round(duration * fps)))
        else:
            total_frames = 0

        return height, width, fps, duration, total_frames

    try:
        # 1) Fast path: avoid scanning the file.
        height, width, fps, duration, total_frames = _read_metadata(seek_mode="approximate")

        # 2) Fallback: do a single precise scan only if needed.
        if not (height > 0 and width > 0 and fps > 0 and duration > 0 and total_frames > 0):
            height, width, fps, duration, total_frames = _read_metadata(seek_mode="exact")

        # 3) Audio presence: if AudioDecoder can be constructed and reports metadata, we have audio.
        has_audio = False
        try:
            adec = AudioDecoder(video_path)  # Works for audio files and videos with audio streams
            has_audio = adec.metadata.sample_rate is not None
        except Exception:
            has_audio = False

    except FileNotFoundError as e:
        print(f"Failed to open file {video_path}. Error message: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

    # Sanity checks similar to the original PyAV-based function.
    assert height > 0 and width > 0 and fps > 0 and duration > 0 and total_frames > 0
    return height, width, fps, duration, total_frames, has_audio




def compress_json(json_item):
    return bz2.compress(json_item.encode('utf-8'))

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, template_name, meta, tokenizer, tcs_loader, num_image_token,
                 image_size=224, is_train=True, pad2square=False, group_by_length=False,
                 dynamic_image_size=False, use_thumbnail=True, min_dynamic_tiles=1,
                 max_dynamic_tiles=6, repeat_time=1, normalize_type='imagenet', sample_length_div=1, 
                 sample_frame_strategy='fixed_32', tile_downsample_size=None, max_token_per_sample=16384,
                 default_fps=2, max_num_frames=128, min_num_frames=8, auto_thinking_prompt=False):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.sample_length_div = sample_length_div
        self.sample_frame_strategy = sample_frame_strategy
        self.tile_downsample_size = [int(x) for x in tile_downsample_size.split("x")] if tile_downsample_size is not None else None
        self.max_token_per_sample = max_token_per_sample
        self.default_fps = default_fps
        self.max_num_frames = max_num_frames
        self.min_num_frames = min_num_frames
        self.auto_thinking_prompt = auto_thinking_prompt
        logger.info(f'[Dataset] sample_frame_strategy: {sample_frame_strategy}')
        logger.info(f'[Dataset] sample_length_div: {sample_length_div}')
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_tiles: {min_dynamic_tiles}, max_dynamic_tiles: {max_dynamic_tiles}')
        logger.info(f'[Dataset] tile_downsample_size: {tile_downsample_size} parsed to {self.tile_downsample_size}')
        logger.info(f'[Dataset] max_token_per_sample: {max_token_per_sample}')
        logger.info(f'[Dataset] auto_thinking_prompt: {auto_thinking_prompt}')
        
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        with open(meta['annotation'], 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            # Split data according to rank
            self.raw_data = all_lines[rank::world_size]

        # Create a list to store compressed data with the same order as raw_data
        compressed_data = [None] * len(self.raw_data)
        
        # Use ThreadPoolExecutor for parallel compression but maintain order
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all compression tasks and keep track of their indices
            future_to_index = {
                executor.submit(compress_json, json_item): i 
                for i, json_item in enumerate(self.raw_data)
            }
            
            # Process completed futures and store results at the correct index
            for future in tqdm(as_completed(future_to_index), 
                              total=len(future_to_index), 
                              desc="Compressing JSON data"):
                index = future_to_index[future]
                compressed_data[index] = future.result()
        
        self.raw_data = compressed_data

        gc.collect()
        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_tiles = min_dynamic_tiles
        self.max_dynamic_tiles = max_dynamic_tiles
        self.normalize_type = normalize_type

 
    def __len__(self):
        return len(self.raw_data)

    def pad_ret_to_division(self, ret):
        org_length = ret['input_ids'][0].size(0)
        if org_length % self.sample_length_div != 0:
            new_length = ((org_length // self.sample_length_div) + 1) * self.sample_length_div
            pad_length = new_length - org_length
            pad_input_ids = torch.tensor([self.tokenizer.pad_token_id] * pad_length).unsqueeze(0)
            pad_labels = torch.tensor([IGNORE_TOKEN_ID] * pad_length).unsqueeze(0)
            pad_attention_masks = torch.tensor([0] * pad_length).unsqueeze(0)
            ret["input_ids"] = torch.cat([ret["input_ids"], pad_input_ids], dim = 1)
            ret["labels"] = torch.cat([ret["labels"], pad_labels], dim = 1)
            ret["attention_mask"] = torch.cat([ret["attention_mask"], pad_attention_masks], dim = 1)
        return ret


    def pure_text_get_item(self, data_item):
        preprocess_function = PROCESS_FUNCTIONS[self.template_name]

        if self.auto_thinking_prompt:
            data_item['conversations'] = self.auto_thinking_prompt_handler(data_item['conversations'])

        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, self.num_image_token * 0, text_only=True,
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)

        assert ret is not None
        targets = ret['labels'][0]
        if (targets!=IGNORE_TOKEN_ID).sum()<1:
            assert False, 'not valid labels'

        ret = self.pad_ret_to_division(ret)

        ret = dict(
            data = json.dumps([]),
            unified_frame_list = json.dumps([]),
            image_original_size_list = json.dumps([]),
            image_target_aspect_ratio_list = json.dumps([]),
            num_tokens_per_image_list = json.dumps([]),
            num_tiles_list = json.dumps([]),
            num_all_tiles = 0,
            label_length = (ret['labels'][0] != IGNORE_TOKEN_ID).sum().item(),
            length = ret['input_ids'][0].size(0),
            conversations = json.dumps(data_item['conversations']),
        )
        return ret


    def check_data_placeholders(self, conversations, media_item, data_type='image'):
        """
        we hope use <data_type-1> <data_type-2> <data_type-3> to represent the data, even only one data point
        Previous we use <image> <video> placeholders for single data point, so we need to check and replace them.
        """
        all_texts_in_conversations = ''.join([conv['value'] for conv in conversations[::2]])
        if len(media_item[f'{data_type}_list']) == 1:
            # Add or replace image placeholders
            if all_texts_in_conversations.count(f'<{data_type}>') == 0:
                conversations[0]['value'] = f'<{data_type}-1>{conversations[0]["value"]}'
            elif all_texts_in_conversations.count(f'<{data_type}>') == 1:
                # Find and replace the image tag in the conversation
                for conv in conversations[::2]:
                    if f'<{data_type}>' in conv['value']:
                        conv['value'] = conv['value'].replace(f'<{data_type}>\n', f'<{data_type}>').replace(f'<{data_type}>', f'<{data_type}-1>')
            all_texts_in_conversations = ''.join([conv['value'] for conv in conversations])
        # check data placeholders, if not exist, add them
        placeholders = ''
        for i in range(len(media_item[f'{data_type}_list'])):
            placeholder_count = all_texts_in_conversations.count(f'<{data_type}-{i+1}>')
            if placeholder_count == 0:
                placeholders = placeholders + f'<{data_type}-{i+1}>'
            elif placeholder_count > 1:
                warnings.warn(f'There are more than one {data_type} placeholders <{data_type}-{i+1}> in the conversations.')
                
        conversations[0]['value'] = placeholders + conversations[0]['value']    
        return conversations
    
    
    def handle_image_data(self, image_data):
        pil_image = parse_image_data(image_data)
        original_image_size = pil_image.size
        num_tiles_per_image, target_aspect_ratio_perimage = fast_dynamic_preprocess(pil_image, min_num=self.min_dynamic_tiles, max_num=self.max_dynamic_tiles,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        return original_image_size, target_aspect_ratio_perimage, num_tiles_per_image, self.num_image_token*num_tiles_per_image
    
    def handle_video_data(self, video_data):

        current_max_dynamic_tiles = 1 # now we only support 1 patch for video
        num_image_token = self.num_image_token
        num_tiles_list = []
        target_aspect_ratios = []
        num_tokens_per_image = []
        
        video_meta = get_video_properties_by_torchcodec(video_data['video'])
        assert video_meta is not None, f'Failed to get video information: {video_data}'
        height, width, fps, duration, total_frames, has_audio = video_meta
        
        start_frame = 0
        
        # maybe a clip of video
        if "start_time" in video_data:
            assert "start_time" in video_data and "end_time" in video_data, f'start_time and end_time must be in video_data: {video_data}'
            total_frames = int((video_data["end_time"] - video_data["start_time"]) * fps)
            start_frame = int(video_data["start_time"] * fps)
        
        DEFAULT_FPS = self.default_fps
        MIN_FRAMES = self.min_num_frames
        MAX_FRAMES = self.max_num_frames
        
        if total_frames <= MIN_FRAMES:
            # some video_list are too short or low-fps, just use the whole video, like sthv2, smit, llava-hound (2fps)
            sample_num_frames = total_frames
            DEFAULT_FPS = fps
            sample_fps_prompt = fps # this is input to video prompt
        else:
            DEFAULT_FPS = min(DEFAULT_FPS, fps)
            sample_num_frames = int(total_frames * DEFAULT_FPS / fps)
            sample_num_frames = min(max(sample_num_frames, MIN_FRAMES), MAX_FRAMES)
            sample_fps_prompt = sample_num_frames / duration # this is input to video prompt
        
        frame_idx = get_seq_frames_v2(total_frames, desired_num_frames=sample_num_frames, temporal_jitter=False, corner_align=True)
        frame_timestamp = [f / fps for f in frame_idx] # timestamp in this clip
        sample_frame_idx = [start_frame + f for f in frame_idx]

        pil_frame_list = [Image.new("RGB", (width, height)) for _ in range(len(frame_timestamp))]
        original_image_size_list = [list(pil_frame_list[i].size) for i in range(len(frame_timestamp))]

        for pil_frame in pil_frame_list:
            num_tiles_per_image, target_aspect_ratio_perimage = fast_dynamic_preprocess(pil_frame, min_num=self.min_dynamic_tiles, max_num=current_max_dynamic_tiles,
                                image_size=self.image_size, use_thumbnail=self.use_thumbnail)
            num_tiles_list.append(num_tiles_per_image)
            target_aspect_ratios.append(target_aspect_ratio_perimage)
            num_tokens_per_image.append(num_image_token*num_tiles_per_image)
        video_frame_list = [(video_data['video'], f_idx) for f_idx in sample_frame_idx]
        return video_frame_list, original_image_size_list, target_aspect_ratios, num_tiles_list, num_tokens_per_image, sample_fps_prompt, duration
    
    def replace_media_placeholder(self, conversations, image_list, video_list):
        # Global counter to track appearance order (starting from 1)
        global_counter = 1
        # Regular expression pattern to match formats like <image-1> or <video-2>
        pattern = re.compile(r"<(image|video)-(\d+)>")
        
        image_original_size_list = []
        image_target_aspect_ratio_list = []
        num_tokens_per_image_list = []
        num_tiles_list = []
        unified_frame_list = []
        
        # Function to replace tags in a single text
        def replace_in_text(text):
            nonlocal global_counter
            # repl callback function for each match replacement operation
            def repl(match):
                nonlocal global_counter
                nonlocal image_original_size_list
                nonlocal image_target_aspect_ratio_list
                nonlocal num_tokens_per_image_list
                nonlocal num_tiles_list
                nonlocal unified_frame_list
                media_type = match.group(1)          # 'image' or 'video'
                idx_in_list = int(match.group(2)) - 1   # Convert to list index (0-based)
                # Select the corresponding path based on media type
                idx_mapper = {0: "first", 1: "second", 2: "third", 3: "fourth", 4: "fifth", 5: "sixth", 6: "seventh", 7: "eighth", 8: "ninth", 9: "tenth"}  
                if media_type == 'image':
                    original_image_size, target_aspect_ratio_perimage, num_tiles_per_image, num_image_token = self.handle_image_data(image_list[idx_in_list])
                    image_original_size_list.append(original_image_size)
                    image_target_aspect_ratio_list.append(target_aspect_ratio_perimage)
                    num_tiles_list.append(num_tiles_per_image)
                    num_tokens_per_image_list.append(num_image_token)
                    special_placeholder = f"<frame-{global_counter}>"
                    if idx_in_list not in idx_mapper: idx_mapper[idx_in_list] = f"{idx_in_list+1}"
                    # special_placeholder = f"The {idx_mapper[idx_in_list]} image: {special_placeholder}\n"
                    extra_image_tag = f"<image {idx_in_list+1}>" if len(image_list) > 1 else ""
                    special_placeholder = f"{extra_image_tag}{special_placeholder}\n"
                    global_counter += 1
                    unified_frame_list.append(image_list[idx_in_list])
                elif media_type == 'video':
                    video_frame_list, original_image_sizes, target_aspect_ratios, num_tiles, num_tokens_per_image, sample_fps_prompt, duration = self.handle_video_data(video_list[idx_in_list])
                    image_original_size_list.extend(original_image_sizes)
                    image_target_aspect_ratio_list.extend(target_aspect_ratios)
                    num_tokens_per_image_list.extend(num_tokens_per_image)
                    num_tiles_list.extend(num_tiles)
                    special_placeholder = [f"Frame {i+1}: <frame-{global_counter+i}>" for i in range(len(original_image_sizes))]
                    special_placeholder = "".join(special_placeholder)
                    if idx_in_list not in idx_mapper: idx_mapper[idx_in_list] = f"{idx_in_list+1}"
                    special_placeholder = f"The {idx_mapper[idx_in_list]} video of {duration:.2f} seconds at {sample_fps_prompt:.2f} fps: {special_placeholder}\n"
                    global_counter += len(original_image_sizes)
                    unified_frame_list.extend(video_frame_list)
                else:
                    raise ValueError(f'Unknown media type: {media_type}')
                return special_placeholder
            return pattern.sub(repl, text)

        # Iterate through all conversations, replacing tags in each conversation
        for conversation in conversations:
            conversation['value'] = replace_in_text(conversation['value'])
    
        return conversations, unified_frame_list, image_original_size_list, image_target_aspect_ratio_list, num_tokens_per_image_list, num_tiles_list

    def multi_modal_get_item_one4all(self, data_item):
        # all_texts_in_conversations = ''
        # for i in range(0, len(data_item['conversations'])):
        #     all_texts_in_conversations = all_texts_in_conversations + data_item['conversations'][i]['value']
        # # get the token length information of all texts in conversations
        # pure_text_length = self.pure_text_get_item(data_item)['length']
        
        additional_data = data_item['data']
        conversations = data_item['conversations']
        
        if conversations[0]['from'] == 'system':
            # make sure conv[0] is human
            system_prompt = conversations[0]['value']
            conversations = conversations[1:]
        else:
            system_prompt = None

        image_list = []
        video_list = []
        for media_item in additional_data:
            conversations = self.check_data_placeholders(conversations, media_item, data_type=media_item['type'])
            if media_item['type'] == 'image':
                image_list = media_item['image_list']
            elif media_item['type'] == 'video':
                video_list = media_item['video_list']
        conversations, unified_frame_list, image_original_size_list, image_target_aspect_ratio_list, num_tokens_per_image_list, num_tiles_list \
            = self.replace_media_placeholder(conversations, image_list, video_list)

        if self.auto_thinking_prompt:
            conversations = self.auto_thinking_prompt_handler(conversations)

        if system_prompt is not None:
            conversations = [{"from": "system", "value": system_prompt}] + conversations
        preprocess_function = PROCESS_FUNCTIONS[self.template_name]


        ret = preprocess_function(self.template_name, deepcopy([conversations]),
                                    self.tokenizer, np.array(num_tokens_per_image_list),
                                    group_by_length=True, ds_name=self.ds_name, placeholder='frame')
        assert ret is not None
        
        ret = self.pad_ret_to_division(ret)
        
        ret = dict(
            data = json.dumps([additional_data]),
            unified_frame_list = json.dumps(unified_frame_list),
            image_original_size_list = json.dumps(image_original_size_list),
            image_target_aspect_ratio_list = json.dumps(image_target_aspect_ratio_list),
            num_tokens_per_image_list = json.dumps(num_tokens_per_image_list),
            num_tiles_list = json.dumps(num_tiles_list),
            num_all_tiles = sum(num_tiles_list),
            label_length = (ret['labels'][0] != IGNORE_TOKEN_ID).sum().item(),
            length = ret['input_ids'][0].size(0),
            conversations = json.dumps(conversations),
        )
        return ret

    def auto_thinking_prompt_handler(self, conversations):
        # human prompt decides the gpt response style, so we do not handle observation
        def find_next_human(conversations, start_idx):
            for i in range(start_idx, len(conversations)):
                if conversations[i]['from'] == 'human':
                    return i
            return len(conversations)
        
        
        def get_next_all_gpt_is_not_thinking(conversations, start_idx):
            is_thinking_mode = False
            gpt_idx_list = []
            for i in range(start_idx, len(conversations)):
                if conversations[i]['from'] == 'human':
                    break
                
                if conversations[i]['from'] == 'gpt':
                    if conversations[i]['value'].startswith('<think>') and "<think>\n\n</think>" not in conversations[i]['value']:
                        is_thinking_mode = True
                    else:
                        is_thinking_mode = False
                    gpt_idx_list.append(i)
            return is_thinking_mode, gpt_idx_list
        
        i = 0
        while i < len(conversations):
            human_idx = find_next_human(conversations, i)
            if human_idx == len(conversations):
                break
            is_thinking_mode, gpt_idx_list = get_next_all_gpt_is_not_thinking(conversations, human_idx+1)
            if not is_thinking_mode and len(gpt_idx_list) > 0:
                conversations[human_idx]['value'] = conversations[human_idx]['value'] + '/nothink'
                for gpt_idx in gpt_idx_list:
                    conversations[gpt_idx]['value'] = '<think>\n\n</think>\n\n' + conversations[gpt_idx]['value']
            i = human_idx + 1
        return conversations
                
    def convert_old_data_format_to_new(self, data_item):
        image = data_item.get("image", None)
        video = data_item.get("video", None)
        image_data = []
        video_data = []
        data_item["data"]  = []
        if image is not None:
            if isinstance(image, str) or isinstance(image, dict):
                image_list = [image]
            elif isinstance(image, list):
                image_list = image
            else:
                raise ValueError(f'Invalid image format: {type(image)}')
            
            for img in image_list:
                if isinstance(img, str):
                    assert img.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'], f'Invalid image file: {img}'
                    image_data.append(osp.join(self.root, img))
                elif isinstance(img, dict):
                    # special case for video
                    """
                    image: dict(video='path/to/video')
                    """
                    if 'video' in img:
                        video_data.append(osp.join(self.root, img['video']))
                    elif 'lmdb_file' in img and 'lmdb_key' in img:
                        # lmdb case
                        image_data.append(img)
                    else:
                        raise ValueError(f'Invalid image format: {type(img)}')
                else:
                    raise ValueError(f'Invalid image format: {type(img)}')

        if video is not None:
            video_list = []
            if isinstance(video, str) or isinstance(video, dict):
                video_list = [video]
            elif isinstance(video, list):
                video_list = video
            else:
                raise ValueError(f'Invalid video format: {type(video)}')
            
            for video in video_list:
                if isinstance(video, str):
                    video_data.append(dict(video=osp.join(self.root, video)))
                elif isinstance(video, dict):
                    if 'video' in video:
                        video["video"] = osp.join(self.root, video['video'])
                        video_data.append(video)
                    else:
                        raise ValueError(f'Invalid video format: {type(video)}')
                else:
                    raise ValueError(f'Invalid video format: {type(video)}')
        
        if len(image_data) > 0:
            data_item["data"].append({"type": "image", "image_list": image_data})
        if len(video_data) > 0:
            data_item["data"].append({"type": "video", "video_list": video_data})
        return data_item
    
    def auto_sample_router(self, data_item):
        """
        In our latest data definition, we use the following keys to indicate the type of data
        {
            "conversations": [{"from": "human", "value": "hello<image-1><image-2><image-3><video-1><video-2><video-3>"}, {"from": "gpt", "value": "world"}],
            "data": [
                {
                    "type": "image",
                    "image_list": ["image1.jpg", dict(lmdb_file="lmdb_file", lmdb_key="lmdb_key"), "image3.jpg"],   
                },
                {
                    "type": "video",
                    "video_list": [dict(video="video1.mp4", start_time=10, end_time=20), dict(video="video2.mp4", start_time=10, end_time=20), dict(video="video3.mp4")],
                }
            ]
        }
        If you only have one image or one video, the tag could be <image> or <video> without index.
          
        We are also compatible with various old data definitions, which are as follows:
        {
            "conversations": [{"from": "human", "value": "hello<image-1><image-2><image-3><video-1><video-2><video-3>"}, {"from": "gpt", "value": "world"}],
            "image": ["image1.jpg", "image2.jpg", "image3.jpg"],
            "video": ["video1.mp4", "video2.mp4", "video3.mp4"],
        }
        """
    
        if "data" not in data_item and ("image" in data_item or "video" in data_item):
            data_item = self.convert_old_data_format_to_new(data_item)
    
        if 'data' not in data_item or len(data_item.get('data', [])) == 0:
            # no other modalities, it is a pure text dataset
            return self.pure_text_get_item(data_item)
        else:
            return self.multi_modal_get_item_one4all(data_item)
           

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.raw_data)
        while True:
            try:
                data_item = json.loads(bz2.decompress(self.raw_data[i]).decode('utf-8'))
                ret = self.auto_sample_router(data_item)
                break
            except Exception as e:
                data_item = json.loads(bz2.decompress(self.raw_data[i]).decode('utf-8'))
                print(data_item)
                print(f'Error in {self.ds_name}, {e}')
                traceback.print_exc()
                # print(e, data_item)
                return None
        return ret

    
def process_dataset(dataset, rank, se):
    start, end = se
    res = []
    # Add tqdm progress bar to display processing progress, current chunk range and rank information
    for i in range(start, end): # position parameter ensures progress bars don't overlap in multiprocessing
        item = dataset[i]
        if item is None: continue
        if isinstance(item, list):
            res.extend(item)
        elif isinstance(item, dict):
            res.append(item)
        else:
            raise ValueError(f'Unknown data type: {type(item)}')
    return res
        

def merge_dataset_shards(data_args, ds_name, ds_extra_tag, world_size, new_ds_collections, ds_collections):
    """Merge dataset shards into a single parquet file.
    
    Args:
        data_args: Data training arguments containing save_root
        ds_name: Name of the dataset
        ds_extra_tag: Extra tag for the dataset (e.g. for video datasets)
        world_size: Number of shards to merge
        new_ds_collections: Dictionary to store updated dataset metadata
        ds_collections: Original dataset collections
    """
    print(f'Merging shards for {ds_name}')
    all_shards = []
    for r in range(world_size):
        # if shard has zero skip
        shard_path = osp.join(data_args.save_root, f'{ds_name}{ds_extra_tag}.{r}.parquet')
        shard = pq.read_table(shard_path)
        if shard.num_rows == 0:
            continue
        all_shards.append(shard)
    
    if len(all_shards) == 0:
        print(f'No shards found for {ds_name}{ds_extra_tag}, skip this')
        return new_ds_collections
    else:  
        print(f'Merging {len(all_shards)} shards for {ds_name}{ds_extra_tag}')
        
    merged_table = pa.concat_tables(all_shards)
    final_path = osp.join(data_args.save_root, f'{ds_name}{ds_extra_tag}.parquet')
    pq.write_table(merged_table, final_path, compression='zstd')
    
    # save merged_table to jsonl
    merged_table.to_pandas().to_json(osp.join(data_args.save_root, f'{ds_name}{ds_extra_tag}.jsonl'), orient='records', lines=True)
    # Update metadata
    new_ds_collections[ds_name]["annotation"] = final_path
    new_ds_collections[ds_name]["length"] = len(merged_table)
    
    # Clean up shards
    for r in range(world_size):
        shard_path = osp.join(data_args.save_root, f'{ds_name}{ds_extra_tag}.{r}.parquet')
        os.remove(shard_path)
        
    # Write MD5
    annotation_md5 = calculate_md5(ds_collections[ds_name]['annotation'])
    with open(osp.join(data_args.save_root, f'{ds_name}{ds_extra_tag}.parquet.md5'), 'w') as f:
        f.write(annotation_md5)
    
    return new_ds_collections

def build_datasets(data_args, tokenizer, tcs_loader=None, num_image_token=256, group_by_length=False,
                   dynamic_image_size=False, use_thumbnail=False, min_dynamic_tiles=1,
                   max_dynamic_tiles=6, normalize_type='imagenet', use_single_process=False):
    # initialize multiprocessing, set spawn mode
    num_processes = 16
    ds_collections = json.loads(open(data_args.meta_path).read())
    
    new_ds_collections = deepcopy(ds_collections)
    
    # Distribute datasets across ranks
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    ds_names = list(ds_collections.keys())
    # Sort datasets by length in descending order
    ds_names.sort(key=lambda x: ds_collections[x].get('length', 0), reverse=True)
    
    cached_ds_collections = {}
    for ds_name in ds_names:
        try:
            ds_modality = ds_collections[ds_name].get('modility', [])
            # for video dataset, we use extra tag to indicate the min and max number of frames and fps
            ds_extra_tag = f'MIN_{data_args.min_num_frames}_MAX_{data_args.max_num_frames}_FPS_{data_args.default_fps}' if 'video' in ds_modality else ''
            if osp.isfile(osp.join(data_args.save_root, f'{ds_name}{ds_extra_tag}.parquet')) and osp.isfile(osp.join(data_args.save_root, f'{ds_name}{ds_extra_tag}.parquet.md5')):
                annotation_md5 = calculate_md5(ds_collections[ds_name]['annotation'])
                cache_md5 = open(osp.join(data_args.save_root, f'{ds_name}{ds_extra_tag}.parquet.md5')).read().strip()
                if (annotation_md5 == cache_md5) and int(os.environ.get('SKIP_CACHE', 0)) == 0:
                    print(f'{ds_name} already exists, skip this')
                    cached_ds_collections[ds_name] = ds_collections[ds_name]
                    cached_ds_collections[ds_name].update({'annotation': osp.join(data_args.save_root, f'{ds_name}{ds_extra_tag}.parquet')})
                    parquet_data = pq.read_table(osp.join(data_args.save_root, f'{ds_name}{ds_extra_tag}.parquet'))
                    cached_ds_collections[ds_name].update({'length': len(parquet_data)})
                    continue
                else:
                    if rank == 0:
                        print(f'{ds_name} already exists, but md5 mismatch, remove it')
                        # os.remove(osp.join(data_args.save_root, f'{ds_name}{ds_extra_tag}.parquet'))
                        try:
                            os.remove(osp.join(data_args.save_root, f'{ds_name}{ds_extra_tag}.parquet.md5'))
                        except Exception as e:
                            print(f"Error removing {ds_name}{ds_extra_tag}.parquet.md5: {e}")
        except Exception as e:
            logger.info(f'Error in loading dataset: {ds_name}, {e}')
            continue
    
        if 'max_dynamic_tiles' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_tiles']
            logger.info(f'max_dynamic_tiles is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_tiles
        try:
            dataset = LazySupervisedDataset(
                data_args.conv_style, ds_collections[ds_name],
                tokenizer,
                tcs_loader,
                num_image_token=num_image_token,
                image_size=data_args.force_image_size,
                pad2square=data_args.pad2square,
                dynamic_image_size=dynamic_image_size,
                use_thumbnail=use_thumbnail,
                min_dynamic_tiles=min_dynamic_tiles,
                max_dynamic_tiles=max_num,
                sample_length_div=data_args.sample_length_div,
                sample_frame_strategy=data_args.sample_frame_strategy,
                tile_downsample_size=data_args.tile_downsample_size,
                max_token_per_sample=data_args.max_token_per_sample,
                default_fps=data_args.default_fps,
                max_num_frames=data_args.max_num_frames,
                min_num_frames=data_args.min_num_frames,
                auto_thinking_prompt=data_args.auto_thinking_prompt,
            )
        except Exception as e:
            logger.info(f'Error in loading dataset: {ds_name}, {e}')
            continue
        dataset.ds_name = ds_name
    
        func = partial(process_dataset, dataset, rank)
        local_results = []
        use_single_process = False
        print(f'Processing chunks for {dataset.ds_name}, rank{rank}, length {len(dataset)}')
        if use_single_process:
            for i in range(0, len(dataset), 1024):
                local_results.extend(func((i, min(i + 1024, len(dataset)))))
        else:
            chunks = [(i, min(i + 1024, len(dataset))) for i in range(0, len(dataset), 1024)]
            with mp.Pool(processes=num_processes) as mp_pool:
                for result in tqdm(mp_pool.imap_unordered(func, chunks), total=len(chunks), desc=f'Processing {dataset.ds_name}'):
                    local_results.extend(result)
                
        # Save shard to parquet
        shard_path = osp.join(data_args.save_root, f'{dataset.ds_name}{ds_extra_tag}.{rank}.parquet')
        os.makedirs(data_args.save_root, exist_ok=True)
        
        table = pa.Table.from_pandas(pd.DataFrame(local_results))
        pq.write_table(table, shard_path, compression='zstd')
        print(f'Saved shard {rank} to {shard_path}')
        
        # Clean up
        del dataset
        del local_results
        gc.collect()
        dist.barrier()
        if rank == 0:
            # Rank 0 merges all shards
            try:
                new_ds_collections = merge_dataset_shards(data_args, ds_name, ds_extra_tag, world_size, new_ds_collections, ds_collections)
            except Exception as e:
                traceback.print_exc()
                print(f"Error merging {ds_name}{ds_extra_tag}: {e}")

    dist.barrier()
    if rank == 0:
        for ds_name in ds_collections.keys():
            if ds_name in cached_ds_collections:
                new_ds_collections[ds_name] = cached_ds_collections[ds_name]
                continue

        # Save final metadata
        new_meta_path = data_args.meta_path.replace(".json", ".prepared.json")
        with open(new_meta_path, 'w') as f:
            json.dump(new_ds_collections, f, indent=4)
        print(f"Finished saving meta file to {new_meta_path}")
        
        total_length = sum(item.get('length', 0)*item.get('repeat_time', 1) for item in new_ds_collections.values())
        print(f"Total length sum: {total_length}")


    dist.barrier()
    exit(0)

def main():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    num_new_tokens = tokenizer.add_tokens(special_tokens_list, special_tokens=True)
    
    # corner case: for specific model, the assistant token is not in the tokenizer
    if len(tokenizer.encode("assistant")) > 1:
        tokenizer.add_tokens(["assistant"], special_tokens=False)
        num_new_tokens += 1

    build_datasets(
        data_args, tokenizer, num_image_token=256, group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_tiles=data_args.min_dynamic_tiles, max_dynamic_tiles=data_args.max_dynamic_tiles,
        normalize_type=data_args.normalize_type)
    os._exit(0)
if __name__ == '__main__':
    main()
