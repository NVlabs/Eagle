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
# This file is adopted from the Hugging Face Transformers project
# (https://github.com/huggingface/transformers), licensed under the Apache License, Version 2.0.
#
# Modifications © 2026 NVIDIA CORPORATION & AFFILIATES, licensed under
# the Apache License, Version 2.0.
#
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""
Processor class for EagleEmbodied.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, List, Union, Literal
import base64
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
import re
import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from typing import Optional, Any
import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
import lmdb
import cv2
import pickle
import decord
import functools
import signal

logger = logging.get_logger(__name__)

VIDEO_READ_TIMEOUT = int(os.environ.get("VIDEO_READ_TIMEOUT", "10"))
FPS = 2.0
MAX_FRAMES = 64
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 32000 * 28 * 28 * 0.9)))
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")


def _video_timeout_handler(signum, frame):
    raise TimeoutError("Video reading timed out")


def _call_backend_with_timeout(backend_func, ele, timeout_sec):
    """Call a single video backend with its own SIGALRM timeout.
    Each backend gets a fresh alarm so one backend's timeout doesn't
    leave subsequent backends unprotected.
    """
    if timeout_sec <= 0:
        return backend_func(ele)
    old_handler = signal.signal(signal.SIGALRM, _video_timeout_handler)
    signal.alarm(timeout_sec)
    try:
        return backend_func(ele)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")

def read_img_from_lmdb_v2(image_data):
    # special case for AgiBotWorld
    lmdb_file, lmdb_key = image_data['lmdb_file'], image_data['lmdb_key']
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

def parse_lmdb_image_data(image_data):
    lmdb_file = image_data['lmdb_file']
    if not os.path.exists(lmdb_file):
        source, separator, target = os.environ.get('EAGLE_LMDB_REMAP', '').partition('=')
        if separator and source and lmdb_file.startswith(source):
            lmdb_file = target + lmdb_file[len(source):]
            image_data['lmdb_file'] = lmdb_file
        if not os.path.exists(lmdb_file):
            raise ValueError(
                f"LMDB file {lmdb_file} does not exist. "
                "Set EAGLE_LMDB_REMAP=/old/prefix=/new/prefix when paths were relocated."
            )
    # special case for AgiBotWorld
    if 'AgiBotWorld' in image_data['lmdb_file']:
        return read_img_from_lmdb_v2(image_data)

    try:
        env = lmdb.open(image_data['lmdb_file'], readonly=True, lock=False, max_readers=10240)
    except Exception as e:
        print(f"Failed to open lmdb file {image_data['lmdb_file']}. Error message: {e}", flush=True)
        raise e

    with env.begin(write=False) as txn:
        try:
            image_bin = txn.get(image_data['lmdb_key'].encode('ascii'))
            buf = BytesIO(image_bin)
        except Exception as e:
            print(f"Failed to get image from lmdb file {image_data['lmdb_file']}. Error message: {e}", flush=True)
            raise e
    try:
        image = Image.open(buf)
    except Exception as e:
        image_np = np.frombuffer(image_bin, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
    return image


def fetch_video_frame_as_image(video_path: str, frame_index: int, video_reader_backend: str = None) -> Image.Image:
    """Decode exactly one video frame for an image-conditioned sample."""
    if not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError(f"frame_index must be a non-negative integer, got {frame_index!r}")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    if video_reader_backend == 'pyav':
        # Reuse the existing strict frame-clip reader, which handles codecs
        # unsupported by Decord (including the AgiBot training videos). The
        # one-frame half-open clip preserves the annotated frame index. Disable
        # reader-side resizing; the ordinary image processor owns that budget.
        frames, _, _ = _call_backend_with_timeout(_read_video_pyav, {
            'video': video_path, 'clip_start_frame': frame_index,
            'clip_end_frame': frame_index + 1, 'nframes': 1, 'max_frames': 1,
            'video_total_pixels': 2**31, 'decoder_threads': 1,
            'exact_frame_clip': True,
        }, VIDEO_READ_TIMEOUT)
        return Image.fromarray(frames[0].permute(1, 2, 0).cpu().numpy())

    # Do not use the legacy read_frames_sequential helper here: its decoding
    # fallback intentionally substitutes a black 224x224 image on failure,
    # which would silently corrupt supervised image samples.
    reader = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
    if frame_index >= len(reader):
        raise IndexError(
            f"frame_index {frame_index} is outside video with {len(reader)} frames: "
            f"{video_path}"
        )
    frame = reader[frame_index]
    # Decord's bridge is process-global.  Whole-video decoding switches it to
    # torch, so mixed image/video packs may return either a torch.Tensor or an
    # NDArray here depending on which sample ran first in the worker.
    if isinstance(frame, torch.Tensor):
        frame_array = frame.detach().cpu().numpy()
    elif hasattr(frame, "asnumpy"):
        frame_array = frame.asnumpy()
    else:
        frame_array = np.asarray(frame)
    return Image.fromarray(frame_array)


def fetch_image(ele: dict[str, str | Image.Image], video_reader_backend: str = None) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, dict) and 'lmdb_file' in image:
        image_obj = parse_lmdb_image_data(image)
    elif isinstance(image, dict) and 'video' in image and 'frame_index' in image:
        image_obj = fetch_video_frame_as_image(
            image['video'],
            int(image['frame_index']),
            video_reader_backend=video_reader_backend,
        )
    elif isinstance(image, dict) and 'file' in image:
        # Plain image file wrapped in a dict so it can carry per-image options
        # such as ``resize`` (see below).
        image_obj = Image.open(image['file'])
    elif isinstance(image, str) and (image.startswith("http://") or image.startswith("https://")):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif isinstance(image, str) and image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif isinstance(image, str) and image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    elif isinstance(image, str):
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    # Optional exact stretch to ``[width, height]`` (aspect ratio NOT kept).
    # Dense-trajectory rows use it to feed every frame as 378x378, matching the
    # MolmoMotion recipe; the image processor then pads 378 -> 392 (14 x 2 x 2)
    # so each frame costs 196 tokens.
    resize = image.get("resize") if isinstance(image, dict) else None
    if resize is None:
        resize = ele.get("resize")
    if resize is not None:
        if len(resize) != 2 or int(resize[0]) <= 0 or int(resize[1]) <= 0:
            raise ValueError(f"image resize must be [width, height] > 0, got {resize!r}")
        image_obj = image_obj.resize((int(resize[0]), int(resize[1])), Image.BICUBIC)
    image = to_rgb(image_obj)

    return image


def resize_images_to_fit_total_pixels(
    images: List[Image.Image],
    total_pixels: int = None
) -> List[Image.Image]:
    """
    Resize multiple images to ensure total pixels don't exceed the limit.
    Iteratively reduce each image by 10% until total pixels fit within the limit.

    Args:
        images: List of PIL Images to resize
        total_pixels: Maximum total pixels allowed. Defaults to VIDEO_TOTAL_PIXELS.

    Returns:
        List of resized PIL Images
    """
    if not images or len(images) <= 1:
        return images

    if total_pixels is None:
        total_pixels = VIDEO_TOTAL_PIXELS

    num_images = len(images)

    # Calculate target pixels per image (same size for all)
    target_pixels_per_image = total_pixels // num_images

    # Resize each image to fit within target_pixels_per_image
    resized_images = []
    for img in images:
        resized_img = img
        # Iteratively reduce by 10% until within target
        while resized_img.width * resized_img.height > target_pixels_per_image:
            new_width = max(1, int(resized_img.width * 0.9))
            new_height = max(1, int(resized_img.height * 0.9))
            resized_img = resized_img.resize((new_width, new_height), Image.BICUBIC)
        resized_images.append(resized_img)

    return resized_images


def get_video_frame_indices(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> tuple[torch.Tensor, float]:
    target_fps = ele.get("fps", FPS)
    max_frames = ele.get("max_frames", MAX_FRAMES)

    if ele.get("exact_frame_clip"):
        # The single-frame clip of fetch_video_frame_as_image.
        clip_start_value = ele.get("clip_start_frame")
        clip_end_value = ele.get("clip_end_frame")
        clip_start = 0 if clip_start_value is None else int(clip_start_value)
        clip_end = total_frames if clip_end_value is None else int(clip_end_value)
        # Frame clips use Python's half-open [start, end) convention.  Invalid
        # empty/out-of-range clips are rejected rather than silently training on
        # the last frame with the wrong label.
        clip_start = max(0, clip_start)
        clip_end = min(clip_end, total_frames)
        if clip_start >= total_frames or clip_end <= clip_start:
            raise ValueError(
                f"Invalid video frame clip [{clip_start}, {clip_end}) for "
                f"{total_frames} frames"
            )
    else:
        clip_start = ele.get("clip_start_frame", 0)
        clip_end = ele.get("clip_end_frame", total_frames)
        clip_start = max(0, min(clip_start, total_frames))
        clip_end = max(clip_start + 1, min(clip_end, total_frames))
    clip_len = clip_end - clip_start

    nframes = (clip_len / video_fps) * target_fps
    nframes = int(round(nframes))
    nframes = max(1, nframes)

    if nframes > max_frames:
        nframes = max_frames

    nframes = min(nframes, clip_len)

    if nframes == clip_len:
        idx = torch.arange(clip_start, clip_end).long()
    else:
        idx = torch.linspace(clip_start, clip_end - 1, nframes).round().long()

    sample_fps = nframes / max(clip_len, 1e-6) * video_fps

    return idx, sample_fps

def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float, list):
    """read video using torchvision.io.read_video and return also per-frame timestamps"""
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()

    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    idx, sample_fps = get_video_frame_indices(ele, total_frames, video_fps)

    start_time = ele.get("video_start", 0.0)
    timestamps = (start_time + idx.to(torch.float32) / video_fps).tolist()

    video = video[idx]
    return video, sample_fps, timestamps


def is_decord_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("decord") is not None


def is_pyav_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("av") is not None

def _compute_decode_resolution(orig_h, orig_w, nframes, video_total_pixels):
    """Compute target decode resolution to stay within the pixel budget."""
    current_pixels = nframes * orig_h * orig_w
    if current_pixels <= video_total_pixels:
        return orig_h, orig_w
    scale = math.sqrt(video_total_pixels / current_pixels)
    target_h = max(2, int(orig_h * scale))
    target_w = max(2, int(orig_w * scale))
    return target_h, target_w


def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float, list):
    """Read video using decord with decode-time resize for efficiency."""
    video_path = ele["video"]
    st = time.time()

    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()

    idx_tensor, sample_fps = get_video_frame_indices(ele, total_frames, video_fps)
    nframes = len(idx_tensor)

    video_total_pixels = ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS)
    probe_frame = vr[0]
    orig_h, orig_w = probe_frame.shape[0], probe_frame.shape[1]

    target_h, target_w = _compute_decode_resolution(
        orig_h, orig_w, nframes, video_total_pixels
    )
    resized_during_decode = False
    if target_h < orig_h or target_w < orig_w:
        try:
            del vr
            vr = decord.VideoReader(video_path, width=target_w, height=target_h)
            resized_during_decode = True
        except Exception:
            vr = decord.VideoReader(video_path)

    idx = idx_tensor.tolist()
    try:
        decord.bridge.set_bridge('torch')
        video = vr.get_batch(idx)
        video = video.permute(0, 3, 1, 2)
    except Exception:
        decord.bridge.set_bridge('native')
        video = vr.get_batch(idx).asnumpy()
        video = torch.from_numpy(video.copy()).permute(0, 3, 1, 2)

    start_time = ele.get("video_start", 0.0)
    timestamps = [start_time + i / video_fps for i in idx]

    logger.info(
        f"decord: {video_path=}, {total_frames=}, sampled={nframes}, "
        f"resized_decode={resized_during_decode}, shape={tuple(video.shape)}, "
        f"time={time.time() - st:.3f}s"
    )
    return video, sample_fps, timestamps


def _read_video_pyav(
    ele: dict,
) -> (torch.Tensor, float, list):
    """Read video using PyAV with early-stop and selective frame decoding."""
    import av

    video_path = ele["video"]
    st = time.time()
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        if ele.get('decoder_threads') is not None:
            stream.codec_context.thread_count = int(ele['decoder_threads'])
            stream.thread_type = "SLICE"
        else:
            stream.thread_type = "AUTO"

        video_fps = float(stream.average_rate or stream.guessed_rate or 25)
        start_time = ele.get("video_start", 0.0)
        end_time = ele.get("video_end", None)
        # fetch_video_frame_as_image counts frames from the stream start; other
        # callers keep the absolute PTS (offset 0).
        stream_start_pts = int(stream.start_time or 0) if ele.get("exact_frame_clip") else 0
        stream_start_seconds = (
            float(stream_start_pts * stream.time_base)
            if stream.time_base is not None
            else 0.0
        )

        total_frames_meta = 0
        if stream.frames and stream.frames > 0:
            total_frames_meta = stream.frames
        elif stream.duration and stream.time_base:
            duration_sec = float(stream.duration * stream.time_base)
            if end_time is not None:
                duration_sec = min(duration_sec, end_time) - start_time
            else:
                duration_sec -= start_time
            total_frames_meta = max(1, int(duration_sec * video_fps))

        if ele.get("exact_frame_clip") and total_frames_meta <= 0 and (
            ele.get("clip_start_frame") is not None
            or ele.get("clip_end_frame") is not None
        ):
            raise ValueError(
                "PyAV cannot apply an absolute frame clip because the video "
                f"reports neither frame count nor duration: {video_path}"
            )

        # If caller provides a frame-index clip, seek directly to the clip
        # start to avoid scanning millions of frames in multi-episode chunks.
        clip_start_frame = ele.get("clip_start_frame", None)
        seek_done = False
        if (
            clip_start_frame is not None
            and clip_start_frame > 0
            and stream.time_base is not None
            and video_fps > 0
        ):
            seek_sec = float(clip_start_frame) / video_fps
            try:
                container.seek(
                    stream_start_pts + int(seek_sec / stream.time_base),
                    stream=stream,
                    any_frame=False,
                    backward=True,
                )
                seek_done = True
            except Exception:
                seek_done = False
        elif start_time > 0 and stream.time_base is not None:
            container.seek(
                stream_start_pts + int(start_time / stream.time_base),
                stream=stream,
            )

        if total_frames_meta > 0:
            idx_tensor, sample_fps = get_video_frame_indices(ele, total_frames_meta, video_fps)
            target_indices = set(idx_tensor.tolist())
            last_needed = max(target_indices)

            sampled_frames = {}
            frame_num = 0
            for frame in container.decode(video=0):
                if frame.pts is not None and stream.time_base is not None:
                    frame_time = (
                        float(frame.pts * stream.time_base) - stream_start_seconds
                    )
                    if frame_time < start_time - 0.5:
                        continue
                    if end_time is not None and frame_time > end_time + 0.5:
                        break
                    # After seeking by clip_start_frame, frame_num is no
                    # longer the absolute frame index. Recover it from pts.
                    if seek_done and video_fps > 0:
                        frame_num = int(round(frame_time * video_fps))
                if frame_num in target_indices:
                    sampled_frames[frame_num] = frame.to_ndarray(format="rgb24")
                    if len(sampled_frames) == len(target_indices):
                        break
                if not seek_done:
                    frame_num += 1
                if frame_num > last_needed + 1:
                    break

            if not sampled_frames:
                raise ValueError(f"Video has 0 decodable frames: {video_path}")

            ordered = []
            fallback_frame = next(iter(sampled_frames.values()))
            for i in idx_tensor.tolist():
                ordered.append(sampled_frames.get(i, fallback_frame))
            sampled = np.stack(ordered)
        else:
            all_frames = []
            for frame in container.decode(video=0):
                if frame.pts is not None and stream.time_base is not None:
                    frame_time = (
                        float(frame.pts * stream.time_base) - stream_start_seconds
                    )
                    if frame_time < start_time - 0.5:
                        continue
                    if end_time is not None and frame_time > end_time + 0.5:
                        break
                all_frames.append(frame.to_ndarray(format="rgb24"))

            total_frames_meta = len(all_frames)
            if total_frames_meta == 0:
                raise ValueError(f"Video has 0 decodable frames: {video_path}")

            idx_tensor, sample_fps = get_video_frame_indices(ele, total_frames_meta, video_fps)
            sampled = np.stack([all_frames[i] for i in idx_tensor.tolist()])

        video = torch.from_numpy(sampled).permute(0, 3, 1, 2)
        timestamps = (start_time + idx_tensor.to(torch.float32) / video_fps).tolist()
        logger.info(
            f"pyav: {video_path=}, total={total_frames_meta}, sampled={len(sampled)}, time={time.time() - st:.3f}s"
        )
        return video, sample_fps, timestamps
    finally:
        container.close()


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "pyav": _read_video_pyav,
    "torchvision": _read_video_torchvision,
}

BACKEND_FALLBACK_ORDER = ["decord", "pyav"]


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if is_decord_available():
        return "decord"
    if is_pyav_available():
        return "pyav"
    return "torchvision"


def fetch_video(ele: dict, return_video_sample_fps: bool = False, video_reader_backend: str = None) -> torch.Tensor | list[Image.Image]:
    """
    Fetches video, samples frames, resizes based on video_total_pixels,
    and returns as uint8 Tensor (TCHW).  Each backend gets its own
    SIGALRM timeout so a hang in one backend cannot steal the timeout
    budget from the next.
    """
    if isinstance(ele["video"], str):
        video_reader_backend = video_reader_backend if video_reader_backend is not None else get_video_reader_backend()
        last_error = None
        tried = set()
        for backend in [video_reader_backend] + BACKEND_FALLBACK_ORDER:
            if backend in tried or backend not in VIDEO_READER_BACKENDS:
                continue
            if backend == "decord" and not is_decord_available():
                continue
            if backend == "pyav" and not is_pyav_available():
                continue

            tried.add(backend)
            try:
                video, sample_fps, timestamps = _call_backend_with_timeout(
                    VIDEO_READER_BACKENDS[backend], ele, VIDEO_READ_TIMEOUT
                )
                break
            except Exception as e:
                last_error = e
                logger.warning(f"video_reader_backend {backend} error, msg: {e}")
        else:
            raise RuntimeError(f"All video backends failed for {ele.get('video', 'unknown')}") from last_error

        nframes, _, height, width = video.shape
        if nframes == 0:
            raise ValueError(f"Video has 0 frames after sampling: {ele.get('video', 'unknown')}")

        video_total_pixels = ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS)
        current_pixels = nframes * height * width

        if current_pixels > video_total_pixels:
            scale_factor = math.sqrt(video_total_pixels / current_pixels)
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)

            video = transforms.functional.resize(
                video,
                [new_height, new_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )

        if return_video_sample_fps:
            return video, sample_fps, timestamps
        return video

    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)

        images = [
            fetch_image({"image": video_element, **process_info})
            for video_element in ele["video"]
        ]
        if len(images) == 0:
            raise ValueError("Video frame list is empty")

        nframes = len(images)
        timestamps = [-1 for i in range(nframes)]

        # For list of images, we return list of PIL images directly,
        # the processor will handle conversion to tensor later.
        if return_video_sample_fps:
            return images, process_info.get("fps", 2.0), timestamps
        return images

class EagleEmbodiedProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
        "videos_kwargs": {},
    }


class EagleEmbodiedProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "num_image_tokens",
        "image_token",
        "video_token",
        "images_kwargs",
        "videos_kwargs",
        "text_kwargs",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token='<IMG_CONTEXT>',
        video_token='<IMG_CONTEXT>',
        merge_kernel_size=[2, 2], # Note: This might need adjustment based on your patch_size (14*14)
        image_placeholder='image',
        video_placeholder='video',
        image_start_token='<img>',
        image_end_token='</img>',
        **kwargs,
    ):
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.video_token = tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.image_placeholder = image_placeholder
        self.video_placeholder = video_placeholder
        self.merge_kernel_size = merge_kernel_size
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        if 'auto_map' in kwargs:
            self.auto_map = kwargs['auto_map']
        super().__init__(image_processor, tokenizer, chat_template=chat_template)


    def replace_media_placeholder(self, text, image_list, video_list, timestamps_list, fps_list, **output_kwargs):

        num_of_images_in_this_sample = 0
        num_of_videos_in_this_sample = 0
        pattern = re.compile(rf"<({self.image_placeholder}|{self.video_placeholder})-(\d+)>")
        unified_frame_list = []

        def replace_in_text(text):
            def repl(match):
                nonlocal unified_frame_list
                nonlocal num_of_images_in_this_sample
                nonlocal num_of_videos_in_this_sample
                media_type = match.group(1)
                idx_in_list = int(match.group(2)) - 1
                idx_mapper = {0: "first", 1: "second", 2: "third", 3: "fourth", 4: "fifth", 5: "sixth", 6: "seventh", 7: "eighth", 8: "ninth", 9: "tenth"}

                if media_type == 'image':
                    if idx_in_list >= len(image_list):
                        raise ValueError(f"Image placeholder index {idx_in_list + 1} out of range, total images: {len(image_list)}")
                    # Call EagleEmbodiedImageProcessor with a single image in a list
                    image_inputs = self.image_processor(images=[image_list[idx_in_list]], **output_kwargs["images_kwargs"])

                    num_of_tokens_list = [int(h * w) // (self.image_processor.merge_kernel_size[0] * self.image_processor.merge_kernel_size[1]) for h, w in image_inputs['image_grid_hws']]

                    special_placeholder = f"<image {idx_in_list+1}>{self.image_start_token}{self.image_token * num_of_tokens_list[0]}{self.image_end_token}"
                    unified_frame_list.append(image_inputs)
                    num_of_images_in_this_sample += 1

                elif media_type == 'video':
                    if idx_in_list >= len(video_list):
                        raise ValueError(f"Video placeholder index {idx_in_list + 1} out of range, total videos: {len(video_list)}")
                    video_obj = video_list[idx_in_list]

                    if isinstance(video_obj, torch.Tensor):
                        if video_obj.ndim != 4 or video_obj.shape[0] == 0:
                            raise ValueError(f"Decoded video tensor is empty or invalid: shape={tuple(video_obj.shape)}")
                        video_inputs = self.image_processor.preprocess_video_tensor(video_obj)
                    elif isinstance(video_obj, list):
                        if len(video_obj) == 0:
                            raise ValueError("Decoded video frame list is empty")
                        video_inputs = self.image_processor(images=video_obj, **output_kwargs["videos_kwargs"])
                    else:
                        raise ValueError("Unsupported video format")
                    if len(video_inputs["image_grid_hws"]) == 0:
                        raise ValueError("Video preprocessing produced 0 frames")

                    # Calculate tokens per frame
                    num_of_tokens_list = [int(h * w) // (self.image_processor.merge_kernel_size[0] * self.image_processor.merge_kernel_size[1]) for h, w in video_inputs['image_grid_hws']]

                    if (
                        timestamps_list is not None
                        and idx_in_list < len(timestamps_list)
                        and -1 not in timestamps_list[idx_in_list]
                    ):
                        frame_timestamps = timestamps_list[idx_in_list]
                    else:
                        frame_timestamps = None
                    sampled_fps = fps_list[idx_in_list] if fps_list is not None and idx_in_list < len(fps_list) else None

                    if frame_timestamps is not None:
                        # Ensure lengths match (sometimes rounding might cause off-by-one if not careful, but usually safe here)
                        if len(frame_timestamps) != len(num_of_tokens_list):
                             logger.warning(f"Timestamp mismatch: {len(frame_timestamps)} vs {len(num_of_tokens_list)}")
                             min_len = min(len(frame_timestamps), len(num_of_tokens_list))
                             frame_timestamps = frame_timestamps[:min_len]
                             num_of_tokens_list = num_of_tokens_list[:min_len]

                        special_placeholder = [f"Frame-{i+1}-{frame_timestamps[i]:.2f}s: {self.image_start_token}{self.image_token * num_of_tokens}{self.image_end_token}" for i, num_of_tokens in enumerate(num_of_tokens_list)]
                    else:
                        special_placeholder = [f"Frame-{i+1}: {self.image_start_token}{self.image_token * num_of_tokens}{self.image_end_token}" for i, num_of_tokens in enumerate(num_of_tokens_list)]

                    if sampled_fps is not None:
                        special_placeholder = f"The {idx_mapper[idx_in_list]} video sampled with {sampled_fps:.2f} fps: " + "".join(special_placeholder)
                    else:
                        special_placeholder = f"The {idx_mapper[idx_in_list]} video: " + "".join(special_placeholder)

                    unified_frame_list.append(video_inputs)
                    num_of_videos_in_this_sample += 1
                else:
                    raise ValueError(f'Unknown media type: {media_type}')
                return special_placeholder
            return pattern.sub(repl, text)

        text = replace_in_text(text)

        if len(unified_frame_list) > 0:
            # Concatenate all pixel values from all images/videos in this sample
            pixel_values = torch.cat([frame['pixel_values'] for frame in unified_frame_list], dim=0)
            # Concatenate grid hws
            image_grid_hws = np.concatenate([frame['image_grid_hws'] for frame in unified_frame_list], axis=0)
        else:
            pixel_values = torch.empty(0)
            image_grid_hws = np.empty(0)

        return text, pixel_values, image_grid_hws, num_of_images_in_this_sample, num_of_videos_in_this_sample

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        video_total_pixels = None,
        **kwargs: Unpack[EagleEmbodiedProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            EagleEmbodiedProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text_list = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")
        elif isinstance(text, list) and isinstance(text[0], str):
            text_list = text

        if images is None: images = []
        if videos is None: videos = []

        # Resize multiple images to fit within VIDEO_TOTAL_PIXELS
        if len(images) > 1:
            max_pixels = video_total_pixels if video_total_pixels is not None else VIDEO_TOTAL_PIXELS
            images = resize_images_to_fit_total_pixels(images, max_pixels)

        pixel_values_list = []
        image_grid_hws_list = []
        new_sample_list = []
        image_start_idx = 0
        video_start_idx = 0
        timestamps_batch = output_kwargs['videos_kwargs'].pop("timestamps", None)
        fps_batch = output_kwargs['videos_kwargs'].pop("fps", None)

        for sample in text_list:
            timestamps_list = timestamps_batch[video_start_idx:] if timestamps_batch is not None else None
            fps_list = fps_batch[video_start_idx:] if fps_batch is not None else None

            sample, pixel_values, image_grid_hws, num_of_images_in_this_sample, num_of_videos_in_this_sample = self.replace_media_placeholder(
                sample, images[image_start_idx:], videos[video_start_idx:], timestamps_list, fps_list, **output_kwargs
            )
            new_sample_list.append(sample)

            if pixel_values.numel() > 0:
                pixel_values_list.append(pixel_values)
                image_grid_hws_list.append(image_grid_hws)

            image_start_idx += num_of_images_in_this_sample
            video_start_idx += num_of_videos_in_this_sample

        image_inputs = {}
        if len(pixel_values_list) > 0:
            # Concatenate across the batch
            image_inputs['pixel_values'] = torch.cat(pixel_values_list, dim=0)
            image_inputs['image_grid_hws'] = np.concatenate(image_grid_hws_list, axis=0)

        video_inputs = {} # Video data is merged into image_inputs now
        text_inputs = self.tokenizer(new_sample_list, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        outputs = super().save_pretrained(save_directory, **kwargs)
        return outputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if isinstance(processor, tuple):
            processor = processor[0]
        return processor

    def process_vision_info(
        self,
        conversations: list[dict] | list[list[dict]],
        return_video_kwargs: bool = False,
        video_reader_backend: str = None,
    ) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

        vision_infos = self.extract_vision_info(conversations)
        image_inputs = []
        video_inputs = []
        video_sample_fps_list = []
        video_timestamps_list = []

        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info, video_reader_backend=video_reader_backend))
            elif "video" in vision_info:
                video_input, video_sample_fps, video_timestamps = fetch_video(vision_info, return_video_sample_fps=True, video_reader_backend=video_reader_backend)
                video_sample_fps_list.append(video_sample_fps)
                video_inputs.append(video_input)
                video_timestamps_list.append(video_timestamps)
            else:
                raise ValueError("image, image_url or video should in content.")

        if len(image_inputs) == 0:
            image_inputs = None
        if len(video_inputs) == 0:
            video_inputs = None

        if return_video_kwargs:
            return image_inputs, video_inputs, {'fps': video_sample_fps_list, 'timestamps': video_timestamps_list}
        return image_inputs, video_inputs

    def extract_vision_info(self, conversations: list[dict] | list[list[dict]]) -> list[dict]:
        vision_infos = []
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")
                        ):
                            vision_infos.append(ele)
        return vision_infos

    def py_apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        assert tokenize == False, "tokenize is not supported yet"
        result = ""
        image_count = 0
        video_count = 0

        thinking_suffix = " /think" if enable_thinking else " /no_think"

        # Find the index of the last user message
        last_user_idx = -1
        for idx, message in enumerate(messages):
            if message.get('role') == 'user':
                last_user_idx = idx

        message_text = ""
        last_user_text = ""
        for idx, message in enumerate(messages):
            if message.get('role') != 'user': continue
            content = message.get('content')
            if isinstance(content, str):
                message_text += content
                if idx == last_user_idx:
                    last_user_text += content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        message_text += item["text"]
                        if idx == last_user_idx:
                            last_user_text += item["text"]
                    elif isinstance(item, str):
                        message_text += item
                        if idx == last_user_idx:
                            last_user_text += item

        # auto_thinking_handler (training) may already inject "/think" or "/no_think"
        # into the last user turn. Avoid appending the suffix a second time, otherwise
        # the model sees e.g. " /no_think /no_think" which differs from training.
        last_user_text = last_user_text.rstrip().lower()
        last_user_has_thinking_marker = last_user_text.endswith(
            ("/think", "/no_think", "/nothink")
        )

        for idx, message in enumerate(messages):
            if idx == 0 and message.get('role') != 'system':
                result += "<|im_start|>system\n"
                result += "You are a helpful assistant.\n"
                result += "<|im_end|>\n"

            result += f"<|im_start|>{message.get('role', '')}\n"
            content = message.get('content')

            is_last_user_message = (message.get('role') == 'user' and idx == last_user_idx)
            should_append_thinking_suffix = (
                is_last_user_message and not last_user_has_thinking_marker
            )

            if isinstance(content, str):
                result += content
                if should_append_thinking_suffix:
                    result += thinking_suffix
                result += "<|im_end|>\n"
            else:
                for item in content:
                    if (isinstance(item, dict) and (item.get('type') == 'image' or 'image' in item or 'image_url' in item)):
                        image_count += 1
                        candidate_token = f"<image-{image_count}>"
                        if candidate_token not in message_text:
                            result += candidate_token
                    elif (isinstance(item, dict) and (item.get('type') == 'video' or 'video' in item)):
                        video_count += 1
                        candidate_token = f"<video-{video_count}>"
                        if candidate_token not in message_text:
                            result += candidate_token
                    elif isinstance(item, dict) and 'text' in item:
                        result += item['text']
                    elif isinstance(item, str):
                        result += item
                if should_append_thinking_suffix:
                    result += thinking_suffix
                result += "<|im_end|>\n"

        if add_generation_prompt:
            result += "<|im_start|>assistant\n"

        return result


    @classmethod
    def from_args_and_dict(cls, args, processor_dict: dict[str, Any], **kwargs):
        processor_dict = processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        if "processor_class" in processor_dict:
            del processor_dict["processor_class"]

        validated_init_kwargs = cls.validate_init_kwargs(
            processor_config=processor_dict, valid_kwargs=cls.valid_kwargs
        )
        if isinstance(validated_init_kwargs, tuple):
            unused_kwargs = validated_init_kwargs[0]
        else:
            unused_kwargs = validated_init_kwargs
        processor = cls(*args, **processor_dict)

        for key in set(kwargs.keys()):
            if hasattr(processor, key):
                setattr(processor, key, kwargs.pop(key))

        if isinstance(unused_kwargs, dict):
            kwargs.update(unused_kwargs)
        logger.info(f"Processor {processor}")
        if return_unused_kwargs:
            return processor, kwargs
        else:
            return processor


__all__ = ["EagleEmbodiedProcessor"]
