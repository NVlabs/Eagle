

from fractions import Fraction
from typing import List
import numpy as np
from PIL import Image
import collections
from typing import Optional

import lmdb
import io
import cv2

def load_image_from_lmdb(lmdb_file, lmdb_key):

    try:
        env = lmdb.open(lmdb_file, readonly=True, lock=False, max_readers=10240)
    except Exception as e:
        raise ValueError(f"Fail to open LMDB file {lmdb_file}")
    
    with env.begin(write=False) as txn:
        try:
            image_bin = txn.get(lmdb_key.encode('ascii'))
            buf = io.BytesIO(image_bin)
        except Exception as e:
            raise ValueError(f"Fail to get image from LMDB file {lmdb_file}")

    try:
        pil_image = Image.open(buf)
    except Exception as e:
        image_np = np.frombuffer(image_bin, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
    return pil_image



def decord_read(video_path: str, frame_indices: List[int]=None, timestamps: List[float]=None):
    from decord import VideoReader
    # ftype = filetype.guess(video_path)
    video_reader = VideoReader(video_path, num_threads=2)
    # get total frames
    total_frames = len(video_reader)
    # get real fps
    input_fps = video_reader.get_avg_fps()
    
    # if timestamps is not None, we need to update the frame_indices
    if timestamps is not None:
        frame_indices = [int(timestamp * input_fps) for timestamp in timestamps]

    # reset frame out of range
    frame_indices = [max(0, min(total_frames-1, frame_idx)) for frame_idx in frame_indices] 
    # remap_frame_indices, because some indices might be repeated
    original_frame_indices = frame_indices
    frame_indices = list(dict.fromkeys(frame_indices))
    frame_idx_to_image_dict = {}
    for frame_idx in frame_indices:
        frame_img = video_reader[frame_idx].asnumpy()
        frame_idx_to_image_dict[frame_idx] = Image.fromarray(frame_img)
    frames = [frame_idx_to_image_dict[frame_idx] for frame_idx in original_frame_indices]
    return frames


def pyav_read(video_path: str, frame_indices=None, timestamps=None):
    import av
    container = av.open(video_path)
    container.streams.video[0].thread_type = 3
    video_stream = container.streams.video[0]
    input_fps: Fraction = video_stream.average_rate
    input_tb: Fraction = video_stream.time_base
    frame_duration = round(1 / (input_fps * input_tb))
    duration_seconds = video_stream.duration * input_tb
    total_frames = round(duration_seconds * input_fps)
    if timestamps is not None:
        frame_indices = [round(timestamp * input_fps) for timestamp in timestamps]
    frame_indices = [max(0, min(total_frames - 1, frame_idx)) for frame_idx in frame_indices]
    original_frame_indices = frame_indices
    frame_idx_to_image_dict = {}
    for frame_idx in frame_indices:
        timestamp = round(frame_idx * frame_duration)
        container.seek(timestamp, stream=video_stream)
        for frame in container.decode(video=0):
            if frame.pts >= timestamp:
                frame_idx_to_image_dict[frame_idx] = Image.fromarray(frame.to_ndarray(format='rgb24'))
                break
    frames = [frame_idx_to_image_dict[frame_idx] for frame_idx in original_frame_indices]
    return frames


def torchcodec_read(
    video_path: str,
    frame_indices: Optional[List[int]] = None,
    timestamps: Optional[List[float]] = None,
) -> List[Image.Image]:
    """
    A high-performance replacement for `decord_read` implemented with TorchCodec.

    Returns
    -------
    List[PIL.Image]
        Frames as PIL images, ordered to match the requested indices/timestamps.

    Notes
    -----
    - Uses batch decoding (`get_frames_at` / `get_frames_played_at`) which is faster
      than fetching frames one-by-one.
    - Sets `dimension_order="NHWC"` so conversion to PIL does not require a tensor permute.
    - Defaults to `seek_mode="approximate"` for faster decoder creation; switch to
      `"exact"` if you need perfect index-accurate seeking on tricky/VFR content.
    """
    from torchcodec.decoders import VideoDecoder
    # Create a decoder; "approximate" makes creation very fast, while still fine for most cases.
    # Use dimension_order="NHWC" to directly map to PIL (H, W, C) without permuting.
    decoder = VideoDecoder(
        video_path,
        seek_mode="approximate",
        dimension_order="NHWC",
        num_ffmpeg_threads=0,  # let FFmpeg decide optimal threading
        device="cpu",
    )

    # Total number of frames (same semantics as len(VideoReader) in decord).
    total_frames = len(decoder)

    # Average FPS for timestamp -> index mapping (fallback to header value if needed).
    m = decoder.metadata
    avg_fps = float(m.average_fps if m.average_fps is not None else (m.average_fps_from_header or 0.0))

    # If timestamps are given, mimic decord: convert to frame indices via average FPS.
    if timestamps is not None:
        if avg_fps > 0:
            frame_indices = [int(ts * avg_fps) for ts in timestamps]
        else:
            # If FPS is unavailable, fall back to time-based batch retrieval.
            # Clamp timestamps to a safe, valid playback range.
            begin_s = float(m.begin_stream_seconds if m.begin_stream_seconds is not None else 0.0)
            end_s = float(
                m.end_stream_seconds
                if m.end_stream_seconds is not None
                else (m.duration_seconds if m.duration_seconds is not None else begin_s)
            )
            eps = 1.0 / 1000.0  # guard to keep inside half-open range
            seconds = [max(begin_s, min(end_s - eps, float(s))) for s in timestamps]

            batch = decoder.get_frames_played_at(seconds=seconds)  # (N, H, W, C), uint8
            data = batch.data.cpu().numpy()
            return [Image.fromarray(data[i]) for i in range(data.shape[0])]

    if frame_indices is None:
        raise ValueError("Either `frame_indices` or `timestamps` must be provided.")

    # Clamp indices into valid range, like the original decord version.
    frame_indices = [max(0, min(total_frames - 1, int(idx))) for idx in frame_indices]

    # Deduplicate while preserving order to avoid redundant decoding work.
    unique_indices = list(dict.fromkeys(frame_indices))

    # Batch-decode the unique set of frames (faster than per-frame calls).
    batch = decoder.get_frames_at(unique_indices)  # FrameBatch with .data of shape (N, H, W, C)
    data = batch.data.cpu().numpy()

    # Build an index -> PIL image map, then reconstruct in the original order.
    idx_to_img = {idx: Image.fromarray(data[i]) for i, idx in enumerate(unique_indices)}
    frames = [idx_to_img[idx] for idx in frame_indices]
    return frames



def read_frames_sequential(video_path: str, frame_indices: List[int]=None, timestamps: List[float]=None):
    assert frame_indices is not None or timestamps is not None
    frames = torchcodec_read(video_path, frame_indices, timestamps)
    return frames



def get_frames_for_multiple_videos_and_images(video_frame_or_image_tuples):
    # [("video1.mp4", 1), ("video1.mp4", 2), ("video2.mp4", 1), ("video2.mp4", 2), Image1, Image2]
    videos_frame_dict = collections.defaultdict(list)
    image_list = []
    
    for i, image_target in enumerate(video_frame_or_image_tuples):
        if isinstance(image_target, list):
            # video case
            video_path, frame_index = image_target
            videos_frame_dict[video_path].append((i, frame_index))
        else:
            # str and dict case
            # image case
            image_list.append((i, image_target))

    output_images = [None] * len(video_frame_or_image_tuples)
    for video_path in videos_frame_dict:
        videos_frame_dict[video_path].sort(key=lambda x: x[1])
        frame_indices = [x[1] for x in videos_frame_dict[video_path]]
        images = read_frames_sequential(video_path, frame_indices)
        for video_frame_idx, image in zip(videos_frame_dict[video_path], images):
            output_images[video_frame_idx[0]] = image

    for image in image_list:
        if isinstance(image[1], dict):
            output_images[image[0]] = load_image_from_lmdb(image[1]['lmdb_file'], image[1]['lmdb_key'])
        else:
            output_images[image[0]] = Image.open(image[1])
    
    assert None not in output_images
    return output_images

