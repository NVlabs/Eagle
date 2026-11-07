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

"""Generation helpers shared by the VFlash runners and evaluation adapters.

The trajectory runners (robot3d, driving, motion) read prepared conversation
records; the bbox harness and the VLMEvalKit wrapper only use the VFlash call
arguments and the bbox answer prefix.
"""

import json
import re
import time

import torch

# Visual input budget of every prepared trajectory benchmark.
MAX_FRAMES = 16
TARGET_FPS = 2.0
VIDEO_TOTAL_PIXELS = 12845056

_BBOX_QUERY_RE = re.compile(r"Locate every instance of (.+?)\. Return only")


def read_records(path, start=0, end=-1):
    """Return ``(line_index, record)`` for the JSONL lines in ``[start, end)``.

    A negative ``end`` reads to the end of the file. Blank lines are skipped.
    """
    records = []
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            if 0 <= end <= index:
                break
            if line.strip():
                records.append((index, json.loads(line)))
    return records


def count_output_rows(path):
    """Number of JSON rows already written to ``path`` (0 if it does not exist)."""
    count = 0
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    json.loads(line)
                    count += 1
    except FileNotFoundError:
        pass
    return count


def decode_frame_references(messages):
    """Replace ``{video, frame_index}`` and ``{file}`` image references in place.

    Video frames are read at the exact index with OpenCV. Videos that OpenCV
    cannot decode (e.g. AV1) are read with the processor's PyAV reader. An
    optional ``resize`` ``[width, height]`` is applied with bicubic filtering.
    """
    import cv2
    from PIL import Image

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            reference = item.get("image") if item.get("type") == "image" else None
            if not isinstance(reference, dict):
                continue
            video_path = reference.get("video")
            frame_index = reference.get("frame_index")
            if video_path is None or frame_index is None:
                if reference.get("file") is None:
                    continue
                image = Image.open(reference["file"]).convert("RGB")
            else:
                capture = cv2.VideoCapture(str(video_path))
                try:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                    ok, frame = capture.read()
                finally:
                    capture.release()
                if ok and frame is not None:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    from eaglevl.utils.moonvit.processing_eagle_embodied import (
                        fetch_video_frame_as_image,
                    )

                    image = fetch_video_frame_as_image(
                        str(video_path), int(frame_index), video_reader_backend="pyav"
                    )
            resize = reference.get("resize")
            if resize is not None:
                image = image.resize((int(resize[0]), int(resize[1])), Image.BICUBIC)
            item["image"] = image


def build_model_inputs(processor, record, media_root="", decode_frames=False):
    """Tokenize the user turns of a prepared record exactly as in training.

    The thinking handler appends `` /no_think`` to the last user turn, so a
    record whose answer is unknown carries an empty assistant turn.
    ``decode_frames`` decodes frame references here instead of in the processor.
    ``record`` itself is not modified.
    """
    from eaglevl.train.tools import process_multimodal_sample

    record = dict(record, conversations=[dict(turn) for turn in record.get("conversations", [])])
    messages = process_multimodal_sample(
        record,
        media_root=media_root,
        max_frames=MAX_FRAMES,
        target_fps=TARGET_FPS,
        video_total_pixels=VIDEO_TOTAL_PIXELS,
        auto_thinking_handler=True,
    )
    messages = [message for message in messages if message["role"] == "user"]
    if decode_frames:
        decode_frame_references(messages)
    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    return processor(
        text=[text],
        images=images or None,
        videos=videos or None,
        return_tensors="pt",
        padding=True,
        video_total_pixels=VIDEO_TOTAL_PIXELS,
    ).to("cuda", dtype=torch.bfloat16)


def vflash_generate_kwargs(tokenizer, max_new_tokens, draft_temperature=0.0):
    """Arguments of one unverified VFlash ``generate`` call.

    The target decodes greedily; ``draft_temperature`` only changes how the
    drafter samples physical-atom slots (0 selects the argmax).
    """
    return {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "generation_mode": "vflash",
        "verify_atoms": False,
        "return_vflash_stats": True,
        "vflash_draft_temperature": float(draft_temperature),
        "atomic_decoding_tokenizer": tokenizer,
    }


def generate_timed(model, inputs, **kwargs):
    """Run ``model.generate`` and return ``(output_ids, vflash_stats, seconds)``."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    generated = model.generate(**inputs, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if kwargs.get("generation_mode") == "vflash":
        output_ids, _, stats = generated
        return output_ids, stats, elapsed
    return generated, None, elapsed


def vflash_stats_record(stats, tokenizer=None):
    """JSON-serializable statistics of one VFlash call; empty for AR decoding."""
    if stats is None:
        return {}
    lengths = [int(value) for value in stats.acceptance_lengths]
    fallback_details = []
    for detail in getattr(stats, "atom_fallback_details", []):
        token_ids = [int(value) for value in detail.get("proposed_token_ids", [])]
        decoded = {
            "reason": str(detail.get("reason", "")),
            "proposed_token_ids": token_ids,
        }
        if tokenizer is not None:
            decoded["proposed_tokens"] = tokenizer.convert_ids_to_tokens(token_ids)
            decoded["proposed_text"] = tokenizer.decode(
                token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        fallback_details.append(decoded)
    return {
        "num_input_tokens": int(stats.num_input_tokens),
        "num_output_tokens": int(stats.num_output_tokens),
        "time_to_first_token_s": float(stats.time_to_first_token),
        "time_per_output_token_s": float(stats.time_per_output_token),
        "acceptance_lengths": lengths,
        "mean_acceptance_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "atom_fallbacks": int(stats.atom_fallbacks),
        "atom_fallback_reasons": list(getattr(stats, "atom_fallback_reasons", [])),
        "atom_fallback_details": fallback_details,
        "drive_oof_pair_repairs": int(getattr(stats, "drive_oof_pair_repairs", 0)),
        "verify_atoms": bool(stats.verify_atoms),
        "text_block_size": int(getattr(stats, "text_block_size", 16)),
    }


def bbox_hybrid_prefix_ids(messages, tokenizer):
    """Token ids of the answer prefix the bbox hybrid decoder starts from.

    The category is read back from the visual-primitive question of the last
    user turn, so the answer starts with ``<ref>{category}</ref><objects>``.
    """
    question = "\n".join(
        item.get("text", "")
        for message in messages[-1:]
        for item in message.get("content", [])
        if item.get("type") == "text"
    )
    matched = _BBOX_QUERY_RE.search(question)
    if matched is None:
        raise ValueError("Cannot derive the bbox category from the user prompt")
    prefix = "<think>\n\n</think>\n\n<ref>" + matched.group(1) + "</ref><objects>"
    return tokenizer.encode(prefix, add_special_tokens=False)
