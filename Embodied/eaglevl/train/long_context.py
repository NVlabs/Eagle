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

"""Configuration helpers for memory-efficient long-context training."""

from typing import Optional


SUPPORTED_ROPE_SCALING_TYPES = {"dynamic", "linear", "yarn"}


def configure_text_config_for_long_context(
    text_config,
    requested_length: int,
    *,
    rope_scaling_type: Optional[str] = "auto",
    rope_scaling_factor: Optional[float] = None,
    original_max_position_embeddings: Optional[int] = None,
) -> Optional[dict]:
    """Extend a Hugging Face text config to the requested context length.

    ``auto`` keeps the pretrained configuration within its native context and
    selects YaRN only when the requested length exceeds it. Passing ``none``
    explicitly changes only ``max_position_embeddings`` and is intended for
    models whose RoPE implementation already handles extrapolation.
    """
    requested_length = int(requested_length)
    if requested_length < 1:
        raise ValueError("requested_length must be positive")

    existing_scaling = getattr(text_config, "rope_scaling", None)
    existing_original_length = (
        existing_scaling.get("original_max_position_embeddings")
        if isinstance(existing_scaling, dict)
        else None
    )
    configured_length = int(
        getattr(text_config, "max_position_embeddings", requested_length)
    )
    native_length = int(
        original_max_position_embeddings
        or existing_original_length
        or configured_length
    )
    scaling_type = (rope_scaling_type or "auto").lower()

    if scaling_type in {"none", "disabled"}:
        text_config.max_position_embeddings = max(configured_length, requested_length)
        text_config.rope_scaling = None
        return None
    if requested_length <= configured_length and rope_scaling_factor is None:
        return existing_scaling

    if scaling_type == "auto":
        scaling_type = "yarn"
    elif scaling_type not in SUPPORTED_ROPE_SCALING_TYPES:
        supported = ", ".join(sorted(SUPPORTED_ROPE_SCALING_TYPES | {"auto", "none"}))
        raise ValueError(f"Unsupported RoPE scaling type {scaling_type!r}; choose one of: {supported}")

    factor = float(rope_scaling_factor or (requested_length / native_length))
    if factor < 1.0:
        raise ValueError("rope_scaling_factor must be >= 1.0")
    scaled_length = int(round(native_length * factor))
    if scaled_length < requested_length:
        raise ValueError(
            f"rope_scaling_factor={factor} only covers {scaled_length} tokens from the "
            f"native {native_length}, below requested_length={requested_length}."
        )

    rope_scaling = {
        "rope_type": scaling_type,
        "factor": factor,
    }
    if scaling_type in {"dynamic", "yarn"}:
        rope_scaling["original_max_position_embeddings"] = native_length

    text_config.max_position_embeddings = scaled_length
    text_config.rope_scaling = rope_scaling
    return rope_scaling
