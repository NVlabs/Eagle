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
# This file is adopted from the DFlash project
# (https://github.com/z-lab/dflash), licensed under the MIT License.
#
# The draft attention and decoder layers also derive, through DFlash, from the
# Qwen3 modeling code of the Hugging Face Transformers project
# (https://github.com/huggingface/transformers), licensed under the Apache License,
# Version 2.0. Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
#
# Modifications © 2026 NVIDIA CORPORATION & AFFILIATES, licensed under
# the Apache License, Version 2.0.

"""Qwen3 draft network used by strict VFlash speculative decoding.

This module intentionally contains neither token embeddings nor an LM head.  The
target model supplies both: its hidden states condition the draft network, its
embedding table represents the noisy block, and its LM head projects the draft
hidden states to token logits.

The architecture is adapted from ``dflash/model.py`` in the official Z Lab
DFlash repository: https://github.com/z-lab/dflash.  The upstream source is
licensed under the MIT License reproduced below.

MIT License

Copyright (c) 2026 Z Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import Unpack

# NVIDIA's PyTorch 25.06 image pairs its 2.8 nightly with an internal Triton
# build that exposes triton_key from runtime.cache, while that TorchInductor
# revision still imports it from compiler.compiler. Bridge the API locations
# before any compiled FlexAttention graph reaches Inductor's cache-key path.
try:
    import triton.compiler.compiler as _triton_compiler
    if not hasattr(_triton_compiler, "triton_key"):
        from triton.runtime.cache import triton_key as _triton_key
        _triton_compiler.triton_key = _triton_key
except ImportError:
    pass

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
        flex_attention,
    )
    _HAS_FLEX_ATTENTION = True
except ImportError:  # torch<2.5; inference remains available.
    BlockMask = None
    create_block_mask = None
    flex_attention = None
    _HAS_FLEX_ATTENTION = False

try:
    _TORCH_MAJOR_MINOR = tuple(
        int(piece) for piece in torch.__version__.split("+", 1)[0].split(".")[:2]
    )
except (TypeError, ValueError):
    _TORCH_MAJOR_MINOR = (0, 0)

if (
    _HAS_FLEX_ATTENTION
    and hasattr(torch, "compile")
    and (2, 8) <= _TORCH_MAJOR_MINOR < (2, 9)
):
    # Calling flex_attention eagerly dispatches its HOP through an eager debug
    # backend, which materializes dense QxK scores.  The outer Inductor compile
    # is mandatory for the block-sparse CUDA kernel. ``dynamic=True`` keeps
    # ragged per-step totals from forcing one graph per exact Q/K length.
    try:
        _compiled_flex_attention = torch.compile(
            flex_attention, backend="inductor", fullgraph=True, dynamic=True
        )
        _compiled_create_block_mask = torch.compile(
            create_block_mask, backend="inductor", fullgraph=True, dynamic=True
        )
    except Exception:  # VFlash training entrypoint will report a clean error.
        _compiled_flex_attention = None
        _compiled_create_block_mask = None
else:
    _compiled_flex_attention = None
    _compiled_create_block_mask = None

from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)


VFLASH_NUM_DRAFT_LAYERS = 5


def module_weight_fingerprint(module: nn.Module, *, chunk_bytes: int = 16 << 20) -> str:
    """Hash logical model state, independent of checkpoint path or sharding."""
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    digest = hashlib.sha256()
    digest.update(b"eagle-state-dict-sha256-v1\0")
    state = module.state_dict()
    for name in sorted(state):
        if name == "vflash_draft" or name.startswith("vflash_draft."):
            continue
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Target state entry {name!r} is not a tensor")
        if tensor.layout != torch.strided:
            raise TypeError(f"Target state entry {name!r} is not a dense tensor")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        if tensor.numel() == 0:
            continue
        flat = tensor.detach().reshape(-1)
        elements_per_chunk = max(1, chunk_bytes // flat.element_size())
        for start in range(0, flat.numel(), elements_per_chunk):
            byte_view = flat[start:start + elements_per_chunk].contiguous().view(torch.uint8)
            digest.update(byte_view.cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _is_flex_block_mask(value) -> bool:
    return _HAS_FLEX_ATTENTION and isinstance(value, BlockMask)


def build_vflash_flex_block_mask(
    *,
    query_context_starts: torch.LongTensor,
    query_anchors: torch.LongTensor,
    query_draft_starts: torch.LongTensor,
    query_draft_ends: torch.LongTensor,
    target_length: int,
) -> "BlockMask":
    """Build one sparse joint-forward mask for all independent draft blocks."""
    if not _HAS_FLEX_ATTENTION:
        raise RuntimeError(
            "Strict sparse VFlash training requires compiled torch FlexAttention "
            "from the supported PyTorch >=2.8,<2.9 runtime"
        )
    vectors = (
        query_context_starts,
        query_anchors,
        query_draft_starts,
        query_draft_ends,
    )
    query_length = int(query_context_starts.numel())
    if query_length <= 0 or any(
        vector.ndim != 1 or vector.numel() != query_length for vector in vectors
    ):
        raise ValueError("VFlash per-query sparse-mask vectors must be equal nonzero 1D tensors")
    if any(vector.device != query_context_starts.device for vector in vectors):
        raise ValueError("VFlash sparse-mask vectors must share one device")
    target_length = int(target_length)
    if target_length < 0:
        raise ValueError("target_length must be non-negative")

    def mask_mod(batch_index, head_index, query_index, key_index):
        del batch_index, head_index
        target_key = key_index < target_length
        target_visible = (
            target_key
            & (key_index >= query_context_starts[query_index])
            & (key_index < query_anchors[query_index])
        )
        draft_key = key_index - target_length
        draft_visible = (
            (~target_key)
            & (draft_key >= query_draft_starts[query_index])
            & (draft_key < query_draft_ends[query_index])
        )
        return target_visible | draft_visible

    if query_context_starts.is_cuda:
        if _compiled_create_block_mask is None:
            raise RuntimeError(
                "Strict VFlash sparse-mask construction requires compiled CUDA "
                "FlexAttention support"
            )
        mask_builder = _compiled_create_block_mask
    else:
        mask_builder = create_block_mask
    return mask_builder(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=query_length,
        KV_LEN=target_length + query_length,
        device=str(query_context_starts.device),
        BLOCK_SIZE=128,
        _compile=False,
    )


def build_target_layer_ids(
    num_target_layers: int,
    num_draft_layers: int = VFLASH_NUM_DRAFT_LAYERS,
) -> list[int]:
    """Select evenly spaced target decoder layers, following official DFlash.

    Target layer indices are zero-based decoder-layer indices.  The first
    selected layer is 1 and the final selected layer is ``N - 3``; this avoids
    the target's shallowest and final two layers.  For a 36-layer target and a
    five-layer draft this returns ``[1, 9, 17, 25, 33]``.

    ``extract_context_feature`` accounts for the embedding entry at index zero
    in Hugging Face's ``output_hidden_states`` tuple.
    """

    if not isinstance(num_target_layers, int) or isinstance(num_target_layers, bool):
        raise TypeError("num_target_layers must be an integer")
    if not isinstance(num_draft_layers, int) or isinstance(num_draft_layers, bool):
        raise TypeError("num_draft_layers must be an integer")
    if num_target_layers <= 0:
        raise ValueError("num_target_layers must be positive")
    if num_draft_layers <= 0:
        raise ValueError("num_draft_layers must be positive")

    if num_draft_layers == 1:
        return [num_target_layers // 2]

    # There must be enough integer positions in [1, N - 3] to select unique
    # layers.  Duplicate context layers silently reduce the FC input rank.
    if num_target_layers < num_draft_layers + 3:
        raise ValueError(
            "num_target_layers must be at least num_draft_layers + 3 for "
            "unique DFlash context layers"
        )

    start = 1
    end = num_target_layers - 3
    span = end - start
    layer_ids = [
        int(round(start + (index * span) / (num_draft_layers - 1)))
        for index in range(num_draft_layers)
    ]
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("uniform target-layer selection produced duplicate layers")
    return layer_ids


def validate_strict_vflash_config(draft_config, target_config) -> None:
    """Reject draft configs that do not clone the target's Qwen3 topology."""
    expected_layers = build_target_layer_ids(
        int(target_config.num_hidden_layers), VFLASH_NUM_DRAFT_LAYERS
    )
    binding = getattr(draft_config, "dflash_config", {}) or {}
    mismatches = []

    exact_values = {
        "num_hidden_layers": VFLASH_NUM_DRAFT_LAYERS,
        "num_target_layers": int(target_config.num_hidden_layers),
        "block_size": 16,
        "vocab_size": int(target_config.vocab_size),
    }
    for name, expected in exact_values.items():
        if int(getattr(draft_config, name, -1)) != expected:
            mismatches.append(name)

    if list(getattr(draft_config, "layer_types", []) or []) != (
        ["full_attention"] * VFLASH_NUM_DRAFT_LAYERS
    ):
        mismatches.append("layer_types")
    if not bool(getattr(draft_config, "use_cache", False)):
        mismatches.append("use_cache")
    if binding.get("target_layer_ids") != expected_layers:
        mismatches.append("target_layer_ids")

    # These inherited fields determine parameter shapes or forward semantics.
    # JSON normalization also compares nested RoPE settings recursively.
    inherited_fields = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_act",
        "rms_norm_eps",
        "attention_bias",
        "attention_dropout",
        "max_position_embeddings",
        "rope_theta",
        "rope_scaling",
        "rope_parameters",
        "use_sliding_window",
        "sliding_window",
        "max_window_layers",
    )
    for name in inherited_fields:
        draft_value = getattr(draft_config, name, None)
        target_value = getattr(target_config, name, None)
        if json.dumps(
            draft_value, sort_keys=True, separators=(",", ":"), default=str
        ) != json.dumps(
            target_value, sort_keys=True, separators=(",", ":"), default=str
        ):
            mismatches.append(name)

    if mismatches:
        raise ValueError(
            "VFlash drafter config does not match strict target-derived topology: "
            + ", ".join(sorted(set(mismatches)))
        )


def extract_context_feature(
    hidden_states: Sequence[torch.Tensor],
    layer_ids: Sequence[int],
    *,
    hidden_state_offset: int = 1,
) -> torch.Tensor:
    """Concatenate selected target hidden states along the feature dimension.

    Args:
        hidden_states: The target model's ``output_hidden_states`` sequence.
            Hugging Face causal decoders put the embedding output at index 0.
        layer_ids: Zero-based target decoder-layer indices.
        hidden_state_offset: Offset from decoder-layer indices to entries in
            ``hidden_states``.  The default of 1 matches Qwen3.

    Returns:
        A tensor shaped ``[batch, context_length, hidden * len(layer_ids)]``.

    The result is not detached.  Callers that freeze the target should produce
    its hidden states under ``torch.no_grad()`` or detach them explicitly.
    """

    if not isinstance(hidden_state_offset, int) or isinstance(hidden_state_offset, bool):
        raise TypeError("hidden_state_offset must be an integer")
    if hidden_state_offset < 0:
        raise ValueError("hidden_state_offset must be non-negative")
    if not layer_ids:
        raise ValueError("layer_ids must contain at least one target layer")
    if not hidden_states:
        raise ValueError("hidden_states must not be empty")

    selected_states: list[torch.Tensor] = []
    reference_shape: Optional[torch.Size] = None
    reference_device: Optional[torch.device] = None
    reference_dtype: Optional[torch.dtype] = None
    for layer_id in layer_ids:
        if not isinstance(layer_id, int) or isinstance(layer_id, bool):
            raise TypeError("every layer_id must be an integer")
        if layer_id < 0:
            raise ValueError("layer_ids must be non-negative")
        state_index = layer_id + hidden_state_offset
        if state_index >= len(hidden_states):
            raise IndexError(
                f"target layer {layer_id} maps to hidden_states[{state_index}], "
                f"but only {len(hidden_states)} states were provided"
            )
        state = hidden_states[state_index]
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"hidden_states[{state_index}] must be a torch.Tensor")
        if state.ndim != 3:
            raise ValueError(
                f"hidden_states[{state_index}] must have shape [batch, sequence, hidden]"
            )

        if reference_shape is None:
            reference_shape = state.shape[:-1]
            reference_device = state.device
            reference_dtype = state.dtype
        elif state.shape[:-1] != reference_shape:
            raise ValueError("selected target hidden states have different batch/sequence shapes")
        elif state.device != reference_device:
            raise ValueError("selected target hidden states are on different devices")
        elif state.dtype != reference_dtype:
            raise ValueError("selected target hidden states have different dtypes")
        selected_states.append(state)

    return torch.cat(selected_states, dim=-1)


def weighted_draft_cross_entropy(
    logits: torch.Tensor,
    labels: torch.LongTensor,
    loss_weight: Optional[torch.Tensor] = None,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute token-weighted CE for aligned, parallel VFlash predictions.

    Unlike causal-LM loss, this helper performs no one-token shift: draft logits
    and labels must already describe the same block positions.  With ``mean``
    reduction, the weighted loss is normalized by the sum of valid weights.
    Ignored positions contribute neither numerator nor denominator.
    """

    if logits.ndim < 2:
        raise ValueError("logits must have at least two dimensions [..., vocab]")
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must equal logits.shape[:-1] "
            f"{tuple(logits.shape[:-1])}"
        )
    if labels.dtype != torch.long:
        raise TypeError("labels must have dtype torch.long")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: 'none', 'mean', or 'sum'")

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid = flat_labels.ne(ignore_index)
    token_loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction="none",
    )

    if loss_weight is None:
        effective_weight = valid.to(dtype=token_loss.dtype)
    else:
        if not isinstance(loss_weight, torch.Tensor):
            loss_weight = torch.as_tensor(loss_weight, device=logits.device)
        else:
            loss_weight = loss_weight.to(device=logits.device)
        try:
            loss_weight = torch.broadcast_to(loss_weight, labels.shape)
        except RuntimeError as error:
            raise ValueError(
                f"loss_weight shape {tuple(loss_weight.shape)} is not broadcastable "
                f"to labels shape {tuple(labels.shape)}"
            ) from error
        if not torch.is_floating_point(loss_weight):
            loss_weight = loss_weight.to(dtype=token_loss.dtype)
        if not torch.isfinite(loss_weight).all():
            raise ValueError("loss_weight must contain only finite values")
        if (loss_weight < 0).any():
            raise ValueError("loss_weight must be non-negative")
        effective_weight = loss_weight.reshape(-1).to(dtype=token_loss.dtype)
        effective_weight = effective_weight * valid.to(dtype=token_loss.dtype)

    weighted_loss = token_loss * effective_weight
    if reduction == "none":
        return weighted_loss.reshape_as(labels)
    if reduction == "sum":
        return weighted_loss.sum()

    denominator = effective_weight.sum()
    # A graph-connected zero is preferable to NaN when a packed batch has no
    # valid draft targets (or all valid positions have zero weight).
    return torch.where(
        denominator > 0,
        weighted_loss.sum() / denominator.clamp_min(torch.finfo(token_loss.dtype).tiny),
        logits.sum() * 0.0,
    )


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply all current positions to K and only noisy-block positions to Q."""

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_length = query.shape[-2]
    if cos.shape[-2] != key.shape[-2]:
        raise ValueError(
            "RoPE position count must equal the current context-plus-noise key length"
        )
    if query_length > key.shape[-2]:
        raise ValueError("query length cannot exceed current key length")

    query_cos = cos[..., -query_length:, :]
    query_sin = sin[..., -query_length:, :]
    query = (query * query_cos) + (rotate_half(query) * query_sin)
    key = (key * cos) + (rotate_half(key) * sin)
    return query, key


class Qwen3DFlashAttention(nn.Module):
    """Non-causal DFlash attention: Q=noise, K/V=context+noise."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_types = getattr(config, "layer_types", None)
        layer_type = layer_types[layer_idx] if layer_types is not None else "full_attention"
        self.sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, query_length, _ = hidden_states.shape
        context_length = target_hidden.shape[1]
        query_shape = (
            batch_size,
            query_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        kv_length = context_length + query_length
        kv_shape = (
            batch_size,
            kv_length,
            self.config.num_key_value_heads,
            self.head_dim,
        )

        # Noise is the only source of queries.  Every layer can attend to both
        # projected target context and all noisy positions through K/V.
        query_states = self.q_norm(self.q_proj(hidden_states).view(query_shape)).transpose(1, 2)
        context_and_noise = torch.cat((target_hidden, hidden_states), dim=1)
        key_states = self.k_norm(self.k_proj(context_and_noise).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context_and_noise).view(kv_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        if _is_flex_block_mask(attention_mask):
            if past_key_values is not None:
                raise ValueError("Flex block-sparse VFlash training does not use a KV cache")
            if self.training and self.attention_dropout:
                raise ValueError(
                    "Flex block-sparse VFlash requires attention_dropout=0"
                )
            if not query_states.is_cuda or _compiled_flex_attention is None:
                raise RuntimeError(
                    "Strict VFlash block-sparse training requires compiled CUDA "
                    "FlexAttention; refusing the dense eager fallback"
                )
            attention_output = _compiled_flex_attention(
                query_states,
                key_states,
                value_states,
                block_mask=attention_mask,
                scale=self.scaling,
                enable_gqa=(
                    self.config.num_attention_heads
                    != self.config.num_key_value_heads
                ),
            )
            attention_output = attention_output.transpose(1, 2).reshape(
                batch_size, query_length, -1
            ).contiguous()
            return self.o_proj(attention_output), None

        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        # Transformers' FlashAttention path expects a 2D padding mask.  The
        # training path deliberately accepts a fully explicit 4D additive mask,
        # so use eager attention there and let the mask define all connectivity.
        attention_interface: Callable = eager_attention_forward
        implementation = getattr(self.config, "_attn_implementation", None) or "eager"
        if attention_mask is None or attention_mask.ndim != 4:
            if implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[implementation]

        attention_output, attention_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attention_output = attention_output.reshape(batch_size, query_length, -1).contiguous()
        return self.o_proj(attention_output), attention_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    """A full-width Qwen3 decoder layer with DFlash attention."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        del position_ids  # RoPE was prepared once by DFlashDraftModel.
        if position_embeddings is None:
            raise ValueError("position_embeddings must be provided")

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DFlashDraftModel(Qwen3PreTrainedModel):
    """Five-layer, full-width Qwen3 DFlash draft backbone.

    ``config`` must carry the target model's width/head/MLP/RoPE settings while
    setting ``num_hidden_layers=5`` and ``num_target_layers`` to the original
    target depth.  ``layer_types`` must describe five full-attention layers.
    Optional ``config.dflash_config`` fields are ``target_layer_ids`` (which
    must equal the uniform default) and ``mask_token_id``.  ``block_size`` is
    metadata only: forward accepts variable noisy-block lengths for physical
    atoms.

    A supplied ``DynamicCache`` is mutated in place.  It contains projected
    target context and noisy-block K/V for the draft only; it must never be
    shared with the target model's causal KV cache.
    """

    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        if not isinstance(config, Qwen3Config):
            raise TypeError("DFlashDraftModel requires a Qwen3Config")
        if config.num_hidden_layers != VFLASH_NUM_DRAFT_LAYERS:
            raise ValueError(
                f"strict VFlash requires exactly {VFLASH_NUM_DRAFT_LAYERS} draft layers; "
                f"got {config.num_hidden_layers}"
            )
        if config.hidden_size <= 0 or config.intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")
        if config.num_attention_heads <= 0 or config.num_key_value_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        layer_types = getattr(config, "layer_types", None)
        if layer_types is None or len(layer_types) != VFLASH_NUM_DRAFT_LAYERS:
            raise ValueError("layer_types must contain exactly five draft layer types")
        if any(layer_type != "full_attention" for layer_type in layer_types):
            raise ValueError("strict VFlash draft layers must all use full_attention")

        dflash_config = getattr(config, "dflash_config", None) or {}
        if not isinstance(dflash_config, dict):
            raise TypeError("config.dflash_config must be a dictionary")
        num_target_layers = getattr(config, "num_target_layers", None)
        if num_target_layers is None:
            num_target_layers = dflash_config.get("num_target_layers")
        if num_target_layers is None:
            raise ValueError("config.num_target_layers is required")
        self.num_target_layers = int(num_target_layers)

        uniform_layer_ids = build_target_layer_ids(
            self.num_target_layers,
            VFLASH_NUM_DRAFT_LAYERS,
        )
        configured_layer_ids = dflash_config.get("target_layer_ids")
        if configured_layer_ids is not None:
            configured_layer_ids = [int(layer_id) for layer_id in configured_layer_ids]
            if configured_layer_ids != uniform_layer_ids:
                raise ValueError(
                    "config.dflash_config.target_layer_ids must equal the uniform "
                    f"DFlash selection {uniform_layer_ids}; got {configured_layer_ids}"
                )
        self.target_layer_ids = uniform_layer_ids

        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config, layer_idx)
                for layer_idx in range(VFLASH_NUM_DRAFT_LAYERS)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.block_size = int(getattr(config, "block_size", 16))
        if self.block_size <= 0:
            raise ValueError("config.block_size must be positive")
        self.mask_token_id = dflash_config.get("mask_token_id")
        self.post_init()

    def forward(
        self,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        """Return draft hidden states aligned with ``noise_embedding`` positions.

        Production training passes a compiled FlexAttention ``BlockMask``;
        focused CPU tests may pass a floating 4D additive mask shaped
        ``[batch|1, heads|1, noise_length, total_kv_length]``. For inference,
        pass an independent ``DynamicCache`` with ``use_cache=True`` and
        usually leave ``attention_mask=None`` for fully non-causal attention.
        """

        if noise_embedding.ndim != 3:
            raise ValueError("noise_embedding must have shape [batch, noise_length, hidden]")
        if target_hidden.ndim != 3:
            raise ValueError(
                "target_hidden must have shape [batch, context_length, selected_hidden]"
            )
        batch_size, noise_length, hidden_size = noise_embedding.shape
        if noise_length <= 0:
            raise ValueError("noise_embedding must contain at least one token")
        if hidden_size != self.config.hidden_size:
            raise ValueError(
                f"noise_embedding hidden size must be {self.config.hidden_size}; got {hidden_size}"
            )
        if target_hidden.shape[0] != batch_size:
            raise ValueError("noise_embedding and target_hidden batch sizes must match")
        expected_context_width = len(self.target_layer_ids) * self.config.hidden_size
        if target_hidden.shape[-1] != expected_context_width:
            raise ValueError(
                f"target_hidden width must be {expected_context_width}; "
                f"got {target_hidden.shape[-1]}"
            )
        if target_hidden.device != noise_embedding.device:
            raise ValueError("noise_embedding and target_hidden must be on the same device")

        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError("past_key_values must implement transformers.cache_utils.Cache")
        if use_cache and past_key_values is None:
            raise ValueError(
                "use_cache=True requires a caller-owned DynamicCache so its mutated "
                "state remains available after forward"
            )
        if isinstance(past_key_values, DynamicCache):
            past_length = past_key_values.get_seq_length()
        elif past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values.get_seq_length()

        context_length = target_hidden.shape[1]
        current_kv_length = context_length + noise_length
        total_kv_length = past_length + current_kv_length

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                total_kv_length,
                dtype=torch.long,
                device=noise_embedding.device,
            ).unsqueeze(0)
        elif position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [batch|1, context_length + noise_length]")
        elif position_ids.shape[0] not in (1, batch_size):
            raise ValueError("position_ids batch dimension must be 1 or match the input batch")
        elif position_ids.shape[1] != current_kv_length:
            raise ValueError(
                f"position_ids length must be context_length + noise_length "
                f"({current_kv_length}); got {position_ids.shape[1]}"
            )
        elif position_ids.device != noise_embedding.device:
            raise ValueError("position_ids must be on the same device as noise_embedding")

        if cache_position is None:
            cache_position = torch.arange(
                past_length,
                total_kv_length,
                dtype=torch.long,
                device=noise_embedding.device,
            )
        elif cache_position.ndim != 1 or cache_position.numel() != current_kv_length:
            raise ValueError(
                "cache_position must be one-dimensional with one entry per current "
                "context-plus-noise key"
            )

        if attention_mask is not None and not _is_flex_block_mask(attention_mask):
            if attention_mask.ndim != 4:
                raise ValueError(
                    "explicit DFlash attention_mask must be a 4D additive mask "
                    "[batch|1, heads|1, noise_length, total_kv_length]"
                )
            if not torch.is_floating_point(attention_mask):
                raise TypeError("4D additive attention_mask must have floating dtype")
            if attention_mask.shape[0] not in (1, batch_size):
                raise ValueError("attention_mask batch dimension must be 1 or match inputs")
            if attention_mask.shape[1] not in (1, self.config.num_attention_heads):
                raise ValueError("attention_mask head dimension must be 1 or num_attention_heads")
            if attention_mask.shape[2] != noise_length:
                raise ValueError("attention_mask query dimension must equal noise_length")
            if attention_mask.shape[3] != total_kv_length:
                raise ValueError(
                    f"attention_mask key dimension must be total KV length "
                    f"({total_kv_length}); got {attention_mask.shape[3]}"
                )
            if attention_mask.device != noise_embedding.device:
                raise ValueError("attention_mask must be on the same device as inputs")

        projected_target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(noise_embedding, position_ids)
        hidden_states = noise_embedding
        for layer in self.layers:
            # Keep both tensors positional so reentrant gradient checkpointing
            # sees an input carrying the FC projection's autograd graph.
            hidden_states = layer(
                hidden_states,
                projected_target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)


__all__ = [
    "VFLASH_NUM_DRAFT_LAYERS",
    "DFlashDraftModel",
    "Qwen3DFlashAttention",
    "Qwen3DFlashDecoderLayer",
    "build_vflash_flex_block_mask",
    "build_target_layer_ids",
    "extract_context_feature",
    "module_weight_fingerprint",
    "validate_strict_vflash_config",
    "weighted_draft_cross_entropy",
]
