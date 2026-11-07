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

"""VFlash drafter training against a frozen Eagle Embodied target.

The training model exposes only the drafter loss, checkpoints contain only the
drafter, and the frozen target is bound to the drafter by its fingerprint.
"""

from __future__ import annotations

import inspect
import os

import torch
import torch.distributed as dist
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from eaglevl.model.vflash.modeling_dflash import (
    DFlashDraftModel,
    build_vflash_flex_block_mask,
    module_weight_fingerprint,
    weighted_draft_cross_entropy,
)
from eaglevl.train.checkpoint_export import export_vflash_draft_checkpoint
from eaglevl.train.liger_loss_weight_ops import LigerFusedLinearCrossEntropyLoss
from eaglevl.train.vflash.loss_weighting import apply_oof_loss_scale


def distributed_target_weight_fingerprint(model) -> str:
    """Hash a large target model once, then broadcast the result/errors."""
    distributed = dist.is_available() and dist.is_initialized()
    payload = [None]
    if not distributed or dist.get_rank() == 0:
        try:
            payload[0] = {
                'ok': True,
                'sha256': module_weight_fingerprint(model),
            }
        except Exception as error:
            payload[0] = {
                'ok': False,
                'error_type': type(error).__name__,
                'error': str(error),
            }
    if distributed:
        dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not result or not result.get('ok'):
        detail = result or {'error_type': 'RuntimeError', 'error': 'missing broadcast'}
        raise RuntimeError(
            'Unable to fingerprint the frozen Eagle target checkpoint '
            f"({detail.get('error_type')}): {detail.get('error')}"
        )
    return result['sha256']


class EagleEmbodiedVFlashTrainingModel(nn.Module):
    """Expose only the VFlash loss while retaining the frozen multimodal target."""

    def __init__(self, target_model, draft_model: DFlashDraftModel):
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.config = draft_model.config
        # The reference drafter configs record that the target stayed frozen.
        self.config.joint_target_training = False
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self.target_model.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_model.train(False)
        self.draft_model.train(mode)
        return self

    def get_input_embeddings(self):
        return self.target_model.language_model.get_input_embeddings()

    def state_dict(self, *args, **kwargs):
        """Persist only the standalone drafter, never the frozen target."""
        return self.draft_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.draft_model.load_state_dict(state_dict, strict=strict, assign=assign)

    def _graph_connected_zero(self):
        zero = None
        for parameter in self.draft_model.parameters():
            current = parameter.reshape(-1)[:1].sum() * 0.0
            zero = current if zero is None else zero + current
        return zero

    def _extract_frozen_target_context(
        self,
        *,
        target_inputs_embeds,
        attention_mask,
        position_ids,
    ):
        """Capture only the target layers consumed by VFlash.

        Hugging Face's ``output_hidden_states=True`` retains every decoder
        layer output until the target forward returns.  VFlash consumes five
        of those tensors, so long packed multimodal sequences paid a large
        allocation and memory-bandwidth tax for the other layers.  Forward
        hooks keep the exact same post-layer tensors without materializing the
        complete hidden-state tuple.
        """

        decoder = self.target_model.language_model.model
        layers = getattr(decoder, "layers", None)
        if layers is None:
            raise AttributeError(
                "VFlash target language model must expose decoder layers as model.layers"
            )

        selected = [None] * len(self.draft_model.target_layer_ids)
        handles = []

        def capture(slot):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(hidden, torch.Tensor):
                    raise TypeError("VFlash target decoder layer did not return a tensor")
                selected[slot] = hidden

            return hook

        try:
            for slot, layer_id in enumerate(self.draft_model.target_layer_ids):
                handles.append(layers[layer_id].register_forward_hook(capture(slot)))
            decoder(
                inputs_embeds=target_inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()

        missing = [
            layer_id
            for layer_id, hidden in zip(self.draft_model.target_layer_ids, selected)
            if hidden is None
        ]
        if missing:
            raise RuntimeError(
                f"Frozen target forward did not expose selected VFlash layers: {missing}"
            )
        return torch.cat(selected, dim=-1).detach()

    def forward(
        self,
        pixel_values,
        input_ids=None,
        attention_mask=None,
        image_grid_hws=None,
        image_flags=None,
        sub_sample_lengths=None,
        labels=None,
        vflash_draft_input_ids=None,
        vflash_direct_labels=None,
        vflash_clean_position_ids=None,
        vflash_draft_position_ids=None,
        vflash_loss_weights=None,
        vflash_block_context_starts=None,
        vflash_block_anchors=None,
        vflash_block_ranges=None,
        vflash_block_sizes=None,
        label_weights=None,
    ):
        del labels  # Clean target labels are intentionally not an AR objective.
        # Per-token data weights are folded into vflash_loss_weights by the
        # collator (see build_online_packed_vflash_batch); nothing to do here.
        del label_weights
        if input_ids is None:
            raise ValueError("VFlash training requires clean target input_ids")
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        with torch.no_grad():
            target_inputs_embeds = self.target_model.prepare_multimodal_inputs_embeds(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_hws=image_grid_hws,
                image_flags=image_flags,
            )
            if vflash_clean_position_ids is None:
                vflash_clean_position_ids = self.target_model._generate_position_ids(
                    target_inputs_embeds, sub_sample_lengths
                )
            elif vflash_clean_position_ids.ndim == 1:
                vflash_clean_position_ids = vflash_clean_position_ids.unsqueeze(0)
            target_context = self._extract_frozen_target_context(
                target_inputs_embeds=target_inputs_embeds,
                attention_mask=attention_mask,
                position_ids=vflash_clean_position_ids,
            )
            del target_inputs_embeds

        if vflash_draft_input_ids is None or vflash_draft_input_ids.numel() == 0:
            return CausalLMOutputWithPast(loss=self._graph_connected_zero(), logits=None)
        if vflash_draft_input_ids.ndim == 1:
            vflash_draft_input_ids = vflash_draft_input_ids.unsqueeze(0)
        if vflash_direct_labels.ndim == 1:
            vflash_direct_labels = vflash_direct_labels.unsqueeze(0)
        if vflash_draft_position_ids.ndim == 1:
            vflash_draft_position_ids = vflash_draft_position_ids.unsqueeze(0)

        noise_embedding = self.target_model.language_model.get_input_embeddings()(
            vflash_draft_input_ids
        )
        if vflash_block_ranges is None:
            raise ValueError("VFlash training requires packed block metadata")
        if vflash_block_ranges.ndim != 2 or vflash_block_ranges.shape[1] != 2:
            raise ValueError("vflash_block_ranges must have shape [num_blocks, 2]")
        num_blocks = vflash_block_ranges.shape[0]
        if num_blocks <= 0:
            raise ValueError("VFlash block metadata must contain at least one block")
        if any(
            value is None
            for value in (
                vflash_block_context_starts,
                vflash_block_anchors,
                vflash_block_sizes,
            )
        ):
            raise ValueError("VFlash block metadata tensors must all be provided")
        if not (
            vflash_block_context_starts.numel()
            == vflash_block_anchors.numel()
            == vflash_block_sizes.numel()
            == num_blocks
        ):
            raise ValueError("VFlash block metadata lengths do not match")

        if target_context.shape[0] != 1:
            raise ValueError(
                "Packed VFlash block metadata currently requires batch size 1"
            )
        ranges = vflash_block_ranges.long()
        context_starts = vflash_block_context_starts.long()
        anchors = vflash_block_anchors.long()
        block_sizes = vflash_block_sizes.long()
        if bool((block_sizes <= 0).any()):
            raise ValueError("VFlash block sizes must all be positive")
        cumulative_ends = block_sizes.cumsum(0)
        expected_starts = torch.cat(
            [cumulative_ends.new_zeros(1), cumulative_ends[:-1]]
        )
        if int(cumulative_ends[-1].item()) != noise_embedding.shape[1]:
            raise ValueError("VFlash block sizes do not cover every draft token")
        if not torch.equal(ranges[:, 0], expected_starts) or not torch.equal(
            ranges[:, 1], cumulative_ends
        ):
            raise ValueError("VFlash block ranges must be contiguous and match block sizes")
        if bool(
            (
                (context_starts < 0)
                | (anchors < context_starts)
                | (anchors > target_context.shape[1])
            ).any()
        ):
            raise ValueError("VFlash target context metadata is out of range")

        query_context_starts = torch.repeat_interleave(
            context_starts, block_sizes
        )
        query_anchors = torch.repeat_interleave(anchors, block_sizes)
        query_draft_starts = torch.repeat_interleave(
            ranges[:, 0], block_sizes
        )
        query_draft_ends = torch.repeat_interleave(
            ranges[:, 1], block_sizes
        )
        combined_position_ids = torch.cat(
            [vflash_clean_position_ids, vflash_draft_position_ids], dim=1
        )
        if noise_embedding.is_cuda:
            sparse_mask = build_vflash_flex_block_mask(
                query_context_starts=query_context_starts,
                query_anchors=query_anchors,
                query_draft_starts=query_draft_starts,
                query_draft_ends=query_draft_ends,
                target_length=target_context.shape[1],
            )
        else:
            # Exact CPU reference for unit tests. Production CUDA never
            # materializes this Q x (L+Q) matrix.
            key_indices = torch.arange(
                target_context.shape[1] + noise_embedding.shape[1],
                device=noise_embedding.device,
            ).unsqueeze(0)
            target_keys = key_indices < target_context.shape[1]
            draft_indices = key_indices - target_context.shape[1]
            allowed = (
                target_keys
                & (key_indices >= query_context_starts.unsqueeze(1))
                & (key_indices < query_anchors.unsqueeze(1))
            ) | (
                (~target_keys)
                & (draft_indices >= query_draft_starts.unsqueeze(1))
                & (draft_indices < query_draft_ends.unsqueeze(1))
            )
            sparse_mask = torch.zeros(
                (1, 1, *allowed.shape),
                dtype=noise_embedding.dtype,
                device=noise_embedding.device,
            )
            sparse_mask.masked_fill_(
                ~allowed.unsqueeze(0).unsqueeze(0),
                torch.finfo(noise_embedding.dtype).min,
            )
        draft_hidden = self.draft_model(
            noise_embedding=noise_embedding,
            target_hidden=target_context,
            position_ids=combined_position_ids,
            attention_mask=sparse_mask,
            use_cache=False,
        )
        labels_flat = vflash_direct_labels.reshape(-1)
        base_weights_flat = vflash_loss_weights.reshape(-1).to(draft_hidden.device)
        physical_atom_config = getattr(self.config, "physical_atom_config", None) or {}
        dflash_config = getattr(self.config, "dflash_config", None) or {}
        weights_flat = apply_oof_loss_scale(
            base_weights_flat,
            labels_flat,
            oof_token_id=physical_atom_config.get("oof_token_id"),
            oof_loss_scale=dflash_config.get("oof_loss_scale", 1.0),
        )
        hidden_flat = draft_hidden.reshape(-1, draft_hidden.shape[-1])
        valid = labels_flat.ne(-100)
        if not bool(valid.any()):
            loss = self._graph_connected_zero()
        elif draft_hidden.is_cuda:
            # Avoid materializing Q x vocabulary logits for production runs.
            loss_fn = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100, reduction="sum"
            )
            numerator = loss_fn(
                self.target_model.language_model.lm_head.weight,
                hidden_flat[valid].contiguous(),
                labels_flat[valid].contiguous(),
                loss_weight=weights_flat[valid].contiguous(),
            )
            # Preserve the position-weight denominator from strict VFlash so
            # an <oof> target contributes an absolute 0.1x numerator/gradient.
            loss = numerator / base_weights_flat[valid].sum().clamp_min(1e-12)
        else:
            logits = self.target_model.language_model.lm_head(draft_hidden)
            loss = weighted_draft_cross_entropy(
                logits,
                vflash_direct_labels,
                weights_flat.reshape_as(vflash_loss_weights),
                ignore_index=-100,
                reduction="sum",
            )
            loss = loss / base_weights_flat[valid].sum().clamp_min(1e-12)
        return CausalLMOutputWithPast(loss=loss, logits=None)


def build_vflash_trainer(stream_trainer_cls):
    """Return the ``stream_trainer_cls`` subclass that trains the separate drafter.

    Its checkpoints contain only the drafter, and its packs are collated by the
    ``VFlashPackedCollator`` passed as ``data_collator``.
    """

    class VFlashStreamPackingTrainer(stream_trainer_cls):
        """Stream trainer whose checkpoints contain only the separate drafter."""

        use_data_collator = True

        def _save(self, output_dir=None, state_dict=None):
            output_dir = output_dir or self.args.output_dir
            draft_model = self.model.draft_model
            # A standalone drafter checkpoint, never a full Eagle checkpoint.
            super()._save(output_dir, state_dict=draft_model.state_dict())
            if self.args.should_save:
                export_vflash_draft_checkpoint(draft_model, output_dir)

        def _save_optimizer_and_scheduler(self, output_dir):
            """Keep ZeRO checkpoints resumable without serializing frozen Eagle."""
            if not self.is_deepspeed_enabled:
                return super()._save_optimizer_and_scheduler(output_dir)
            save_checkpoint = self.model_wrapped.save_checkpoint
            if "exclude_frozen_parameters" not in inspect.signature(
                save_checkpoint
            ).parameters:
                raise RuntimeError(
                    "This DeepSpeed release cannot exclude the frozen Eagle target "
                    "from VFlash checkpoints; upgrade DeepSpeed before training."
                )
            save_checkpoint(output_dir, exclude_frozen_parameters=True)

            # Match Trainer's handling when the scheduler is owned by Trainer
            # rather than wrapped and checkpointed by DeepSpeed itself.
            from accelerate.utils.deepspeed import DeepSpeedSchedulerWrapper
            from transformers.trainer import SCHEDULER_NAME

            if self.args.should_save and not isinstance(
                self.lr_scheduler, DeepSpeedSchedulerWrapper
            ):
                torch.save(
                    self.lr_scheduler.state_dict(),
                    os.path.join(output_dir, SCHEDULER_NAME),
                )

    return VFlashStreamPackingTrainer


__all__ = [
    "EagleEmbodiedVFlashTrainingModel",
    "build_vflash_trainer",
    "distributed_target_weight_fingerprint",
]
