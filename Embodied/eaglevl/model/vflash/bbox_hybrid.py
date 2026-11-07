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
# Modifications © 2026 NVIDIA CORPORATION & AFFILIATES, licensed under
# the Apache License, Version 2.0.
#
# --------------------------------------------------------
# DFlash
# Copyright (c) 2026 Z Lab
# Licensed under The MIT License
# --------------------------------------------------------

"""LocateAnything-style hybrid bbox decoding with a drafter.

This is not exact-target verification. Ordinary text is decoded by the target
in AR mode, starting from a target-checked query prefix. Each ``<box>`` is
drafted in one forward, and LocateAnything's full-vocabulary top-k uncertainty
rule marks unreliable coordinates: best probability below ``confidence``, more
than one numeric candidate, and a token-id spread above ``spread``. The draft
prefix before the first uncertain or missing coordinate is kept, and the target
regenerates the rest of the box under a numeric-only constraint. The closing
token is forced, a box that does not fit the budget is dropped whole, and
neither draft nor target applies xyxy geometry masks.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from . import generation
from .modeling_dflash import extract_context_feature


@dataclass(frozen=True)
class BBoxHybridConfig:
    """Settings of the hybrid bbox decoder; the defaults are the paper protocol.

    A drafted coordinate is uncertain when its best numeric candidate in the
    vocabulary top-``topk`` has probability below ``confidence``, the top-k
    holds more than one numeric candidate, and their token IDs span more than
    ``spread`` (one ID per unit of the 0..1000 grid). A coordinate without any
    numeric candidate in the top-k is repaired as well.

    Attributes:
        numeric_repair_head: Project repair logits through the numeric rows of
            the LM head only.
        fused_prefill: Check the query prefix in the prompt prefill forward.
        fused_close: Feed the last repaired coordinate and ``</box>`` together.
    """

    topk: int = 6
    spread: int = 30
    confidence: float = 0.9
    numeric_repair_head: bool = True
    fused_prefill: bool = True
    fused_close: bool = True


class _NumericSampler:
    def __init__(self, config, device, *, topk=4, confidence=0.9, spread=60):
        if type(topk) is not int or not 1 <= topk <= 1001:
            raise ValueError("Hybrid topk must be an integer in [1,1001]")
        if not 0 <= confidence <= 1 or not 0 <= spread <= 1000:
            raise ValueError("Hybrid confidence/spread outside supported range")
        self.topk, self.confidence, self.spread = topk, confidence, spread
        self.mapping = {int(k): int(v) for k, v in config["numeric_id_to_value"].items()}
        if not self.mapping:
            raise ValueError("BBox hybrid decoding requires numeric tokens")
        self.ids = torch.tensor(sorted(self.mapping), device=device, dtype=torch.long)

    def slot(self, logits, slot, spec):
        if slot == 4:
            return self.ids.new_tensor([int(spec["end_id"])])
        # Numeric membership only. Reversed and degenerate xyxy are permitted.
        return self.ids[logits.index_select(-1, self.ids).argmax(dim=-1)]

    def proposal(self, logits, spec):
        """Return chosen coordinates and per-slot uncertainty/missing masks.

        Probabilities are normalized across the FULL vocabulary, not the
        numeric subset. Spread uses token IDs, as official LocateAnything does.
        Missing numeric slots get an internal placeholder which is never kept.
        """
        probabilities = torch.softmax(logits[0, :4].float(), dim=-1)
        top_probs, top_ids = torch.topk(probabilities, k=min(self.topk, probabilities.shape[-1]), dim=-1)
        values = top_probs.tolist()
        ids = top_ids.tolist()
        coords, ambiguous, missing = [], 0, 0
        for slot, (row_ids, row_probs) in enumerate(zip(ids, values)):
            candidates = [(token, p) for token, p in zip(row_ids, row_probs) if token in self.mapping]
            if not candidates:
                missing |= 1 << slot
                coords.append(int(self.ids[0]))
                continue
            # topk is probability-sorted; retain its tie order, not a new
            # coordinate-only argmax or expected-value/average prediction.
            token, confidence = candidates[0]
            coords.append(token)
            spread = max(t for t, _ in candidates) - min(t for t, _ in candidates)
            if confidence < self.confidence and len(candidates) > 1 and spread > self.spread:
                ambiguous |= 1 << slot
        tail = self.ids.new_tensor([*coords, int(spec["end_id"])]).unsqueeze(0)
        return tail, ambiguous, missing


class NumericRepairHead(nn.Module):
    """Numeric-only projection for constrained coordinate repair.

    The returned logits retain the full vocabulary shape. Non-numeric entries
    are -inf and must never be used for unconstrained text or boundary decoding.
    The smaller GEMM may differ from the full LM head in floating point.
    """

    def __init__(self, head, numeric_ids):
        super().__init__()
        if not isinstance(head, nn.Linear):
            raise TypeError("Numeric repair requires an ordinary linear lm_head")
        ids = numeric_ids.detach().clone()
        if ids.ndim != 1 or not ids.numel() or ids.unique().numel() != ids.numel():
            raise ValueError("Expected nonempty unique numeric IDs")
        if ids.min() < 0 or ids.max() >= head.out_features:
            raise ValueError("Numeric IDs outside vocabulary")
        self.vocab_size = head.out_features
        self.register_buffer("ids", ids)
        self.register_buffer("numeric_weight", head.weight.detach().index_select(0, ids))
        self.register_buffer("numeric_bias", None if head.bias is None else head.bias.detach().index_select(0, ids))

    def forward(self, hidden):
        logits = F.linear(hidden, self.numeric_weight, self.numeric_bias)
        full = logits.new_full((*logits.shape[:-1], self.vocab_size), -torch.inf)
        return full.index_copy_(-1, self.ids, logits)


@contextmanager
def use_numeric_head(model, head):
    original = model.lm_head
    model.lm_head = head
    try:
        yield
    finally:
        model.lm_head = original


def _bbox_frame_valid(tokens, spec, mapping):
    return (len(tokens) == 6 and int(tokens[0]) == int(spec["start_id"])
            and int(tokens[-1]) == int(spec["end_id"])
            and all(int(token) in mapping for token in tokens[1:5]))


def _geometry(tokens, mapping):
    x1, y1, x2, y2 = [mapping[int(token)] for token in tokens[1:5]]
    return int(x2 < x1 or y2 < y1), int(x2 == x1 or y2 == y1)


def _validate_generated(tokens, specs, mapping, config):
    """Validate membership/framing, deliberately NOT bbox corner ordering."""
    reserved_nonstarts = {int(spec["end_id"]) for spec in specs.values()}
    for key in ("mask_token_id", "null_token_id", "oof_token_id"):
        if config.get(key) is not None:
            reserved_nonstarts.add(int(config[key]))
    cursor = 0
    while cursor < len(tokens):
        token = int(tokens[cursor])
        spec = specs.get(token)
        if spec is None:
            if token in reserved_nonstarts:
                raise RuntimeError("Hybrid output contains an orphan physical token")
            cursor += 1
            continue
        size = int(spec["block_size"])
        block = tokens[cursor:cursor + size]
        if spec.get("name") == "bbox_2d":
            valid = _bbox_frame_valid(block, spec, mapping)
            reason = "incomplete bbox or nonnumeric bbox payload"
        else:
            valid, reason = generation.validate_serialized_atom_tail(token, block[1:], config)
        if not valid:
            raise RuntimeError(f"Hybrid output contains an invalid physical frame: {reason}")
        cursor += size


@torch.inference_mode()
def bbox_hybrid_generate(
    *, draft_model, target_language_model, input_ids, inputs_embeds,
    attention_mask, max_new_tokens, stop_token_ids, physical_atom_config,
    config, return_stats, initial_prefix_token_ids=None,
):
    """Decode one unpadded bbox query; see the module docstring.

    ``initial_prefix_token_ids`` is the ordinary-text answer prefix derived from
    the query. The target checks it causally and keeps only the matching part.
    """
    if not isinstance(config, BBoxHybridConfig):
        raise TypeError(f"config must be a BBoxHybridConfig, got {type(config).__name__}")
    fused_prefill = bool(config.fused_prefill)
    fused_close = bool(config.fused_close)
    device = inputs_embeds.device
    prompt_length = input_ids.shape[1]
    specs = generation._atom_specs_by_start_id(physical_atom_config)
    sampler = _NumericSampler(
        physical_atom_config, device,
        topk=config.topk,
        confidence=float(config.confidence),
        spread=float(config.spread),
    )
    if initial_prefix_token_ids is not None:
        if (not isinstance(initial_prefix_token_ids, (list, tuple))
                or not 1 <= len(initial_prefix_token_ids) <= 32
                or any(type(t) is not int or t < 0 for t in initial_prefix_token_ids)):
            raise ValueError("Initial prefix must contain 1..32 nonnegative integer IDs")
        forbidden = set(specs) | {int(s["end_id"]) for s in specs.values()}
        forbidden |= set(map(int, stop_token_ids or []))
        forbidden |= {int(physical_atom_config[k]) for k in ("mask_token_id", "null_token_id", "oof_token_id")
                      if physical_atom_config.get(k) is not None}
        if any(t in forbidden or t in sampler.mapping for t in initial_prefix_token_ids):
            raise ValueError("Initial prefix may only propose ordinary text, not physical or stop tokens")
    maximum_atom_size = max([int(s["block_size"]) for s in specs.values()] or [16])
    capacity = prompt_length + max_new_tokens + maximum_atom_size + 1
    mask_id = getattr(draft_model, "mask_token_id", None)
    mask_id = int(physical_atom_config["mask_token_id"] if mask_id is None else mask_id)
    output_ids = torch.full((1, capacity), mask_id, device=device, dtype=torch.long)
    output_ids[:, :prompt_length] = input_ids
    position_ids = torch.arange(capacity, device=device)[None]
    target_cache, draft_cache = generation.DynamicCache(), generation.DynamicCache()
    stop_ids = {int(value) for value in (stop_token_ids or [])}
    ordinary_banned = generation._ordinary_banned_ids(mask_id, physical_atom_config)
    objects_end = physical_atom_config.get("objects_end_token_id")
    max_length = prompt_length + max_new_tokens
    counters = {name: 0 for name in (
        "fused_prefill_prefix_calls", "numeric_repair_head_calls", "fused_close_calls",
        "initial_prefix_calls", "initial_prefix_input_tokens", "initial_prefix_matched_tokens",
        "prefill_calls", "target_forward_calls", "draft_forward_calls", "text_ar_steps",
        "atom_proposals", "atom_accepted", "bbox_proposals", "bbox_accepted",
        "target_block_forward_calls", "fallback_target_steps", "atom_fallbacks",
        "skipped_target_block_forwards", "draft_cache_reused_fallbacks",
        "pending_context_token_total", "bbox_budget_truncations",
    )}
    native = {name: 0 for name in (
        "guard_pass_boxes", "guard_repair_boxes", "missing_numeric_boxes",
        "retained_coords", "repaired_coords", "changed_coords", "comparable_repaired_coords",
        "suffix_target_calls", "suffix_target_input_tokens", "prefix_chunk_input_tokens",
        "nonbbox_proposals", "nonbbox_fallback_calls", "format_failures",
        "draft_geometry_observed", "draft_reversed", "draft_degenerate",
        "output_reversed", "output_degenerate", "output_boxes",
        "draft_consecutive_repeats", "output_consecutive_repeats", "repeat_terminations",
        "budget_initial", "budget_text", "budget_atom_entry", "budget_bonus",
        "stop_prefill", "stop_text", "stop_bonus", "objects_end_prefill",
        "objects_end_text", "objects_end_bonus",
    )}
    for slot in range(4):
        for kind in ("ambiguous_slot", "missing_slot", "first_uncertain", "retained_prefix"):
            native[f"{kind}_{slot}"] = 0
    for mask in range(16):
        native[f"uncertain_mask_{mask}"] = 0
    trace, acceptance_lengths, round_kinds, reasons, details = [], [], [], [], []
    pending, pending_length = [], 0
    previous_draft_box, previous_output_box = None, None

    def terminal(token, origin):
        if int(token) in stop_ids:
            native[f"stop_{origin}"] += 1
        if objects_end is not None and int(token) == int(objects_end):
            native[f"objects_end_{origin}"] += 1

    numeric_head = None
    if config.numeric_repair_head:
        numeric_head = NumericRepairHead(target_language_model.lm_head, sampler.ids)

    def target_forward(numeric_only=False, **kwargs):
        counters["target_forward_calls"] += 1
        if numeric_only and numeric_head is not None:
            counters["numeric_repair_head_calls"] += 1
            with use_numeric_head(target_language_model, numeric_head):
                return target_language_model(**kwargs)
        return target_language_model(**kwargs)

    def append_context(hidden):
        nonlocal pending_length
        pending.append(hidden)
        pending_length += hidden.shape[1]

    def record_box(block, draft=False):
        nonlocal previous_draft_box, previous_output_box
        tokens = tuple(int(v) for v in block[0].tolist())
        reversed_, degenerate = _geometry(tokens, sampler.mapping)
        prefix = "draft" if draft else "output"
        native[f"{prefix}_reversed"] += reversed_
        native[f"{prefix}_degenerate"] += degenerate
        if draft:
            native["draft_geometry_observed"] += 1
            native["draft_consecutive_repeats"] += int(previous_draft_box == tokens)
            previous_draft_box = tokens
        else:
            native["output_boxes"] += 1
            native["output_consecutive_repeats"] += int(previous_output_box == tokens)
            previous_output_box = tokens

    def result_stats(result, first_seconds, decode_seconds):
        stats = dict(counters)
        stats.update({f"bbox_hybrid_{name}": value for name, value in native.items()})
        stats.update(
            output_ids=result, num_input_tokens=prompt_length, num_output_tokens=result.shape[1],
            time_to_first_token=first_seconds, time_per_output_token=decode_seconds / max(result.shape[1], 1),
            acceptance_lengths=acceptance_lengths, round_kinds=round_kinds,
            atom_fallback_reasons=reasons, atom_fallback_details=details, drive_oof_pair_repairs=0,
            verify_atoms=False, text_block_size=16,
            fast_path=False, bbox_hybrid_enabled=True, bbox_hybrid_policy="prefix",
            bbox_hybrid_topk=sampler.topk, bbox_hybrid_confidence=sampler.confidence,
            bbox_hybrid_spread=sampler.spread,
            bbox_hybrid_numeric_repair_head_enabled=numeric_head is not None,
            bbox_hybrid_fused_prefill_enabled=fused_prefill,
            bbox_hybrid_trace_version=1,
            bbox_hybrid_fused_close_enabled=fused_close,
            bbox_hybrid_initial_prefix_enabled=initial_prefix_token_ids is not None,
            # One [ambiguous_mask, missing_mask, retained, repaired, changed_mask]
            # row per box, serialized so it survives scalar-only stat records.
            # It describes the draft guard, not target acceptance.
            bbox_hybrid_trace_json=json.dumps(trace, separators=(",", ":")),
            bbox_budget_policy="truncate_complete",
        )
        return SimpleNamespace(**stats)

    if max_new_tokens == 0:
        result = input_ids[:, :0]
        return result_stats(result, 0.0, 0.0) if return_stats else result
    if return_stats and device.type == "cuda":
        torch.cuda.synchronize(device)
    prefill_started = time.perf_counter()
    cached_prefix = None
    proposed = initial_prefix_token_ids[:max_new_tokens - 1] if (fused_prefill and initial_prefix_token_ids and max_new_tokens > 1) else []
    if proposed:
        embedding = target_language_model.get_input_embeddings()
        if max(proposed) >= embedding.weight.shape[0]:
            raise ValueError("Initial prefix ID outside target vocabulary")
        proposal_ids = output_ids.new_tensor([proposed])
        combined_embeds = torch.cat([inputs_embeds, embedding(proposal_ids)], dim=1)
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("Fused prefill requires a 2D attention mask")
        combined_mask = None if attention_mask is None else torch.cat([
            attention_mask, attention_mask.new_ones((attention_mask.shape[0], len(proposed)))], dim=1)
        combined = target_forward(
            inputs_embeds=combined_embeds, attention_mask=combined_mask,
            position_ids=position_ids[:, :prompt_length + len(proposed)], past_key_values=target_cache,
            use_cache=True, logits_to_keep=len(proposed) + 1, output_hidden_states=True, return_dict=True)
        prefill = SimpleNamespace(logits=combined.logits[:, :1],
            hidden_states=tuple(h[:, :prompt_length] for h in combined.hidden_states))
        cached_prefix = SimpleNamespace(logits=combined.logits[:, 1:],
            hidden_states=tuple(h[:, prompt_length:] for h in combined.hidden_states))
        counters["fused_prefill_prefix_calls"] += 1
    else:
        prefill = target_forward(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids[:, :prompt_length], past_key_values=target_cache,
            use_cache=True, logits_to_keep=1, output_hidden_states=True, return_dict=True,
        )
    counters["prefill_calls"] = 1
    if initial_prefix_token_ids is not None and max(initial_prefix_token_ids) >= prefill.logits.shape[-1]:
        raise ValueError("Initial prefix ID outside target vocabulary")
    first_token = generation._sample(prefill.logits, 0.0, ordinary_banned)
    first_id = int(first_token.item())
    if cached_prefix is not None and first_id != proposed[0]:
        target_cache.crop(prompt_length)
        cached_prefix = None
    first_spec = specs.get(first_id)
    output_ids[:, prompt_length] = first_token.reshape(1)
    start = prompt_length
    if first_spec is not None and max_new_tokens < int(first_spec["block_size"]):
        if first_spec.get("name") != "bbox_2d":
            raise ValueError("max_new_tokens truncates a non-bbox physical atom")
        counters["bbox_budget_truncations"] += 1
        native["budget_initial"] += 1
        start -= 1
    else:
        terminal(first_id, "prefill")
    append_context(extract_context_feature(prefill.hidden_states, draft_model.target_layer_ids))
    if return_stats and device.type == "cuda":
        torch.cuda.synchronize(device)
    first_seconds = time.perf_counter() - prefill_started
    decode_started = time.perf_counter()

    # Query-template proposal is checked causally by TARGET, never forced.
    # On mismatch crop target KV/hidden state to exactly the accepted prefix;
    # the correction token remains the uncached next anchor for the main loop.
    if (initial_prefix_token_ids is not None and start == prompt_length
            and first_id == initial_prefix_token_ids[0] and max_new_tokens > 1):
        proposal = initial_prefix_token_ids[:max_new_tokens - 1]
        ids = output_ids.new_tensor([proposal])
        if cached_prefix is not None:
            checked = cached_prefix
        else:
            checked = target_forward(
                input_ids=ids, position_ids=position_ids[:, start:start + len(proposal)],
                past_key_values=target_cache, use_cache=True, output_hidden_states=True, return_dict=True,
            )
            counters["initial_prefix_calls"] += 1
        predictions = generation._sample(checked.logits, 0.0, ordinary_banned)[0].tolist()
        keep = 1
        while keep < len(proposal) and predictions[keep - 1] == proposal[keep]:
            keep += 1
        output_ids[:, start:start + keep] = ids[:, :keep]
        correction = int(predictions[keep - 1])
        correction_position = start + keep
        target_cache.crop(correction_position)
        hidden = extract_context_feature(checked.hidden_states, draft_model.target_layer_ids)
        append_context(hidden[:, :keep])
        counters["initial_prefix_input_tokens"] += len(proposal)
        counters["initial_prefix_matched_tokens"] += keep
        correction_spec = specs.get(correction)
        if correction_spec is not None and max_length - correction_position < int(correction_spec["block_size"]):
            if correction_spec.get("name") != "bbox_2d":
                raise ValueError("max_new_tokens truncates a non-bbox physical atom")
            counters["bbox_budget_truncations"] += 1
            native["budget_text"] += 1
            start = correction_position - 1
            # Prefix tokens are all ordinary. No partial physical anchor emitted.
            prefix_budget_exhausted = True
        else:
            output_ids[:, correction_position] = correction
            start = correction_position
            terminal(correction, "text")
            prefix_budget_exhausted = False
    else:
        prefix_budget_exhausted = False

    while not prefix_budget_exhausted and start >= prompt_length and start < max_length - 1 and not generation._contains_stop(output_ids[:, :start + 1], prompt_length, stop_ids):
        if target_cache.get_seq_length() != start:
            raise AssertionError("Hybrid target KV must end before current anchor")
        if draft_cache.get_seq_length() + pending_length != start:
            raise AssertionError("Hybrid clean draft KV plus pending context must equal anchor position")
        anchor_id = int(output_ids[0, start])
        block_size, spec = generation.resolve_vflash_block_size(anchor_id, physical_atom_config, 16)
        if spec is None:
            # A closing anchor belongs to the preceding box; other text breaks
            # raw adjacent-box repetition, unlike mere bbox-stream equality.
            if anchor_id not in {int(s["end_id"]) for s in specs.values()}:
                previous_draft_box, previous_output_box = None, None
            result = target_forward(
                input_ids=output_ids[:, start:start + 1], position_ids=position_ids[:, start:start + 1],
                past_key_values=target_cache, use_cache=True, output_hidden_states=True, return_dict=True,
            )
            counters["text_ar_steps"] += 1
            round_kinds.append("text_ar")
            append_context(extract_context_feature(result.hidden_states, draft_model.target_layer_ids))
            next_token = generation._sample(result.logits[:, -1:], 0.0, ordinary_banned)
            next_id, next_start = int(next_token.item()), start + 1
            next_spec = specs.get(next_id)
            if next_spec is not None and max_length - next_start < int(next_spec["block_size"]):
                if next_spec.get("name") != "bbox_2d":
                    raise ValueError("max_new_tokens truncates a non-bbox physical atom")
                counters["bbox_budget_truncations"] += 1
                native["budget_text"] += 1
                acceptance_lengths.append(0)
                break
            terminal(next_id, "text")
            output_ids[:, next_start] = next_token.reshape(1)
            start = next_start
            acceptance_lengths.append(1)
            continue
        if max_length - start < block_size:
            if spec.get("name") != "bbox_2d":
                raise ValueError("max_new_tokens truncates a non-bbox physical atom")
            counters["bbox_budget_truncations"] += 1
            native["budget_atom_entry"] += 1
            start -= 1
            break
        is_bbox = spec.get("name") == "bbox_2d"
        if not is_bbox:
            previous_draft_box, previous_output_box = None, None
        if is_bbox and (block_size != 6 or int(spec["payload_length"]) != 4):
            raise ValueError("BBox hybrid decoding requires B6 bbox with four coordinates")
        counters["atom_proposals"] += 1
        counters["bbox_proposals"] += int(is_bbox)
        native["nonbbox_proposals"] += int(not is_bbox)
        round_kinds.append("bbox" if is_bbox else "atom")
        # Do not recycle any discarded proposal suffix as noisy input.
        block_ids = torch.full((1, block_size), mask_id, device=device, dtype=torch.long)
        block_ids[:, 0] = anchor_id
        context = pending[0] if len(pending) == 1 else torch.cat(pending, dim=1)
        counters["pending_context_token_total"] += context.shape[1]
        draft_hidden = draft_model(
            target_hidden=context, noise_embedding=target_language_model.get_input_embeddings()(block_ids),
            position_ids=position_ids[:, draft_cache.get_seq_length():start + block_size],
            past_key_values=draft_cache, use_cache=True, is_causal=False,
        )
        counters["draft_forward_calls"] += 1
        pending, pending_length = [], 0
        draft_cache.crop(start)
        if draft_cache.get_seq_length() != start:
            raise AssertionError("Hybrid draft failed to retain clean KV")
        ambiguous = missing = retained = 0
        if is_bbox:
            logits = target_language_model.lm_head(draft_hidden[:, 1:5])
            block_ids[:, 1:], ambiguous, missing = sampler.proposal(logits, spec)
            uncertain = ambiguous | missing
            for slot in range(4):
                native[f"ambiguous_slot_{slot}"] += int(bool(ambiguous & (1 << slot)))
                native[f"missing_slot_{slot}"] += int(bool(missing & (1 << slot)))
            native[f"uncertain_mask_{uncertain}"] += 1
            native["missing_numeric_boxes"] += int(bool(missing))
            if not missing:
                record_box(block_ids, draft=True)
            else:
                previous_draft_box = None
            if uncertain:
                # Keep the draft coordinates before the first uncertain slot.
                retained = next(slot for slot in range(4) if uncertain & (1 << slot))
                native[f"first_uncertain_{retained}"] += 1
                native[f"retained_prefix_{retained}"] += 1
            valid = not uncertain
            reason = f"missing numeric top{sampler.topk}" if missing else "numeric spatial ambiguity"
        else:
            logits = target_language_model.lm_head(draft_hidden[:, 1:])
            block_ids[:, 1:] = generation._sample_atom_tail(logits, spec, physical_atom_config, 0.0)
            valid, reason = generation.validate_serialized_atom_tail(anchor_id, block_ids[0, 1:].tolist(), physical_atom_config)
        if not valid:
            counters["atom_fallbacks"] += 1
            counters["skipped_target_block_forwards"] += 1
            reasons.append(reason)
            details.append({"reason": reason, "retained_coords": retained, "ambiguous_mask": ambiguous, "missing_mask": missing})
            if is_bbox:
                native["guard_repair_boxes"] += 1
                native["retained_coords"] += retained
                native["repaired_coords"] += 4 - retained
                native["prefix_chunk_input_tokens"] += retained + 1
            else:
                native["format_failures"] += 1
            # Exactly the retained prefix is supplied to target; discarded
            # coordinates are NEVER placed in its KV or pending hidden states.
            output_ids[:, start:start + retained + 1] = block_ids[:, :retained + 1]
            current_ids = block_ids[:, :retained + 1]
            prior_tail = [int(v) for v in block_ids[0, 1:retained + 1].tolist()]
            changed_mask = 0
            fuse = is_bbox and fused_close
            for slot in range(retained, block_size - 1 - int(fuse)):
                left = start if slot == retained else start + slot
                ar_output = target_forward(
                    numeric_only=is_bbox and slot < 4,
                    input_ids=current_ids, position_ids=position_ids[:, left:start + slot + 1],
                    past_key_values=target_cache, use_cache=True, logits_to_keep=1,
                    output_hidden_states=True, return_dict=True,
                )
                counters["fallback_target_steps"] += 1
                if is_bbox:
                    native["suffix_target_calls"] += 1
                    native["suffix_target_input_tokens"] += current_ids.shape[1]
                else:
                    native["nonbbox_fallback_calls"] += 1
                append_context(extract_context_feature(ar_output.hidden_states, draft_model.target_layer_ids))
                next_id = (sampler.slot(ar_output.logits[:, -1], slot, spec) if is_bbox else
                           generation._sample_atom_slot(ar_output.logits[:, -1], spec, slot, prior_tail, physical_atom_config, 0.0))
                value = int(next_id.item())
                if is_bbox and slot < 4 and not missing & (1 << slot):
                    native["comparable_repaired_coords"] += 1
                    changed_mask |= int(value != int(block_ids[0, slot + 1])) << slot
                prior_tail.append(value)
                output_ids[:, start + slot + 1] = next_id.reshape(1)
                current_ids = next_id.reshape(1, 1)
            if fuse:
                # Closing token is already forced by the bbox grammar. Feed the
                # final coordinate + close together, and use the final logits
                # for the boundary instead of two separate target calls.
                output_ids[:, start + 5] = int(spec["end_id"])
                boundary = target_forward(
                    input_ids=output_ids[:, start + 4:start + 6],
                    position_ids=position_ids[:, start + 4:start + 6],
                    past_key_values=target_cache, use_cache=True, logits_to_keep=1,
                    output_hidden_states=True, return_dict=True,
                )
                counters["fused_close_calls"] += 1
                append_context(extract_context_feature(boundary.hidden_states, draft_model.target_layer_ids))
            if is_bbox:
                completed = output_ids[:, start:start + block_size]
                if not _bbox_frame_valid(completed[0].tolist(), spec, sampler.mapping):
                    raise RuntimeError("Hybrid numeric suffix produced an invalid bbox frame")
                native["changed_coords"] += bin(changed_mask).count("1")
                record_box(completed)
                trace.append([ambiguous, missing, retained, 1, changed_mask])
            else:
                valid, reason = generation.validate_serialized_atom_tail(anchor_id, prior_tail, physical_atom_config)
                if not valid:
                    raise RuntimeError(f"Constrained target produced invalid non-bbox atom: {reason}")
            counters["draft_cache_reused_fallbacks"] += 1
            if fuse:
                bonus = generation._sample(boundary.logits[:, -1:], 0.0, ordinary_banned)
                bonus_id, bonus_position = int(bonus.item()), start + block_size
                bonus_spec = specs.get(bonus_id)
                if bonus_position < max_length and bonus_spec is not None and max_length - bonus_position < int(bonus_spec["block_size"]):
                    if bonus_spec.get("name") != "bbox_2d":
                        raise ValueError("max_new_tokens truncates a non-bbox physical atom")
                    counters["bbox_budget_truncations"] += 1
                    native["budget_bonus"] += 1
                    start += block_size - 1
                    acceptance_lengths.append(1)
                    break
                if bonus_position < max_length:
                    terminal(bonus_id, "bonus")
                    output_ids[:, bonus_position] = bonus.reshape(1)
                    start = bonus_position
                else:
                    start += block_size - 1
                acceptance_lengths.append(1)
                continue
            start += block_size - 1
            acceptance_lengths.append(1)
            continue
        # This target call refreshes KV/context and predicts the boundary. No
        # internal exact-token agreement is computed or used as a guard.
        accepted = target_forward(
            input_ids=block_ids, position_ids=position_ids[:, start:start + block_size],
            past_key_values=target_cache, use_cache=True, logits_to_keep=1,
            output_hidden_states=True, return_dict=True,
        )
        counters["target_block_forward_calls"] += 1
        counters["atom_accepted"] += 1
        counters["bbox_accepted"] += int(is_bbox)
        output_ids[:, start:start + block_size] = block_ids
        append_context(extract_context_feature(accepted.hidden_states, draft_model.target_layer_ids))
        if is_bbox:
            native["guard_pass_boxes"] += 1
            native["retained_coords"] += 4
            record_box(block_ids)
            trace.append([ambiguous, missing, 4, 0, 0])
        bonus = generation._sample(accepted.logits[:, -1:], 0.0, ordinary_banned)
        bonus_id, bonus_position = int(bonus.item()), start + block_size
        bonus_spec = specs.get(bonus_id)
        if bonus_position < max_length and bonus_spec is not None and max_length - bonus_position < int(bonus_spec["block_size"]):
            if bonus_spec.get("name") == "bbox_2d":
                counters["bbox_budget_truncations"] += 1
                native["budget_bonus"] += 1
            start += block_size - 1
            acceptance_lengths.append(block_size)
            break
        if bonus_position < max_length:
            terminal(bonus_id, "bonus")
        output_ids[:, bonus_position] = bonus.reshape(1)
        start += block_size
        target_cache.crop(start)
        acceptance_lengths.append(block_size)

    result = output_ids[:, prompt_length:min(start + 1, max_length)]
    tokens = result[0].tolist()
    first_stop = next((i for i, token in enumerate(tokens) if token in stop_ids), None)
    if first_stop is not None:
        result, tokens = result[:, :first_stop + 1], tokens[:first_stop + 1]
    _validate_generated(tokens, specs, sampler.mapping, physical_atom_config)
    if counters["target_forward_calls"] != counters["prefill_calls"] + counters["fused_close_calls"] + counters["initial_prefix_calls"] + counters["text_ar_steps"] + counters["target_block_forward_calls"] + counters["fallback_target_steps"]:
        raise AssertionError("Hybrid target forward accounting mismatch")
    if counters["draft_forward_calls"] != counters["atom_proposals"]:
        raise AssertionError("Hybrid draft forward accounting mismatch")
    if native["output_boxes"] != counters["bbox_proposals"] or len(trace) != counters["bbox_proposals"]:
        raise AssertionError("Hybrid bbox publication accounting mismatch")
    if native["retained_coords"] + native["repaired_coords"] != 4 * native["output_boxes"]:
        raise AssertionError("Hybrid coordinate accounting mismatch")
    if native["suffix_target_calls"] != sum((5 - int(fused_close) - j) * native[f"retained_prefix_{j}"] for j in range(4)):
        raise AssertionError("Hybrid suffix call accounting mismatch")
    if native["suffix_target_input_tokens"] != (5 - int(fused_close)) * native["guard_repair_boxes"]:
        raise AssertionError("Hybrid repaired context accounting mismatch")
    if len(acceptance_lengths) != counters["text_ar_steps"] + counters["atom_proposals"]:
        raise AssertionError("Hybrid round accounting mismatch")
    if not return_stats:
        return result
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result_stats(result, first_seconds, time.perf_counter() - decode_started)
