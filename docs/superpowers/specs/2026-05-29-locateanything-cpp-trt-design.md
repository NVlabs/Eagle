# LocateAnything-3B — C++ + TensorRT / TensorRT-LLM Deployment Design

**Status:** Draft for review · **Date:** 2026-05-29

## 1. Overview

LocateAnything-3B is an NVIDIA vision-language grounding model: a **MoonViT**
native-resolution ViT encodes image patches into visual tokens; an **mlp1**
connector projects them into the LLM space; they are spliced into a **Qwen2.5**
decoder's embedding stream at `<IMG_CONTEXT>` placeholder positions; the decoder
emits text plus structured localization tokens (`<box>`, coordinate tokens
`<0>..<1000>`, `<ref>`). Its headline feature is **Parallel Box Decoding (PBD)**
— a Multi-Token-Prediction (MTP) scheme that emits a whole
`<box><x1><y1><x2><y2></box>` frame (6 token slots) in **one** transformer
forward instead of six autoregressive steps, with a hybrid AR fallback when a
box is malformed.

This document specifies a C++ application that runs LocateAnything entirely on
TensorRT (vision) + TensorRT-LLM (decoder), with all orchestration,
tokenization, preprocessing, the PBD/AR decode state machine, and box parsing
reimplemented natively.

- The **scaffolding reference** for export/build/glue is the existing Eagle 2.5
  deployment: `/lake/workspaces/nvidia_ws/Eagle/Eagle2_5/deployment/`.
- The **canonical port surface** is
  `/lake/workspaces/nvidia_ws/Eagle/Embodied/eaglevl/utils/locany/` — the
  `AutoModel trust_remote_code` package loaded by `locateanything_worker.py` —
  **not** the training-time copy under `eaglevl/model/locany/`. Porting the wrong
  copy loses the entire MTP/hybrid logic.

**The hard truth up front:** the Qwen2.5 transformer core is stock and TRT-LLM
supports it out of the box, but **PBD has no native TRT-LLM analog**.
Medusa/EAGLE speculative decoding does *not* match PBD semantics (PBD commits a
whole structured box atomically and reverts malformed suffixes to AR; there is
no per-token draft/verify acceptance). The schedule-dominating work is:
(a) driving the decoder from a custom C++ loop instead of `ModelRunner.generate`,
(b) feeding a custom dense 3-region attention mask as a runtime engine input,
(c) per-step KV-cache append-then-rollback of the 6 speculative columns, and
(d) bit-exact replication of the box-format FSM and its probability thresholds.

## 2. Scope (locked)

Decisions confirmed with the user on 2026-05-29:

- **Hardware:** Hopper / Blackwell only. Keep the Eagle 2.5 toolchain
  (`nvcr.io/nvidia/tensorrt:25.06-py3`, TRT 10.11.0.33, TRT-LLM 0.20 @ commit
  `7c828d7`). The **magi** attention backend is dropped; we use a custom
  SDPA-equivalent dense additive mask (PBD requires a bespoke mask regardless).
- **Request shape:** single-image, batch=1 only (matches the worker's
  `predict()` and the model's `batch_size==1` assertion).
- **PBD in v1:** required. NTP-only (slow) mode is retained as an *internal*
  correctness checkpoint (Phase 2), but the shippable deliverable spans through
  hybrid PBD (Phase 4).
- **Decode:** greedy (deterministic) with a tight parity bar (near-exact
  box-IoU / argmax match vs the PyTorch bf16 reference run in greedy mode).
  Repetition penalty is retained to match decode behavior; production sampling
  (`do_sample=True`, T=0.7) is **not** implemented in v1.

**Out of scope (v1):** multi-image per request; video / fastseek frame sampling;
training / Liger fused CE; the magi attention backend; quantization
(W4A16 / FP8) — a later optimization gated on a separate accuracy budget once
FP16 parity and PBD correctness are proven.

Tasks supported: `detect`, `ground_single`, `ground_multi`, `ground_text`,
`detect_text`, `ground_gui` (box/point), `point`.

## 3. System Architecture and Data Flow

Three engines plus a C++ host driver. All control flow, sampling, and box
parsing live on the host; only per-step neural forwards are engines.

```
                          C++ Host Driver (batch=1)
  image ──────────► [Preprocess: CUDA/NPP]
        │             resize(BICUBIC, in_token_limit guard) → ceil-to-28
        │             normalize x*2-1 → patchify → pixel_values[L,3,14,14], grid_hws[N,2]
        │
  prompt text ────► [Tokenizer (HF tokenizers, tokenizer.json)]
        │             chat template (<|im_start|>/<|im_end|>, system inject)
        │             <image-k> → <img> + <IMG_CONTEXT>*((h*w)//4) + </img>
        │             → input_ids[1,S]
        ▼
  ┌─────────────────────┐
  │ ENGINE 1: MoonViT    │  in: pixel_values[L,3,14,14]
  │  + mlp1 (folded)     │  out: vit_embeds[L/4, H_llm]  (LLM space)
  └─────────────────────┘
        │   scatter vit_embeds into embed(input_ids) where id==image_token_index(151667)
        ▼   → inputs_embeds[1,S,H_llm]
  ┌─────────────────────┐
  │ ENGINE 2: Qwen2.5    │  in: inputs_embeds, 4D additive mask, position_ids, KV-cache I/O
  │  decoder + lm_head   │  out: logits (last-N positions)
  └─────────────────────┘
        ▼
  [Host PBD/AR decode loop]
     fast=MTP only | slow=AR only | hybrid=MTP w/ AR fallback (v1 default)
     per step: build MTP-window or AR-causal mask, position_ids (-1 on trailing n_future),
               forward, sample 6 (MTP) or 1 (AR) logit positions,
               decode_bbox_avg / is_valid_box_frame / handle_pattern,
               KV-cache rollback to committed length, mode switching
        ▼
  [Output parse: scan token ids → <box>/<ref>; coord=(id-coord_start)/1000 → pixels]
```

**Engine-count decision (de-risk in Phase 2):** ENGINE 2 may need to be a
standalone TRT engine (decoder layers with the mask as an explicit input) rather
than the TRT-LLM executor, depending on whether TRT-LLM 0.20 accepts a runtime
dense 4D additive mask (its built-ins are causal / sliding-window). This must be
resolved before committing the LLM architecture — see Risk R5.

## 4. Subsystem 1 — Vision (MoonViT + mlp1)

### 4.1 What it computes
Conv2d(3→1152, k=14, s=14) patch embed (mathematically `Gemm(588→1152)` on
flattened 14×14 patches) → learnable 2D-interpolated absolute pos-emb (bicubic
from a 64×64 grid) → 27× transformer layers (LayerNorm; `wqkv` Linear+bias →
q,k,v; 2D-RoPE; attention; `wo` Linear+bias; residual; LayerNorm; GELU-tanh MLP
1152→4304→1152) → final LayerNorm → 2×2 patch-merger (→ 4608-d merged tokens) →
**mlp1** = `LayerNorm(4608) → Linear(4608→H_llm) → GELU → Linear(H_llm→H_llm)`
(**no pixel-shuffle**, unlike Eagle 2.5). Output `[L/4, H_llm]`.

### 4.2 Export strategy — pin geometry
The encoder runs on one packed, fully data-dependent sequence with per-image
Python loops (pos-emb interpolate, RoPE gather, patch_merger) and
`flash_attn_varlen_func` — none exportable. **Pin N=1 at a small set of canonical
resolution buckets** (host preprocessing already quantizes H,W to multiples of 28
under `in_token_limit`). With N=1:

- `cu_seqlens` degenerates to `[0, L]` → plain full self-attention, **no mask**.
- Absolute pos-emb becomes a **baked constant** per bucket (the bicubic
  interpolate resolved at trace time) — avoids dynamic-output-size
  `F.interpolate(bicubic)`.
- 2D-RoPE `freqs_cis (L,36)` becomes a **baked cos/sin constant** per bucket;
  `apply_rope` is rewritten from `view_as_complex`/`polar` to **real interleaved
  cos/sin arithmetic** (the complex ops are not TRT-exportable).
- patch_merger becomes a single static reshape/permute.
- Conv2d exports as a Gemm.

Wrap MoonViT + mlp1 in a `FeatureExtractorWrapper(nn.Module)` (Eagle 2.5 pattern,
rewritten for the no-pixel-shuffle mlp1) so the ONNX emits LLM-space `vit_embeds`
directly. Export with `_attn_implementation='sdpa'`/eager, `.float()`,
`opset_version=17`, `do_constant_folding=True`. Build the FP16 TRT engine via
programmatic `generate_trt_engine()` (EXPLICIT_BATCH + OnnxParser + FP16). Cover
resolution buckets via **2–3 optimization profiles over total patch count L**,
not per-pixel spatial dims (Phase 5).

## 5. Subsystem 2 — LLM (Qwen2.5) + PBD / MTP decode

### 5.1 Backbone
Bit-identical vanilla Qwen2.5 dense decoder (RMSNorm pre-norm, GQA with QKV bias
+ biasless o_proj, RoPE, SwiGLU, lm_head). Reuse Eagle 2.5's
`export_llm_engine.py` + upstream TRT-LLM Qwen `convert_checkpoint.py` +
`tensorrt_llm.patch` (its `hf_config.text_config` unwrap is exactly what a nested
VLM config needs). **Override `_attn_implementation` from the forced default
`'magi'` to `'sdpa'`** before export — `flex_flash_attn` has no TRT op.

### 5.2 Engine contract
Inputs: `inputs_embeds[1,Lq,H]`, dense additive 4D float mask `[1,1,Lq,Lkv]`,
`position_ids[1,Lq]`, KV-cache I/O. Output: last-N logits. Vision/mlp1 embeds are
spliced into `image_token_index==151667` positions on the host; `inputs_embeds`
is fed to the engine.

### 5.3 The two masks (built in C++ each step)
- **AR / causal:** plain lower-triangular over `[0, kv_len]`.
- **MTP window** (`update_causal_mask_for_one_gen_window_2d`): three regions —
  (a) causal x0 prefix to `[0, prefix_len = kv_len - block_size)`; (b) the
  `block_size × block_size` bottom-right window is **bidirectional**; (c) the
  window attends fully to prefix keys but **column `kv_len - block_size - 1` is
  set to -inf** (the load-bearing "one blocked KV column" tied to the
  duplicated-last-token trick). An off-by-one here shifts every box.

`block_size = n_future_tokens = 6`. Query length is 6 (MTP) / 1 (AR) / S
(prefill) — covered by explicit optimization profiles.

### 5.4 The decode loop (port of `utils/locany` `generate()` + `generate_utils.py`)
- **MTP input build:** `generated_with_mask = [committed…, last_committed(dup),
  mask_id(151676) × 5]`. `position_ids = arange(start_idx…len)` then **subtract 1
  from the last `n_future_tokens` entries** (the PE drop marking the x0|MTP
  boundary). `start_idx` = current KV length.
- **Forward**, then **slice `logits[:, -6:, :]`**: pos0 = `<box>` slot, pos1–4 =
  coords, pos5 = terminator.
- **Sampling:** `apply_repetition_penalty` (over running unique ids), greedy
  argmax (T=0 per locked decision).
- **Box decode (`decode_bbox_avg`, `is_valid_box_frame`):** empty-box thresholds
  (`p_start≥0.6 & probs[1,none]>0.2 & probs[2,box_end]>0.2 & probs[3,null]>0.1 &
  probs[4,null]>0.1`); legal-box (`probs[5,{box_end,null,im_end}].sum()≥0.2` else
  illegal→None). For 4 coord slots: top-5, require a coord candidate per slot,
  take highest-prob coord; **hybrid ambiguity rule** per slot: if
  `top_coord_prob<0.9 AND #coord_candidates>1 AND (max_coord_id - min_coord_id)>60`
  → output sentinel id 0 (uncertain). `keep_k=5`.
- **`handle_pattern` FSM:** classifies coord_box (commit 6) / point_box
  (commit 4) / empty_box / error_box (hybrid: commit only the verified coord
  prefix, set `need_switch_to_ar`) / ref_object / im_end (terminal). This is the
  "format irregularity" detector that triggers AR revert.
- **KV-cache rollback:** after **every** forward, truncate K/V back to committed
  length so the 6 speculative columns are discarded. The KV manager must expose
  append-then-rollback.
- **Mode state machine (hybrid, v1 default):** start MTP;
  `error_box → use_mtp=false` (AR fallback); AR `box_end → use_mtp=true` (resume
  MTP); `im_end → break`. fast = always MTP; slow = always AR (internal
  checkpoint).
- **Stop:** committed length ≥ `min(model_max_length, S + max_new_tokens(2048))`,
  or `im_end`.

### 5.5 Token-id constants
Read from checkpoint config / tokenizer; do **not** hardcode (assigned at
`add_tokens` time). Indicative values from the released checkpoint:
`image_token_index=151667`, `box_start=151668`, `box_end=151669`,
`ref_start=151672`, `ref_end=151673`, `mask=151676`, `coord_start=151677`,
`coord_end=152677` (contiguous 1001-wide band; **assert
`coord_end - coord_start == 1000`**), `none=4064`, `null=152678`,
`im_end/eos=151645`, `switch=152679`.

## 6. Subsystem 3 — C++ host pipeline (tokenizer, preprocessing, API)

- **Tokenizer:** bind HF `tokenizers` (Rust) loading the checkpoint's
  `tokenizer.json`. Do not hand-port byte-level BPE; the 14 special tokens +
  1001 coord tokens are added at runtime and **add-order is load-bearing**.
- **Chat template:** reimplement `py_apply_chat_template` (system injection,
  `<|im_start|>/<|im_end|>` framing, `<image-k>` → `<img>` +
  `<IMG_CONTEXT>`×`((h*w)//4)` + `</img>`).
- **Preprocessing (CUDA/NPP):** RGB convert (RGBA→white composite),
  `in_token_limit` guard (config value **25600**, not the class-default 4096 —
  load from `preprocessor_config.json`), BICUBIC scale, ceil-to-multiple-of-28,
  normalize `x*2-1` (mean=std=0.5), patchify to `(L,3,14,14)` + `grid_hws`.
  **Must match PIL BICUBIC** (coordinate-sensitive); **assert `(h*w)//4` equals
  the `<IMG_CONTEXT>` count** or the splice misaligns (hard crash).
- **Public API:** mirror `LocateAnythingWorker` methods verbatim — prompt wording
  is load-bearing (e.g. `detect` joins categories with `</c>` and uses "matches";
  `ground_multi` uses "match"; `point` / `ground_gui`-point use "Point to:").
  `parse_boxes` vs `parse_points` are distinguished by integer-group count
  (4 vs 2) inside the same `<box></box>` wrapper.
- **Output decode:** coord value = `(token_id - coord_start)/1000`; map to pixels
  via the inverse of the resize/pad transform; handle `<box>none</box>`;
  sentinel id 0 = uncertain coordinate.
- **Coordinate ordering (Phase 0 verification):** `decode_bbox_avg` comments say
  `x1,x2,y1,y2` while `handle_pattern` / the template / the worker regex imply
  `<x1><y1><x2><y2>` (xyxy). The model does not reorder. **Confirm the
  checkpoint's actual convention by dumping a known prediction in Phase 0** and
  fix the C++ output ordering to match — do not assume.

## 7. Build and Deploy

- Base: `nvcr.io/nvidia/tensorrt:25.06-py3` (TRT 10.11.0.33, CUDA ~12.9,
  Python 3.12.3).
- TRT-LLM 0.20 built from source @ `7c828d7` with `tensorrt_llm.patch`
  (text_config unwrap, FindTensorRT, pinned version string),
  `ENABLE_MULTI_DEVICE=0`, cmake 3.27.6.
- Vision: `export_vision_onnx.py`-style wrapper → ONNX opset 17 → TRT FP16
  (programmatic build or trtexec with L-bucket profiles).
- LLM: `convert_checkpoint.py` → `trtllm-build --gemm_plugin auto
  --max_batch_size 1 --max_input_len <budget> --max_multimodal_len <budget>
  --max_seq_len <budget>`. Budget = `in_token_limit/4` visual tokens + prompt +
  2048 generated.
- C++ app: CMake linking TRT 10.11 + TRT-LLM cpp runtime + CUDA/NPP + HF
  tokenizers FFI.

## 8. Testing / Parity-Validation Strategy

Validate against the PyTorch reference (`locateanything_worker.py` +
`utils/locany`, run in **greedy** mode) at progressively higher integration
levels, gating each phase on a parity bar:

1. **Op-level unit tests (pre-export):** rewritten real-arithmetic `apply_rope`
   vs the `view_as_complex` path (near-bitwise); C++ 4D MTP mask bit-for-bit vs
   `update_causal_mask_for_one_gen_window_2d` for several `(kv_len, block_size)`;
   position-id construction (the −1 shift) vs reference; preprocessing
   pixel-for-pixel vs `image_processing_locateanything.py` and grid `(h*w)//4`
   placeholder count.
2. **Vision parity:** `vit_embeds` cosine similarity vs PyTorch bf16 reference per
   bucket (target ≥ 0.999 mean cosine — confirm bar in Phase 1).
3. **LLM NTP parity (slow mode):** greedy logits / argmax match on a fixed prompt
   set; lm_head + final softmax kept FP32.
4. **PBD parity (fast mode):** committed token sequences and per-step KV tensors
   match the PyTorch `generate()` each step; box frames identical.
5. **Hybrid parity:** mode-switch transitions (error_box → AR → box_end → MTP)
   match on a malformed-box / ambiguous-coordinate test set.
6. **End-to-end accuracy:** box IoU / point error / detection mAP on a
   calibration set vs the PyTorch reference, per generation_mode (FP16).

## 9. Phased Implementation Plan

| Phase | Goal | Key deliverable | Gate |
|---|---|---|---|
| **0 — Scaffolding & parity harness** | Build env, golden-reference harness, op-level bit-parity tests before any engine work | Docker image (tensorrt:25.06-py3 + TRT-LLM 0.20 @ 7c828d7 + patch); CMake C++ skeleton; HF tokenizers binding w/ verified token-id constants (coord band==1000, image_token_index); Python harness dumping reference tensors (pixel_values, grid_hws, vit_embeds, per-layer logits, per-step committed tokens + KV); unit tests for real-arithmetic `apply_rope`, C++ MTP/AR 4D masks, position-id −1 shift, CUDA/NPP preprocessing; **coordinate-ordering verification** | every op-level test near-bitwise; preprocessing pixel-for-pixel; `(h*w)//4` == `<IMG_CONTEXT>` count; coord ordering confirmed |
| **1 — Vision engine (fixed-res, N=1)** | Export MoonViT+mlp1 at one canonical bucket, validate embeddings | `FeatureExtractorWrapper` (baked RoPE/pos-emb, eager/SDPA, no mask at N=1, Conv2d→Gemm); ONNX opset 17; FP16 TRT engine; C++ path → `vit_embeds[L/4,H_llm]` | `vit_embeds` mean cosine ≥ agreed bar on calibration images |
| **2 — LLM engine + NTP decode (internal checkpoint)** | Stand up Qwen2.5 TRT-LLM engine, **resolve the runtime-dense-4D-mask / engine-count decision**, splice vision embeds, validate pure-AR correctness | `convert_checkpoint` + `trtllm-build` FP16 engine (attn → sdpa, magi dropped); C++ embed splice at `image_token_index`; KV management; slow (pure AR) decode loop w/ repetition penalty, coord detokenization, `parse_boxes`/`parse_points`; public API prompt builders | NTP greedy logits/argmax match PyTorch; slow-mode box IoU within tolerance |
| **3 — PBD / MTP fast mode** | Implement Parallel Box Decoding | MTP-window 4D mask (3-region + blocked column `kv_len-block_size-1`); MTP input builder (dup last + 5 mask tokens); `decode_bbox_avg`/`is_valid_box_frame`/`handle_pattern` ported with all thresholds; KV append-then-rollback; `fast` mode | committed tokens + per-step KV match PyTorch `generate()` bit-for-bit; box frames identical |
| **4 — Hybrid mode (v1 release)** | MTP↔AR fallback state machine | hybrid FSM (error_box→AR, box_end→MTP, im_end→stop), commit-only-verified-prefix on error_box, dual-profile engine for q_len ∈ {1,6,prefill} | mode transitions + final outputs match PyTorch hybrid reference on malformed/ambiguous set |
| **5 — Dynamic resolution + perf** | Multiple resolution buckets, latency tuning | 2–3 vision profiles over total patch count L; LLM profiles for q_len/kv_len; CUDA-graph / perf tuning; latency benchmarks per mode | accuracy held across buckets; documented latency win, no regression vs Phase 4 |

**v1 ships at the end of Phase 4** (FP16, hybrid PBD, single-image batch=1,
Hopper/Blackwell). Phase 5 is a fast-follow. Quantization (W4A16/FP8) is a
separate later effort gated on its own accuracy budget.

## 10. Top Risks

1. **PBD has no TRT-LLM analog.** The whole decode loop, the 3-region MTP-window
   mask as a runtime engine input, and per-step KV append-then-rollback must be
   bespoke C++ — likely bypassing `ModelRunner.generate` and possibly requiring a
   standalone TRT decoder engine. *Schedule-dominating.* Mitigation: de-risk the
   mask-input question in Phase 2; build NTP-only first as a known-good baseline.
2. **Load-bearing MTP details.** The position-id −1 shift on trailing future
   tokens, the single blocked KV column at `kv_len-block_size-1`, and the
   bidirectional window must be bit-for-bit. Any off-by-one silently shifts every
   box; KV rollback must discard the speculative window each step. Mitigation:
   Phase 0 bit-parity unit tests against the reference mask/position builders.
3. **PIL BICUBIC fidelity.** Coords are per-mille `[0,1000]` relative to the
   resized+padded canvas; resize divergence shifts every coordinate. Mitigation:
   a resize reproducing PIL BICUBIC, validated pixel-for-pixel; assert the
   `(h*w)//4` token count matches the placeholder count.
4. **FP16 numerical drift vs threshold-sensitive FSM.** `is_valid_box_frame`
   (`p_start≥0.6`, spread>60, end_score≥0.2) can flip under logit noise.
   Mitigation: keep lm_head + softmax FP32; gate FP16 on logit/argmax then
   end-to-end IoU parity before any further tuning.
5. **Runtime dense 4D mask support in TRT-LLM 0.20.** Built-ins are
   causal / sliding-window. If an arbitrary additive 4D mask isn't accepted,
   ENGINE 2 becomes a standalone TRT decoder engine with mask + KV as explicit
   I/O — a materially larger build. Mitigation: resolve in Phase 2 before
   committing the LLM architecture.
6. **Two divergent code copies.** Canonical inference logic is `utils/locany`
   (the `trust_remote_code` package the worker loads), not the training-time
   `model/locany`. Mitigation: confirm the shipped checkpoint loads `utils/locany`
   before freezing the port.

## 11. Resolved Decisions Log

- Target GPU: **Hopper/Blackwell**; magi dropped, custom SDPA-equivalent mask.
- Request shape: **single-image batch=1**.
- PBD: **required in v1** (ships at Phase 4); NTP is an internal Phase 2 gate.
- Decode: **greedy + tight parity**; no production sampling in v1.
- Quantization: **out of v1 scope**.
- Coordinate ordering: **verify in Phase 0** (do not assume xyxy).
