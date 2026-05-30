# Phase 1 — Vision Export Pipeline (MoonViT + mlp1 → ONNX → TRT FP16)

This directory exports the LocateAnything-3B vision tower (MoonViT) and its
`mlp1` connector into a single TensorRT FP16 engine:

```
pixel_values[L, 3, 14, 14]  ──►  ENGINE 1  ──►  vit_embeds[L/4, H_llm]   (LLM space)
```

`H_llm` is the Qwen2.5 decoder hidden size; `L = grid_h * grid_w` is the patch
count for one **pinned resolution bucket** (N = 1, single image). See design doc
`docs/superpowers/specs/2026-05-29-locateanything-cpp-trt-design.md` §4 and §9
(Phase 1).

> These scripts are written against the **real checkpoint** and are intended to
> be **run on a Blackwell GPU** with TensorRT installed. They are *not* runnable
> in the porting environment (no torch / no TRT / no checkpoint there).

## Files

| file | purpose |
|------|---------|
| `feature_extractor_wrapper.py` | `FeatureExtractorWrapper(nn.Module)` — wraps MoonViT+mlp1 at pinned geometry; bakes bicubic abs pos-emb and 2D-RoPE cos/sin to constants; rewrites `apply_rope` to real interleaved cos/sin arithmetic; static 2×2 patch-merger; no attention mask (cu_seqlens=`[0,L]`); emits `vit_embeds[L/4, H_llm]`. |
| `export_vision_onnx.py` | load checkpoint → wrap → `torch.onnx.export(opset 17, do_constant_folding=True, .float())`; documented I/O names; `--selftest-rope` op-level RoPE parity check. |
| `build_vision_engine.py` | programmatic TRT build (EXPLICIT_BATCH + OnnxParser + FP16) for one fixed L. Marks where Phase 5 L-bucket optimization profiles go. |
| `compare_vision_parity.py` | torch wrapper vs TRT engine on a calibration image → mean cosine similarity of `vit_embeds` (Phase 1 gate ≥ 0.999). |

## Required environment

- **GPU:** Blackwell (RTX 5090) or Hopper.
- **TensorRT 10.11** (matches the Eagle 2.5 toolchain, `nvcr.io/nvidia/tensorrt:25.06-py3`).
- **Python:** 3.12.
- **PyTorch** with CUDA (build/parity only).
- **transformers==4.57.1** (pinned; the checkpoint loads via `trust_remote_code`).
- **onnx**, and for parity: **pycuda**, **numpy**, **Pillow**.
- The **LocateAnything-3B checkpoint** (config + tokenizer + weights).

```bash
pip install "transformers==4.57.1" onnx numpy Pillow pycuda
# tensorrt comes from the TRT 10.11 install / container
```

## Run order

```bash
# 0) (recommended) verify the real-arithmetic RoPE rewrite matches the complex
#    reference BEFORE building anything. No checkpoint needed for the math, but
#    --ckpt is required to read head_dim.
python export_vision_onnx.py --ckpt /path/to/LocateAnything-3B \
    --grid-h 64 --grid-w 64 --selftest-rope

# 1) export ONNX at the pinned bucket (default 64x64 patches = 896x896 px,
#    L=4096, 1024 merged LLM tokens)
python export_vision_onnx.py --ckpt /path/to/LocateAnything-3B \
    --grid-h 64 --grid-w 64 --out vision_b64x64.onnx

# 2) build the FP16 TRT engine
python build_vision_engine.py \
    --onnx vision_b64x64.onnx \
    --engine vision_b64x64.fp16.plan --workspace-gb 8

# 3) parity gate (target mean cosine >= 0.999)
python compare_vision_parity.py \
    --ckpt /path/to/LocateAnything-3B \
    --engine vision_b64x64.fp16.plan \
    --image calib.jpg --grid-h 64 --grid-w 64 --force-grid
```

The emitted engine (`.plan`) and the documented I/O names `pixel_values` /
`vit_embeds` are consumed by the C++ vision stage.

## Pinned bucket

| field | value (default bucket) |
|-------|------------------------|
| patch grid (H,W) | 64 × 64 |
| image px (H,W) | 896 × 896 |
| L (patches) | 4096 |
| merged LLM tokens (L/4) | 1024 |
| input | `pixel_values [4096, 3, 14, 14]` fp32 |
| output | `vit_embeds [1024, H_llm]` fp32 (engine internally FP16) |

Pick the bucket matching how the host preprocessing quantizes H,W to multiples of
28 under `in_token_limit = 25600`. Each bucket is a separate ONNX/engine pair in
Phase 1; Phase 5 unifies 2–3 buckets behind optimization profiles over `L` (see
the marked block in `build_vision_engine.py`).

## Verified against source / items to confirm on the build machine

The wrapper was written against the re-read `modeling_vit.py` and the C++
`cpp_trt/src/vision_rope/`. The following are matched to source:

- **RoPE** — `build_2d_rope_cos_sin` / `apply_rope_real` reproduce
  `Rope2DPosEmb._precompute_freqs_cis` + `apply_rope` exactly: `inv_freq[i] =
  theta^(-4i/head_dim)`, even slot = column/width (`pos % grid_w`), odd slot =
  row/height (`pos // grid_w`), adjacent-pair complex multiply. cos/sin width is
  `head_dim//2 == 36` (head_dim = 1152/16 = **72**), identical in layout to the
  C++ `compute_rope_freqs`. `export_vision_onnx.py --selftest-rope` asserts
  near-bitwise (<1e-4) match vs the `view_as_complex`/`polar` path. **Run it.**
- **Block structure** — `MoonVitEncoderLayer`: `norm0` (pre-attn), `wqkv`
  (bias=True, 3·hidden), `wo` (bias=True), `norm1` (pre-mlp), `mlp` =
  `MLP2(fc0 → GELU-tanh → fc1)` (1152→4304→1152). qkv: `view(L,3,16,72)` then
  `unbind(-3)`. Encoder ends with `final_layernorm`. Patch-merger has no params.
- **abs pos-emb** — `Learnable2DInterpPosEmb.weight` is `[64, 64, 1152]`; baked
  via `F.interpolate(weight.permute(2,0,1)[None], size=(gh,gw), mode="bicubic")`
  with torch-default `align_corners=False` (the eager code passes no
  `align_corners`). If `(gh,gw)==(64,64)` the weight is used directly (no
  interpolate), matching the eager fast path.
- **2×2 patch-merger** — `view(new_h,2,new_w,2,d).permute(0,2,1,3,4)` →
  concat order (top-left, top-right, bottom-left, bottom-right). Matches eager.

Confirm on the build machine:

1. **mlp1 internals** — the LocateAnything connector is applied via its own
   `forward` (`model.mlp1`), so its `LayerNorm(4608) → Linear → GELU → Linear`
   and exact GELU variant come straight from the checkpoint. The loader prints
   which attribute it bound; confirm it is the no-pixel-shuffle mlp1.
2. **Submodule binding printout** — the loader/wrapper print bound module names;
   confirm against the checkpoint and adjust candidate lists if a name differs.
3. **`num_attention_heads`** — config default is 16 (head_dim 72). Read from the
   actual config (the loader does); the head count drives the RoPE width.
