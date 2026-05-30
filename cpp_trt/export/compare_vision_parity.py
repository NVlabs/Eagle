# SPDX-License-Identifier: Apache-2.0
"""
compare_vision_parity.py
========================

Phase 1 parity gate: run the PyTorch ``FeatureExtractorWrapper`` (reference) vs
the FP16 TRT engine on a calibration image and report the **mean cosine
similarity** of ``vit_embeds``. Target: >= 0.999 (the Phase 1 gate from the
design doc, Section 8 step 2).

RUN ON A BLACKWELL MACHINE with torch + transformers==4.57.1 + TensorRT 10.11 +
the checkpoint. Not executed in the porting env.

    python compare_vision_parity.py \
        --ckpt /path/to/LocateAnything-3B \
        --engine vision_b64x64.fp16.plan \
        --image calib.jpg \
        --grid-h 64 --grid-w 64

Preprocessing note
------------------
The host C++ preprocessing (CUDA/NPP, PIL-BICUBIC) is the production path. Here
we reuse the checkpoint's own image processor (``trust_remote_code``) to produce
``pixel_values[L,3,14,14]`` so the parity test isolates the *engine vs torch
wrapper* delta, not preprocessing differences. If the processor yields a grid
different from (grid_h, grid_w) for this image, pass an image already sized to
the bucket (grid*14 px), or override with ``--force-grid``.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from feature_extractor_wrapper import FeatureExtractorWrapper
from export_vision_onnx import load_model, INPUT_NAME, OUTPUT_NAME


def make_pixel_values(ckpt, image_path, grid_h, grid_w, force_grid):
    """Return pixel_values[L,3,14,14] float32 for the calibration image."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")

    if force_grid:
        # bypass the processor: resize to bucket and patchify with x*2-1 norm.
        H, W = grid_h * 14, grid_w * 14
        img = img.resize((W, H), Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0                       # mean=std=0.5 -> x*2-1
        t = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]
        t = t.reshape(3, grid_h, 14, grid_w, 14)
        t = t.permute(1, 3, 0, 2, 4).reshape(grid_h * grid_w, 3, 14, 14)
        return t.contiguous(), grid_h, grid_w

    # Use the checkpoint's processor (matches training preprocessing).
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(ckpt, trust_remote_code=True)
    out = proc(images=img, return_tensors="pt")
    pv = out["pixel_values"]
    # processor may return [L,3,14,14] directly, or [1,3,H,W] needing patchify.
    if pv.dim() == 4 and pv.shape[-2:] == (14, 14):
        L = pv.shape[0]
        gh = out.get("image_grid_thw", out.get("grid_hws", None))
        if gh is not None:
            gh = gh.flatten().tolist()
            ggh, ggw = int(gh[-2]), int(gh[-1])
        else:
            ggh, ggw = grid_h, grid_w
        return pv.float(), ggh, ggw
    raise RuntimeError(
        f"Processor returned pixel_values shape {tuple(pv.shape)}; "
        "use --force-grid to patchify manually for the pinned bucket."
    )


def run_torch(ckpt, pv, grid_h, grid_w):
    model, vit, mlp1, n_heads, hidden, h_llm = load_model(ckpt)
    wrapper = FeatureExtractorWrapper(
        vit=vit, mlp1=mlp1, grid_h=grid_h, grid_w=grid_w,
        n_heads=n_heads, hidden=hidden,
    ).float().eval()
    with torch.no_grad():
        return wrapper(pv).cpu().numpy()


def run_trt(engine_path, pv):
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
    import pycuda.driver as cuda

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    pv_np = np.ascontiguousarray(pv.cpu().numpy().astype(np.float32))
    # set input shape (works for static and dynamic engines)
    ctx.set_input_shape(INPUT_NAME, tuple(pv_np.shape))

    out_shape = tuple(ctx.get_tensor_shape(OUTPUT_NAME))
    out_np = np.empty(out_shape, dtype=np.float32)

    d_in = cuda.mem_alloc(pv_np.nbytes)
    d_out = cuda.mem_alloc(out_np.nbytes)
    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_in, pv_np, stream)
    ctx.set_tensor_address(INPUT_NAME, int(d_in))
    ctx.set_tensor_address(OUTPUT_NAME, int(d_out))
    ctx.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(out_np, d_out, stream)
    stream.synchronize()
    return out_np


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float((num / den).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 1 vision parity gate (cosine sim)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--grid-h", type=int, default=64)
    ap.add_argument("--grid-w", type=int, default=64)
    ap.add_argument("--force-grid", action="store_true",
                    help="manual resize+patchify to the pinned bucket")
    ap.add_argument("--gate", type=float, default=0.999)
    args = ap.parse_args(argv)

    pv, gh, gw = make_pixel_values(args.ckpt, args.image, args.grid_h, args.grid_w, args.force_grid)
    print(f"[pixel_values] L={pv.shape[0]} grid=({gh},{gw})")

    ref = run_torch(args.ckpt, pv, gh, gw)
    eng = run_trt(args.engine, pv)
    print(f"[shapes] torch={ref.shape} trt={eng.shape}")
    if ref.shape != eng.shape:
        raise RuntimeError("Shape mismatch between torch and TRT outputs")

    cos = mean_cosine(ref, eng)
    print(f"[parity] mean cosine similarity = {cos:.6f}  (gate >= {args.gate})")
    if cos >= args.gate:
        print("[parity] PASS")
        return 0
    print("[parity] FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
