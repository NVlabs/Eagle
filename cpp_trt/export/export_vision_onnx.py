# SPDX-License-Identifier: Apache-2.0
"""
export_vision_onnx.py
=====================

Phase 1 vision export: load the LocateAnything-3B checkpoint, wrap MoonViT+mlp1
with ``FeatureExtractorWrapper`` at a *pinned* resolution bucket, and emit an
ONNX graph (opset 17) whose single input/output is

    input :  pixel_values  [L, 3, 14, 14]   (float32)
    output:  vit_embeds     [L/4, H_llm]     (float32, LLM space)

This script is written against the real checkpoint and must be RUN ON A
BLACKWELL MACHINE WITH torch + transformers==4.57.1 + the checkpoint. It is NOT
executed in the porting environment (no torch / no checkpoint here).

Mirrors the Eagle 2.5 ``deployment/export_vision_onnx.py`` pattern:
  * force an exportable attention impl ('sdpa'/'eager'),
  * cast the wrapper to ``.float()`` (FP32 ONNX; FP16 happens at TRT build),
  * ``torch.onnx.export(..., opset_version=17, do_constant_folding=True)``,
  * documented input/output names.

Run order (see README):
    python export_vision_onnx.py \
        --ckpt /path/to/LocateAnything-3B \
        --grid-h 64 --grid-w 64 \
        --out vision_b64x64.onnx \
        [--selftest-rope]

Bucket geometry
---------------
``--grid-h``/``--grid-w`` are the *patch* grid (image pixels / 14). They must be
even (2x2 merger). The pinned default bucket is 64x64 patches == 896x896 px ==
L=4096 patches -> 1024 merged LLM tokens. Pick the bucket that your host
preprocessing most commonly quantizes to under in_token_limit==25600
(=> L <= 25600, so 64x64 is well inside). Additional buckets are separate ONNX
exports in Phase 5.
"""

from __future__ import annotations

import argparse
import sys

import torch

from feature_extractor_wrapper import (
    FeatureExtractorWrapper,
    apply_rope_real,
    build_2d_rope_cos_sin,
)


# Documented ONNX I/O names (the C++ engine bindings rely on these).
INPUT_NAME = "pixel_values"
OUTPUT_NAME = "vit_embeds"


def load_model(ckpt_path: str):
    """Load LocateAnything via AutoModel(trust_remote_code=True).

    Returns (full_model, vit_module, mlp1_module, n_heads, hidden, h_llm).
    Attribute discovery is defensive because the exact field names could not be
    re-verified in the porting env; the bound names are printed for confirmation.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        ckpt_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()

    # force exportable attention before anything traces
    for cfg_attr in ("config", "vision_config"):
        cfg = getattr(model, cfg_attr, None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "sdpa"
    for m in model.modules():
        if hasattr(m, "_attn_implementation"):
            m._attn_implementation = "sdpa"

    # vision tower
    vit = None
    for name in ("vision_model", "vision_tower", "vit", "visual"):
        if hasattr(model, name):
            vit = getattr(model, name)
            print(f"[bind] vision tower  <- model.{name}  ({type(vit).__name__})")
            break
    if vit is None:
        raise RuntimeError("Could not locate the MoonViT vision tower on the model.")

    # mlp1 connector
    mlp1 = None
    for name in ("mlp1", "connector", "mm_projector", "multi_modal_projector"):
        if hasattr(model, name):
            mlp1 = getattr(model, name)
            print(f"[bind] mlp1 connector <- model.{name}  ({type(mlp1).__name__})")
            break
    if mlp1 is None:
        raise RuntimeError("Could not locate the mlp1 connector on the model.")

    # dims
    vcfg = getattr(vit, "config", None) or getattr(model, "vision_config", None)
    hidden = getattr(vcfg, "hidden_size", None) or getattr(vcfg, "embed_dim", 1152)
    n_heads = getattr(vcfg, "num_attention_heads", None) or getattr(vcfg, "num_heads", None)
    if n_heads is None:
        raise RuntimeError("Could not read MoonViT num_attention_heads from config.")

    # LLM hidden (output width of mlp1's last Linear)
    h_llm = None
    last_linear = [m for m in mlp1.modules() if isinstance(m, torch.nn.Linear)]
    if last_linear:
        h_llm = last_linear[-1].out_features
    print(f"[dims] hidden={hidden} n_heads={n_heads} head_dim={hidden // n_heads} H_llm={h_llm}")
    return model, vit, mlp1, int(n_heads), int(hidden), int(h_llm)


def selftest_rope(head_dim: int, grid_h: int, grid_w: int):
    """Op-level check: real-arithmetic apply_rope vs the EXACT MoonViT complex
    path (Rope2DPosEmb._precompute_freqs_cis + apply_rope).

    Phase-0/1 near-bitwise gate for the RoPE rewrite. No checkpoint needed.
    Rebuilds freqs_cis the same way modeling_vit.py does (x=col even slot,
    y=row odd slot, inv_freq=theta^(-4i/dim)) and the view_as_complex/polar
    multiply, then compares against build_2d_rope_cos_sin + apply_rope_real.
    """
    import math as _m
    torch.manual_seed(0)
    L = grid_h * grid_w
    H = 4
    theta_base = 10000.0
    x = torch.randn(L, H, head_dim, dtype=torch.float32)

    # ---- exact MoonViT freqs_cis (interleaved axes) -> complex apply_rope ----
    flat = torch.arange(L).float()
    x_pos = flat % grid_w                                 # column / width
    y_pos = torch.div(flat, grid_w, rounding_mode="floor")  # row / height
    dim_range = torch.arange(0, head_dim, 4)[: head_dim // 4].float()
    inv_freq = 1.0 / (theta_base ** (dim_range / head_dim))   # [dim//4]
    x_freqs = torch.outer(x_pos, inv_freq)
    y_freqs = torch.outer(y_pos, inv_freq)
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
    freqs_cis = freqs_cis.reshape(L, head_dim // 2)        # [L, dim/2] complex
    xc = torch.view_as_complex(x.view(L, H, head_dim // 2, 2))
    out_cplx = torch.view_as_real(xc * freqs_cis[:, None, :]).flatten(-2)

    # ---- rewritten real path ----
    cos, sin = build_2d_rope_cos_sin(grid_h, grid_w, head_dim)
    out_real = apply_rope_real(x, cos, sin)

    err = (out_real - out_cplx).abs().max().item()
    print(f"[selftest-rope] head_dim={head_dim} max abs err real-vs-complex = {err:.3e}")
    assert err < 1e-4, "RoPE rewrite does NOT match the MoonViT complex reference"
    print("[selftest-rope] PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export MoonViT+mlp1 to ONNX (fixed bucket)")
    ap.add_argument("--ckpt", required=True, help="Path to LocateAnything-3B checkpoint")
    ap.add_argument("--grid-h", type=int, default=64, help="patch grid height (px/14)")
    ap.add_argument("--grid-w", type=int, default=64, help="patch grid width  (px/14)")
    ap.add_argument("--out", default=None, help="output .onnx path")
    ap.add_argument("--selftest-rope", action="store_true",
                    help="run the real-vs-complex RoPE parity check and exit")
    args = ap.parse_args(argv)

    out = args.out or f"vision_b{args.grid_h}x{args.grid_w}.onnx"

    model, vit, mlp1, n_heads, hidden, h_llm = load_model(args.ckpt)
    head_dim = hidden // n_heads

    if args.selftest_rope:
        selftest_rope(head_dim, args.grid_h, args.grid_w)
        return

    wrapper = FeatureExtractorWrapper(
        vit=vit, mlp1=mlp1,
        grid_h=args.grid_h, grid_w=args.grid_w,
        n_heads=n_heads, hidden=hidden,
    ).float().eval()

    L = args.grid_h * args.grid_w
    dummy = torch.zeros(L, 3, 14, 14, dtype=torch.float32)

    # sanity forward (catches binding errors before the exporter)
    with torch.no_grad():
        y = wrapper(dummy)
    print(f"[shape] input pixel_values{[L,3,14,14]} -> output vit_embeds{list(y.shape)} "
          f"(expected [{L//4}, {h_llm}])")
    assert y.shape == (L // 4, h_llm), "output shape mismatch -- check binders"

    # Static geometry: no dynamic axes. One bucket == one engine in Phase 1.
    torch.onnx.export(
        wrapper,
        (dummy,),
        out,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[ok] wrote {out}")
    print(f"[bucket] grid=({args.grid_h},{args.grid_w}) "
          f"H={args.grid_h*14} W={args.grid_w*14} L={L} merged_tokens={L//4} H_llm={h_llm}")


if __name__ == "__main__":
    sys.exit(main())
