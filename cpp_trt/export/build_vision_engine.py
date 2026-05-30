# SPDX-License-Identifier: Apache-2.0
"""
build_vision_engine.py
======================

Programmatic TensorRT FP16 engine build for the Phase 1 vision ONNX
(MoonViT + mlp1). Mirrors the Eagle 2.5 ``generate_trt_engine()`` pattern:
EXPLICIT_BATCH network + OnnxParser + FP16 builder flag.

RUN ON A BLACKWELL MACHINE with TensorRT 10.11 installed. Not executed in the
porting env (no TRT here).

    python build_vision_engine.py \
        --onnx vision_b64x64.onnx \
        --engine vision_b64x64.fp16.plan \
        [--workspace-gb 8] [--no-fp16]

Phase 1 == ONE fixed L (the ONNX is static, no dynamic axes), so no optimization
profile is required. The L-bucket optimization profiles (2-3 ranges over the
total patch count L) belong in Phase 5; see the clearly marked block below for
where they go.
"""

from __future__ import annotations

import argparse
import sys

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_gb: int = 8,
):
    builder = trt.Builder(TRT_LOGGER)

    # EXPLICIT_BATCH (matches Eagle 2.5 generate_trt_engine).
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[onnx-parse-error] {parser.get_error(i)}", file=sys.stderr)
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb) * (1 << 30)
    )

    if fp16:
        if not builder.platform_has_fast_fp16:
            print("[warn] platform reports no fast FP16; building FP16 anyway")
        config.set_flag(trt.BuilderFlag.FP16)

    # --- log the bindings (sanity that input/output names match the contract) ---
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"[input ] {t.name} {tuple(t.shape)} {t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"[output] {t.name} {tuple(t.shape)} {t.dtype}")

    # =====================================================================
    # PHASE 5 — L-bucket optimization profiles go HERE.
    # ---------------------------------------------------------------------
    # In Phase 5 the ONNX is re-exported with a dynamic leading dim L on
    # pixel_values ([L,3,14,14]) and on vit_embeds ([L/4,H_llm]). Then add one
    # profile spanning 2-3 buckets over the TOTAL PATCH COUNT L (not per-pixel
    # H/W), e.g.:
    #
    #     profile = builder.create_optimization_profile()
    #     profile.set_shape("pixel_values",
    #                       min=(L_min, 3, 14, 14),
    #                       opt=(L_opt, 3, 14, 14),
    #                       max=(L_max, 3, 14, 14))
    #     config.add_optimization_profile(profile)
    #
    # For multiple discrete buckets, either add multiple profiles (each opt at a
    # bucket centroid) or keep one wide profile. The baked pos-emb / RoPE cos-sin
    # constants are bucket-specific, so dynamic-L additionally requires those to
    # become inputs or per-bucket engines -- a Phase 5 design decision.
    # In Phase 1 the network is fully static; no profile is added.
    # =====================================================================

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None (build failed)")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[ok] wrote engine {engine_path} (fp16={fp16})")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build FP16 TRT engine for vision ONNX")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--workspace-gb", type=int, default=8)
    ap.add_argument("--no-fp16", action="store_true", help="build FP32 instead")
    args = ap.parse_args(argv)
    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        fp16=not args.no_fp16,
        workspace_gb=args.workspace_gb,
    )


if __name__ == "__main__":
    sys.exit(main())
