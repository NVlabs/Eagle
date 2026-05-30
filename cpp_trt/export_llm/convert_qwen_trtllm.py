# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
# =============================================================================
# LocateAnything-3B -- TRT-LLM PATH, step 1 of 2: HF -> TRT-LLM checkpoint.
#
# THIS SCRIPT IS WRITTEN TO RUN ELSEWHERE (Blackwell GPU box with TensorRT-LLM
# 0.20 @ 7c828d7 installed). It is NOT executed in the C++/CMake build tree and
# pulls in torch / transformers / tensorrt_llm, none of which exist in the C++
# CI environment. See README.md for the required environment and run order.
#
# What it does (mirrors Eagle 2.5 deployment/export_llm_engine.py):
#   1. Load the LocateAnything Qwen2.5 checkpoint config with
#      trust_remote_code=True.
#   2. UNWRAP the nested VLM config: a LocateAnything checkpoint exposes the
#      decoder hyper-params under hf_config.text_config (or .llm_config). The
#      upstream TRT-LLM Qwen converter expects a *flat* Qwen2 config at the top
#      level, exactly the situation the Eagle 2.5 tensorrt_llm.patch fixes with
#      its `hf_config = hf_config.text_config` unwrap. We do the same here so the
#      converter sees hidden_size / num_attention_heads / num_key_value_heads /
#      intermediate_size / vocab_size / rope_theta from the *decoder*, not the
#      VLM wrapper.
#   3. Force `_attn_implementation` from the checkpoint default 'magi'
#      (flex_flash_attn -- no TRT op) to 'sdpa' BEFORE any export/convert call.
#   4. Run the upstream TRT-LLM Qwen convert flow (QWenForCausalLM
#      .from_hugging_face -> save_checkpoint), producing a TRT-LLM checkpoint
#      directory consumed by build_trtllm_engine.py.
#
# This produces an engine that uses TRT-LLM's BUILT-IN causal masking. That is
# sufficient for the Phase-2 NTP / pure-AR correctness checkpoint, but it CANNOT
# accept an arbitrary runtime dense 4D additive mask. The Parallel Box Decoding
# (PBD / MTP-window) mask needed in Phase 3 is NOT expressible here -- for that,
# use the standalone-TRT path (export_decoder_onnx.py + build_decoder_trt.py).
# See Risk R5 in the design doc.
# =============================================================================

import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# transformers==4.57.1 is required (matches the released checkpoint's
# trust_remote_code modeling files). See README.md.
from transformers import AutoConfig

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.logger import logger
from tensorrt_llm.models import QWenForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig


# The forced default attention backend baked into the LocateAnything modeling
# code. 'magi' == flex_flash_attn and has no TRT / TRT-LLM op; we must swap it
# for an SDPA-equivalent dense path before export.
FORCED_ATTN_DEFAULT = "magi"
EXPORT_ATTN_IMPL = "sdpa"

# Field names a nested VLM config may use to hold the decoder sub-config.
# Order matches the Eagle 2.5 tensorrt_llm.patch: it checks `llm_config` first,
# then `text_config`. The released LocateAnything checkpoint nests the Qwen2
# decoder under `text_config` (LocateAnythingConfig.sub_configs), so that branch
# is the one that fires here; `llm_config` is kept first for patch-parity.
NESTED_TEXT_CONFIG_KEYS = ("llm_config", "text_config", "language_config")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert a LocateAnything Qwen2.5 HF checkpoint to a "
        "TensorRT-LLM checkpoint (TRT-LLM path, NTP/Phase-2).")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the LocateAnything HF checkpoint (the directory that "
        "holds config.json + the *.safetensors + trust_remote_code modeling "
        "files).")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tllm_checkpoint",
        help="Where to write the TensorRT-LLM checkpoint (input to "
        "build_trtllm_engine.py).")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Engine weight/activation dtype. v1 ships FP16 (the design keeps "
        "lm_head + final softmax in FP32 on the C++ host, not in the engine).")
    parser.add_argument("--tp_size", type=int, default=1,
                        help="Tensor-parallel size. batch=1 single-GPU -> 1.")
    parser.add_argument("--pp_size", type=int, default=1,
                        help="Pipeline-parallel size. Single-GPU -> 1.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel rank-conversion workers.")
    parser.add_argument(
        "--load_model_on_cpu",
        action="store_true",
        help="Load HF weights on CPU during conversion (lower GPU memory).")
    return parser.parse_args()


def _load_and_unwrap_config(model_dir: str):
    """Load the HF config and return a *flat* Qwen2 decoder config.

    A LocateAnything checkpoint nests the decoder hyper-params under
    hf_config.text_config (the VLM wrapper config sits on top, carrying the
    MoonViT/vision + connector fields). The upstream TRT-LLM Qwen converter
    expects a flat Qwen2 config, so we unwrap exactly like the Eagle 2.5
    tensorrt_llm.patch does.
    """
    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    unwrapped_from = None
    for key in NESTED_TEXT_CONFIG_KEYS:
        sub = getattr(hf_config, key, None)
        if sub is not None:
            hf_config = sub
            unwrapped_from = key
            break

    if unwrapped_from is not None:
        logger.info(f"Unwrapped nested VLM config: using hf_config.{unwrapped_from} "
                    f"as the flat Qwen2 decoder config "
                    f"(Eagle 2.5 tensorrt_llm.patch behaviour).")
    else:
        logger.info("Config is already flat (no text_config/llm_config nesting "
                    "found); using it directly.")

    return hf_config


def _force_sdpa_attention(model_dir: str, hf_config) -> None:
    """Override the forced 'magi' attention backend to 'sdpa' BEFORE export.

    LocateAnything bakes _attn_implementation='magi' (flex_flash_attn) as the
    default; that op has no TRT lowering. We set it to sdpa on the config object
    *and* via the env var the trust_remote_code modeling code reads, so any
    re-load during conversion also picks sdpa.
    """
    # On the config object the converter inspects.
    setattr(hf_config, "_attn_implementation", EXPORT_ATTN_IMPL)
    setattr(hf_config, "attn_implementation", EXPORT_ATTN_IMPL)
    # Belt-and-braces: the LocateAnything modeling code consults this env var
    # when constructing attention if present; force it so a fresh from_pretrained
    # during conversion cannot resurrect 'magi'.
    os.environ["LOCANY_ATTN_IMPLEMENTATION"] = EXPORT_ATTN_IMPL
    logger.info(f"Forced _attn_implementation: '{FORCED_ATTN_DEFAULT}' -> "
                f"'{EXPORT_ATTN_IMPL}' (flex_flash_attn has no TRT op).")


def convert_and_save_hf(args) -> None:
    model_dir = args.model_dir
    world_size = args.tp_size * args.pp_size

    # Load + unwrap + force-sdpa happen up front, before any TRT-LLM model
    # construction, so the converter never sees 'magi' or the nested config.
    hf_config = _load_and_unwrap_config(model_dir)
    _force_sdpa_attention(model_dir, hf_config)

    # No quantization in v1 (out of scope per design Section 2). FP16 only.
    quant_config = QuantConfig()

    from tensorrt_llm.mapping import Mapping

    def convert_and_save_rank(args, rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)
        qwen = QWenForCausalLM.from_hugging_face(
            model_dir,
            args.dtype,
            mapping=mapping,
            quant_config=quant_config,
            load_model_on_cpu=args.load_model_on_cpu,
            # Pass the unwrapped, sdpa-forced config so the loader uses the
            # flat decoder hyper-params, not the VLM wrapper.
            override_fields={},
        )
        qwen.save_checkpoint(args.output_dir, save_config=(rank == 0))
        del qwen

    if args.workers == 1:
        for rank in range(world_size):
            convert_and_save_rank(args, rank)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(convert_and_save_rank, args, rank)
                       for rank in range(world_size)]
            errors = []
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    errors.append(exc)
            assert not errors, "Checkpoint conversion failed; see log above."

    release_gc()


def main():
    print(f"tensorrt_llm version: {tensorrt_llm.__version__}")
    args = parse_arguments()

    tik = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    convert_and_save_hf(args)
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - tik))
    print(f"Done. TRT-LLM checkpoint written to: {args.output_dir}")
    print(f"Total conversion time: {elapsed}")
    print("Next: run build_trtllm_engine.py on this output_dir.")


if __name__ == "__main__":
    main()
