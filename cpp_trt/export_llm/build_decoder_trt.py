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
# LocateAnything-3B -- STANDALONE-TRT PATH, step 2 of 2: ONNX -> FP16 TRT engine.
#
# RUNS ELSEWHERE (Blackwell GPU + TensorRT 10.11.0.33). Programmatic FP16 build
# of the decoder ONNX from export_decoder_onnx.py, with the KV cache wired as
# explicit engine I/O and an optimization profile covering the three query
# lengths used by the decode loop:
#     q_len = 1       (AR step)
#     q_len = 6       (MTP window / block_size = n_future_tokens)
#     q_len = prefill (the full input sequence; sized from the length budget)
# and KV length growing from 0 up to the budget.
#
# This engine accepts the runtime DENSE 4D additive mask (attn_mask_4d) needed
# for Parallel Box Decoding -- the capability the TRT-LLM path lacks (Risk R5).
# lm_head output (logits) is kept FP32 so the C++ host softmax/argmax and the
# threshold-sensitive box FSM stay numerically stable (design Risk R4).
# =============================================================================

import argparse
import sys

import tensorrt as trt

# Length budget components (design Sections 5.4 / 6 / 7), mirrored from
# build_trtllm_engine.py so both paths size identically.
DEFAULT_IN_TOKEN_LIMIT = 25600
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_PROMPT_BUDGET = 512

# Decode-loop query lengths.
Q_MIN = 1     # AR step
Q_OPT = 6     # MTP window (block_size = n_future_tokens)
# Q_MAX = prefill length = budget (set below).


def compute_budget(in_token_limit, prompt_budget, max_new_tokens):
    return (in_token_limit // 4) + prompt_budget + max_new_tokens


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Build an FP16 TensorRT engine for the LocateAnything "
        "decoder with KV as IO and a runtime 4D mask (standalone path).")
    p.add_argument("--onnx_path", type=str, required=True,
                   help="Decoder ONNX from export_decoder_onnx.py.")
    p.add_argument("--engine_path", type=str, default="decoder_fp16.engine",
                   help="Output serialized engine.")
    p.add_argument("--hidden_size", type=int, required=True,
                   help="Decoder hidden size H (from the checkpoint config).")
    p.add_argument("--num_layers", type=int, required=True,
                   help="num_hidden_layers (for per-layer KV IO profiles).")
    p.add_argument("--num_kv_heads", type=int, required=True,
                   help="num_key_value_heads.")
    p.add_argument("--head_dim", type=int, required=True,
                   help="head_dim = hidden_size // num_attention_heads.")
    p.add_argument("--in_token_limit", type=int, default=DEFAULT_IN_TOKEN_LIMIT)
    p.add_argument("--prompt_budget", type=int, default=DEFAULT_PROMPT_BUDGET)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--workspace_gb", type=int, default=8,
                   help="TRT builder workspace pool in GiB.")
    return p.parse_args()


def add_profile(builder, network, args, budget):
    """One optimization profile spanning q_len {1,6,prefill} and growing KV."""
    profile = builder.create_optimization_profile()
    H = args.hidden_size
    n_kv = args.num_kv_heads
    hd = args.head_dim

    q_max = budget
    kv_min, kv_opt, kv_max = 1, budget // 2, budget

    # Map ONNX input names -> their dynamic dim ranges.
    for i in range(network.num_inputs):
        t = network.get_input(i)
        name = t.name
        if name == "inputs_embeds":
            profile.set_shape(name, (1, Q_MIN, H), (1, Q_OPT, H), (1, q_max, H))
        elif name == "attn_mask_4d":
            profile.set_shape(name,
                              (1, 1, Q_MIN, kv_min),
                              (1, 1, Q_OPT, kv_opt),
                              (1, 1, q_max, kv_max))
        elif name == "position_ids":
            profile.set_shape(name, (1, Q_MIN), (1, Q_OPT), (1, q_max))
        elif name.startswith("past_key_") or name.startswith("past_value_"):
            # past length ranges from 0 (prefill) to budget-1.
            profile.set_shape(name,
                              (1, n_kv, 0, hd),
                              (1, n_kv, kv_opt, hd),
                              (1, n_kv, kv_max, hd))
        else:
            raise RuntimeError(f"Unexpected ONNX input: {name}")
    return profile


def main():
    args = parse_arguments()
    budget = compute_budget(args.in_token_limit, args.prompt_budget,
                            args.max_new_tokens)
    print(f"Length budget (prefill q_max / kv_max) = {budget}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    with open(args.onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 args.workspace_gb * (1 << 30))
    # FP16 weights/activations; logits output stays FP32 (marked in the ONNX).
    config.set_flag(trt.BuilderFlag.FP16)

    config.add_optimization_profile(add_profile(builder, network, args, budget))

    print("Building FP16 engine (this can take several minutes on Blackwell)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("Engine build failed.", file=sys.stderr)
        sys.exit(1)

    with open(args.engine_path, "wb") as f:
        f.write(serialized)
    print(f"Done. Standalone decoder engine written to {args.engine_path}")
    print("This engine accepts the runtime 4D MTP mask required for PBD "
          "(Phase 3+). KV cache is explicit engine I/O.")


if __name__ == "__main__":
    main()
