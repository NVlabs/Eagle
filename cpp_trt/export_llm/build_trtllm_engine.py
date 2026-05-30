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
# LocateAnything-3B -- TRT-LLM PATH, step 2 of 2: trtllm-build.
#
# RUNS ELSEWHERE (Blackwell GPU + TRT-LLM 0.20 @ 7c828d7). Takes the TRT-LLM
# checkpoint produced by convert_qwen_trtllm.py and invokes `trtllm-build` to
# produce a serialized engine.
#
#   trtllm-build --checkpoint_dir <ckpt> --output_dir <engine>
#                --gemm_plugin auto
#                --max_batch_size 1
#                --max_input_len  <budget>
#                --max_seq_len    <budget>
#                --max_multimodal_len <budget>
#
# *** MASKING LIMITATION (Risk R5) ***
# The engine built here uses TRT-LLM's BUILT-IN attention masking
# (causal / sliding-window). This is sufficient for the Phase-2 NTP / pure-AR
# correctness checkpoint. It CANNOT take an arbitrary runtime dense 4D additive
# mask, so it CANNOT express the 3-region MTP-window mask that Parallel Box
# Decoding needs in Phase 3. For PBD use the standalone-TRT path
# (export_decoder_onnx.py + build_decoder_trt.py). See README.md / design R5.
#
# SIZING. The design (Section 7) sets the length budget to:
#     budget = visual_tokens + prompt_tokens + generated_tokens
#            = (in_token_limit // 4)  +  prompt  +  2048
# with in_token_limit = 25600 (preprocessor_config.json value, NOT the class
# default 4096) -> 6400 visual tokens. We set --max_input_len / --max_seq_len /
# --max_multimodal_len all to this budget for the single-image batch=1 shape.
# =============================================================================

import argparse
import subprocess
import sys

# preprocessor_config.json in_token_limit for the released checkpoint
# (design Section 6). The class default of 4096 is WRONG for this model.
DEFAULT_IN_TOKEN_LIMIT = 25600
# Max new tokens the decode loop may emit (design Section 5.4 stop rule).
DEFAULT_MAX_NEW_TOKENS = 2048
# Headroom for the chat-template prompt (system + task wording + framing).
# Generous; the real prompt is tiny vs the visual-token budget.
DEFAULT_PROMPT_BUDGET = 512


def compute_budget(in_token_limit: int, prompt_budget: int,
                   max_new_tokens: int) -> int:
    """Length budget = visual tokens (in_token_limit//4) + prompt + generated.

    in_token_limit counts raw 14x14 patches; the 2x2 patch-merger turns every 4
    patches into 1 visual LLM token, hence //4. This must bound the worst-case
    input sequence (prompt + all <IMG_CONTEXT> placeholders) plus the generated
    tail.
    """
    visual_tokens = in_token_limit // 4
    return visual_tokens + prompt_budget + max_new_tokens


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build a TensorRT-LLM engine for the LocateAnything "
        "Qwen2.5 decoder (TRT-LLM path, NTP/Phase-2; built-in causal mask).")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="TRT-LLM checkpoint dir from convert_qwen_trtllm.py.")
    parser.add_argument("--output_dir", type=str, default="trtllm_engine",
                        help="Where to write the serialized engine.")
    parser.add_argument(
        "--in_token_limit", type=int, default=DEFAULT_IN_TOKEN_LIMIT,
        help="Raw patch budget from preprocessor_config.json (default 25600). "
        "Visual LLM tokens = in_token_limit // 4.")
    parser.add_argument("--prompt_budget", type=int, default=DEFAULT_PROMPT_BUDGET,
                        help="Reserved prompt-token headroom.")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="Generated-token budget (design stop rule = 2048).")
    parser.add_argument("--gemm_plugin", type=str, default="auto",
                        help="trtllm-build --gemm_plugin value.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the trtllm-build command without running it.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    budget = compute_budget(args.in_token_limit, args.prompt_budget,
                            args.max_new_tokens)
    visual_tokens = args.in_token_limit // 4
    print(f"Length budget = {visual_tokens} visual + {args.prompt_budget} prompt "
          f"+ {args.max_new_tokens} generated = {budget} tokens")

    cmd = [
        "trtllm-build",
        "--checkpoint_dir", args.checkpoint_dir,
        "--output_dir", args.output_dir,
        "--gemm_plugin", args.gemm_plugin,
        "--max_batch_size", "1",            # single-image batch=1 (design 2)
        "--max_input_len", str(budget),
        "--max_seq_len", str(budget),
        "--max_multimodal_len", str(budget),
    ]
    print("trtllm-build command:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        print("[dry-run] not executing.")
        return

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"trtllm-build failed with exit code {proc.returncode}",
              file=sys.stderr)
        sys.exit(proc.returncode)

    print(f"Done. TRT-LLM engine written to: {args.output_dir}")
    print("NOTE: this engine uses BUILT-IN causal masking (Risk R5). For PBD "
          "(runtime 4D MTP mask) build the standalone engine instead.")


if __name__ == "__main__":
    main()
