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
# LocateAnything-3B -- STANDALONE-TRT PATH, step 1 of 2: decoder -> ONNX.
#
# RUNS ELSEWHERE (torch + transformers==4.57.1; GPU optional for export). Exports
# the Qwen2.5 decoder stack with EXPLICIT inputs so the resulting TRT engine can
# take an ARBITRARY runtime dense 4D additive attention mask -- the thing the
# TRT-LLM path (built-in causal mask) cannot do. This is the path that supports
# Parallel Box Decoding (the 3-region MTP-window mask). See Risk R5.
#
# ENGINE CONTRACT (design Section 5.2), exported as ONNX graph I/O:
#   inputs:
#     inputs_embeds   [1, Lq, H]            fp32  (vision/mlp1 embeds already
#                                                  spliced at image_token_index
#                                                  on the host)
#     attn_mask_4d    [1, 1, Lq, Lkv]       fp32  (DENSE ADDITIVE mask: 0 keep,
#                                                  -inf block; built in C++)
#     position_ids    [1, Lq]               int64 (the MTP -1 shift on trailing
#                                                  n_future is applied on host)
#     past_key_<l>    [1, n_kv, Lpast, hd]  fp32  ) per layer
#     past_value_<l>  [1, n_kv, Lpast, hd]  fp32  )
#   outputs:
#     logits          [1, Lq, vocab]        fp32  (lm_head kept FP32; host does
#                                                  the FP32 softmax / argmax)
#     present_key_<l> [1, n_kv, Lpast+Lq, hd]
#     present_value_<l>[1, n_kv, Lpast+Lq, hd]
#
# The decoder stack ported is vanilla Qwen2.5 (design Section 5.1):
#   RMSNorm pre-norm, GQA with QKV bias + biasless o_proj, RoPE, SwiGLU, lm_head.
# We override _attn_implementation 'magi' -> 'eager'/'sdpa' first (flex_flash_attn
# has no TRT op) and we feed the dense additive mask straight into the attention
# score add so no built-in causal mask is generated inside the graph.
# =============================================================================

import argparse
import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

# Match convert_qwen_trtllm.py.
FORCED_ATTN_DEFAULT = "magi"
EXPORT_ATTN_IMPL = "eager"  # eager exposes the additive-mask add cleanly to ONNX
# Order matches the Eagle 2.5 tensorrt_llm.patch (llm_config then text_config).
# The LocateAnything checkpoint nests the Qwen2 decoder under `text_config`.
NESTED_TEXT_CONFIG_KEYS = ("llm_config", "text_config", "language_config")
OPSET = 17


def _load_unwrap_force_sdpa(model_dir: str):
    """Load HF config, unwrap nested VLM config, force non-magi attention."""
    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    for key in NESTED_TEXT_CONFIG_KEYS:
        sub = getattr(hf_config, key, None)
        if sub is not None:
            print(f"Unwrapped nested VLM config via hf_config.{key} "
                  f"(Eagle 2.5 tensorrt_llm.patch behaviour).")
            hf_config = sub
            break
    setattr(hf_config, "_attn_implementation", EXPORT_ATTN_IMPL)
    setattr(hf_config, "attn_implementation", EXPORT_ATTN_IMPL)
    os.environ["LOCANY_ATTN_IMPLEMENTATION"] = EXPORT_ATTN_IMPL
    print(f"Forced _attn_implementation '{FORCED_ATTN_DEFAULT}' -> "
          f"'{EXPORT_ATTN_IMPL}'.")
    return hf_config


class DecoderStepWrapper(nn.Module):
    """One decoder forward with explicit KV-cache and a dense 4D additive mask.

    Wraps the HF Qwen2.5 causal LM. The wrapper:
      * passes our additive mask straight through as `attention_mask` so the HF
        SDPA/eager path uses it verbatim (no internal causal mask synthesis when
        a 4D float mask is supplied),
      * threads the per-layer past_kv in/out as a legacy tuple cache,
      * returns FP32 logits for the host softmax.
    """

    def __init__(self, lm, num_layers: int):
        super().__init__()
        self.lm = lm
        self.num_layers = num_layers

    def forward(self, inputs_embeds, attn_mask_4d, position_ids, *past_kv):
        # Reassemble the legacy past_key_values tuple-of-tuples from the flat
        # (k0, v0, k1, v1, ...) arg list.
        past = tuple(
            (past_kv[2 * i], past_kv[2 * i + 1]) for i in range(self.num_layers)
        ) if len(past_kv) else None

        out = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask_4d,   # dense additive 4D mask, verbatim
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits.float()
        present = out.past_key_values
        flat_present = []
        for k, v in present:
            flat_present.append(k)
            flat_present.append(v)
        return (logits, *flat_present)


def build_io_names(num_layers: int):
    inputs = ["inputs_embeds", "attn_mask_4d", "position_ids"]
    for i in range(num_layers):
        inputs += [f"past_key_{i}", f"past_value_{i}"]
    outputs = ["logits"]
    for i in range(num_layers):
        outputs += [f"present_key_{i}", f"present_value_{i}"]
    return inputs, outputs


def build_dynamic_axes(input_names, output_names):
    axes = {
        "inputs_embeds": {1: "Lq"},
        "attn_mask_4d": {2: "Lq", 3: "Lkv"},
        "position_ids": {1: "Lq"},
        "logits": {1: "Lq"},
    }
    for name in input_names:
        if name.startswith("past_key_") or name.startswith("past_value_"):
            axes[name] = {2: "Lpast"}
    for name in output_names:
        if name.startswith("present_key_") or name.startswith("present_value_"):
            axes[name] = {2: "Lpresent"}
    return axes


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Export the LocateAnything Qwen2.5 decoder to ONNX with "
        "explicit inputs_embeds + attn_mask_4d + position_ids + past_kv "
        "(standalone-TRT path; supports the PBD 4D mask).")
    p.add_argument("--model_dir", type=str, required=True,
                   help="LocateAnything HF checkpoint dir.")
    p.add_argument("--onnx_path", type=str, default="decoder.onnx",
                   help="Output ONNX file.")
    p.add_argument("--seq_len", type=int, default=6,
                   help="Dummy query length for tracing (6 = MTP window). The "
                   "axis is dynamic; this only seeds the trace.")
    p.add_argument("--past_len", type=int, default=16,
                   help="Dummy past KV length for tracing (dynamic at runtime).")
    return p.parse_args()


def main():
    args = parse_arguments()
    cfg = _load_unwrap_force_sdpa(args.model_dir)

    # Load the full LM in FP32 on CPU for a clean, deterministic export.
    lm = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        config=cfg,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation=EXPORT_ATTN_IMPL,
    ).eval()

    H = cfg.hidden_size
    num_layers = cfg.num_hidden_layers
    num_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = H // cfg.num_attention_heads

    wrapper = DecoderStepWrapper(lm, num_layers).eval()

    Lq, Lpast = args.seq_len, args.past_len
    Lkv = Lpast + Lq
    NEG = torch.finfo(torch.float32).min

    dummy_embeds = torch.zeros(1, Lq, H, dtype=torch.float32)
    # Dense additive 4D mask: 0 = attend, NEG = blocked. (Lower-triangular over
    # the new block + full attention to the past, as a representative shape.)
    dummy_mask = torch.zeros(1, 1, Lq, Lkv, dtype=torch.float32)
    for qi in range(Lq):
        for kj in range(Lpast + qi + 1, Lkv):
            dummy_mask[0, 0, qi, kj] = NEG
    dummy_pos = torch.arange(Lpast, Lpast + Lq, dtype=torch.int64).unsqueeze(0)
    past = []
    for _ in range(num_layers):
        past.append(torch.zeros(1, num_kv, Lpast, head_dim, dtype=torch.float32))
        past.append(torch.zeros(1, num_kv, Lpast, head_dim, dtype=torch.float32))

    input_names, output_names = build_io_names(num_layers)
    dynamic_axes = build_dynamic_axes(input_names, output_names)

    print(f"Exporting decoder: H={H} layers={num_layers} kv_heads={num_kv} "
          f"head_dim={head_dim} -> {args.onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_mask, dummy_pos, *past),
            args.onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            do_constant_folding=True,
        )
    print(f"Done. ONNX written to {args.onnx_path}")
    print("Next: build_decoder_trt.py to build the FP16 engine with KV as IO.")


if __name__ == "__main__":
    main()
