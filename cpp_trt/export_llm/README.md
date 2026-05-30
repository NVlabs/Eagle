# LocateAnything-3B — LLM (Qwen2.5) export / engine-build scripts

Phase-2 deliverables for the **decoder** side of the LocateAnything C++/TRT port.

> **These Python scripts run ELSEWHERE.** They require torch, transformers,
> TensorRT, and TensorRT-LLM and must run on a Blackwell (or Hopper) GPU box
> with the released LocateAnything checkpoint. They are **not** part of the C++
> CMake build and are never executed in the C++ CI environment (which has no
> TRT / TRT-LLM / PyTorch / checkpoint). The pure-C++ decode loop, sampling,
> repetition penalty, coordinate detokenize, box/point parsing, and prompt
> builders live under the C++ source tree and are unit-tested there — they have
> **no** dependency on anything in this directory.

---

## Two build paths (dual-engine design)

The Qwen2.5 transformer core is stock and TRT-LLM supports it out of the box —
but **Parallel Box Decoding (PBD) has no native TRT-LLM analog.** PBD needs a
bespoke 3-region dense 4D *additive* attention mask supplied at runtime, and
TRT-LLM 0.20's built-in masking is only causal / sliding-window. Hence two
paths:

| Path | Scripts | Mask | Use when |
|------|---------|------|----------|
| **TRT-LLM** | `convert_qwen_trtllm.py` → `build_trtllm_engine.py` | **built-in causal** (no runtime 4D mask) | Phase 2 NTP / pure-AR correctness checkpoint. Fast to stand up; known-good baseline. |
| **Standalone TRT** | `export_decoder_onnx.py` → `build_decoder_trt.py` | **runtime dense 4D additive mask** (explicit engine input) | Phase 3+ Parallel Box Decoding (MTP-window mask). The path that exists *because* TRT-LLM may not accept the dense 4D MTP mask. |

### Risk R5 (why the standalone path exists)

> TRT-LLM 0.20's built-in attention masking is causal / sliding-window. If an
> arbitrary additive 4D mask is not accepted at runtime, ENGINE 2 must become a
> **standalone TRT decoder engine** with the mask + KV cache as explicit I/O.
> The MTP-window mask (3 regions: causal x0 prefix; bidirectional
> `block_size × block_size` bottom-right window; the single blocked KV column at
> `kv_len - block_size - 1`) is **not expressible** through TRT-LLM's built-ins.
>
> **Decision rule:** stand up the TRT-LLM path first and validate NTP / pure-AR
> parity (Phase 2). Before committing the LLM architecture, verify whether
> TRT-LLM 0.20 accepts a runtime dense 4D mask. If it does not — which is the
> expected outcome for PBD — switch ENGINE 2 to the standalone path for Phase 3
> onward. The standalone engine is a strict superset (it can also run pure AR
> with a causal 4D mask), so it can replace, not just supplement, the TRT-LLM
> engine once PBD is required.

---

## Required environment (run ELSEWHERE)

- **GPU:** Hopper / Blackwell (e.g. RTX 5090 / H100). FP16.
- **Base image:** `nvcr.io/nvidia/tensorrt:25.06-py3` (TensorRT **10.11.0.33**,
  CUDA ~12.9, Python 3.12.3).
- **TensorRT-LLM 0.20** built from source @ commit `7c828d7` with
  `tensorrt_llm.patch` (the `hf_config.text_config` unwrap, `FindTensorRT`,
  pinned version string), `ENABLE_MULTI_DEVICE=0`, cmake 3.27.6.
  Only required for the **TRT-LLM path**.
- **transformers==4.57.1** (matches the checkpoint's `trust_remote_code`
  modeling files). Required for both paths.
- **torch** (CUDA build matching the image). Required for the standalone-path
  ONNX export and recommended for the TRT-LLM conversion.
- The released **LocateAnything checkpoint** (config.json + *.safetensors +
  trust_remote_code modeling files + `preprocessor_config.json`).

The config-unwrap and the `magi → sdpa` attention override are re-implemented
inside these scripts so they do not depend on the exact contents of the
upstream `tensorrt_llm.patch` at runtime; the patch is still required to build
TRT-LLM 0.20 itself.

---

## Run order

### TRT-LLM path (Phase 2 NTP)

```bash
# 1. HF checkpoint -> TRT-LLM checkpoint
#    - unwraps nested VLM config (hf_config.text_config) like the Eagle 2.5 patch
#    - forces _attn_implementation 'magi' -> 'sdpa' BEFORE conversion
python convert_qwen_trtllm.py \
    --model_dir   /path/to/locateanything_checkpoint \
    --output_dir  ./tllm_checkpoint \
    --dtype       float16

# 2. TRT-LLM checkpoint -> serialized engine (built-in causal mask)
#    budget = in_token_limit//4 (=6400) + prompt + 2048
python build_trtllm_engine.py \
    --checkpoint_dir ./tllm_checkpoint \
    --output_dir     ./trtllm_engine \
    --gemm_plugin    auto \
    --in_token_limit 25600
```

Produces:

```
trtllm-build --checkpoint_dir ./tllm_checkpoint --output_dir ./trtllm_engine \
             --gemm_plugin auto --max_batch_size 1 \
             --max_input_len 8960 --max_seq_len 8960 --max_multimodal_len 8960
```

(`8960 = 25600//4 + 512 + 2048`; pass `--dry_run` to print the command without
building.)

### Standalone-TRT path (Phase 3+ PBD)

```bash
# 1. Decoder stack -> ONNX (explicit inputs_embeds + attn_mask_4d
#    + position_ids + past_kv, opset 17)
python export_decoder_onnx.py \
    --model_dir /path/to/locateanything_checkpoint \
    --onnx_path ./decoder.onnx

# 2. ONNX -> FP16 TRT engine with KV as IO and a 4D-mask-capable profile
#    (q_len in {1, 6, prefill}; KV grows 0..budget). Pull H / layers / kv_heads
#    / head_dim from the checkpoint config.
python build_decoder_trt.py \
    --onnx_path     ./decoder.onnx \
    --engine_path   ./decoder_fp16.engine \
    --hidden_size   2048 \
    --num_layers    36 \
    --num_kv_heads  2 \
    --head_dim      128 \
    --in_token_limit 25600
```

> The `--hidden_size / --num_layers / --num_kv_heads / --head_dim` values above
> are placeholders — read the real values from the **unwrapped** decoder config
> of the released checkpoint. `export_decoder_onnx.py` prints them during export.

---

## Engine contracts

### TRT-LLM engine (`build_trtllm_engine.py`)
Standard TRT-LLM Qwen executor I/O. **Built-in causal masking only.** Cannot
take an arbitrary runtime 4D mask → NTP / pure-AR only.

### Standalone engine (`export_decoder_onnx.py` + `build_decoder_trt.py`)
Design Section 5.2 contract, as explicit ONNX/TRT I/O:

| Tensor | Shape | dtype | Notes |
|--------|-------|-------|-------|
| `inputs_embeds` | `[1, Lq, H]` | fp32 | vision/mlp1 embeds spliced at `image_token_index` (151667) on the host |
| `attn_mask_4d` | `[1, 1, Lq, Lkv]` | fp32 | dense **additive** mask (0 keep, −inf block); causal or MTP-window, built in C++ |
| `position_ids` | `[1, Lq]` | int64 | the MTP `−1` shift on the trailing `n_future_tokens` is applied on the host |
| `past_key_<l>` / `past_value_<l>` | `[1, n_kv, Lpast, head_dim]` | fp32 | per-layer KV cache in; host does append-then-rollback |
| `logits` | `[1, Lq, vocab]` | **fp32** | lm_head kept FP32 for the threshold-sensitive box FSM (Risk R4) |
| `present_key_<l>` / `present_value_<l>` | `[1, n_kv, Lpast+Lq, head_dim]` | fp32 | per-layer KV cache out |

`block_size = n_future_tokens = 6`; query length is 1 (AR) / 6 (MTP) / prefill,
all covered by the single optimization profile in `build_decoder_trt.py`.

---

## Sizing (both paths)

```
budget = visual_tokens + prompt + generated
       = (in_token_limit // 4) + prompt(512) + max_new_tokens(2048)
```

`in_token_limit = 25600` comes from `preprocessor_config.json` — **not** the
class default of 4096. The `//4` is the 2×2 patch-merger (4 patches → 1 visual
LLM token). With the defaults: `6400 + 512 + 2048 = 8960`.

---

## What is NOT here

These scripts only export/build the **decoder**. The vision (MoonViT + mlp1)
export, the C++ host decode loop / PBD FSM / box parsing / prompt builders, the
tokenizer FFI, and the CUDA/NPP preprocessing live elsewhere in the port. Per
the locked scope: no quantization, no sampling (`do_sample=True`) — greedy only;
the `magi` backend is dropped.
