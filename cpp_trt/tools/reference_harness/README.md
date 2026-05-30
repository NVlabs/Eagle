# Golden-Reference Harness (`dump_reference.py`)

This directory contains the **PyTorch golden-reference dumper** for the
LocateAnything-3B model. It produces the frozen "golden vectors" that the C++ /
TensorRT port is validated against in the Phase 1-4 parity gates.

> **This script does NOT run in the C++/TRT build environment.**
> The build box (RTX-5090 / Blackwell, CUDA 13.1, g++ 15.2, no PyTorch, no
> TensorRT, no checkpoint) cannot execute it. It is a **pure-Python, offline**
> artifact that must be run **once** on a separate GPU machine that has PyTorch
> and the model checkpoint. The resulting `.npy` files + `manifest.json` are then
> copied into the repo and consumed (read-only) by the C++ tests.

The script is written against the **actual canonical API** (it imports and
drives `LocateAnythingWorker` and `LocateAnythingForConditionalGeneration` —
it does not re-implement the model). See the docstring in `dump_reference.py`
for the exact code path it mirrors.

---

## 1. Required environment (where you DO run it)

| Requirement | Notes |
|---|---|
| GPU | Hopper (H100/H200) **or** Blackwell (RTX-5090), CUDA build of PyTorch. |
| Python | 3.10-3.12 |
| `torch` | CUDA build matching the GPU. |
| `transformers` | **`==4.57.1`** (the `trust_remote_code` model code is version-sensitive). |
| `numpy`, `pillow`, `torchvision`, `peft` | deps of the canonical `locany` package (the model uses `peft.LoraConfig`, the processor uses `torchvision`). |
| Optional | `flash-attn` (MoonViT falls back to sdpa if absent); `decord`/`lmdb`/`cv2` only if you feed video/lmdb inputs (not needed for a plain image). |
| Checkpoint | **LocateAnything-3B** HF dir (e.g. `nvidia/LocateAnything-3B`). |
| Canonical code | `Embodied/eaglevl/utils/locany/` and `Embodied/locateanything_worker.py` from this repo, importable via `EAGLE_ROOT`. |

```bash
python -m venv .venv && source .venv/bin/activate
pip install "transformers==4.57.1" pillow numpy torchvision peft
# install the torch wheel that matches YOUR GPU/CUDA:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
export EAGLE_ROOT=/path/to/Eagle      # repo root containing Embodied/
```

(If `EAGLE_ROOT` is unset, the script assumes it lives three levels above
itself, i.e. the repo root.)

---

## 2. Exact command

```bash
export EAGLE_ROOT=/path/to/Eagle

python /path/to/Eagle/cpp_trt/tools/reference_harness/dump_reference.py \
    --checkpoint     nvidia/LocateAnything-3B \
    --image          /path/to/sample.jpg \
    --prompt         "Locate all the instances that match the following description: the red mug." \
    --out-dir        /path/to/Eagle/cpp_trt/tools/reference_harness/golden/case01 \
    --generation-mode hybrid \
    --max-new-tokens 512 \
    --mtp-steps      0,1,2 \
    --temperature    0 \
    --seed           0
```

Run it once per `--generation-mode` you intend to validate (`fast` exercises the
pure-MTP window, `slow` exercises pure-AR, `hybrid` exercises the switch logic).
Then copy/commit `golden/caseNN/` into the repo.

### Prompt formats (from `LocateAnythingWorker`)
- Detection: `Locate all the instances that matches the following description: <c1></c><c2>.`
- Phrase grounding (multi): `Locate all the instances that match the following description: <phrase>.`
- Phrase grounding (single): `Locate a single instance that matches the following description: <phrase>.`
- Pointing: `Point to: <phrase>.`
- Scene text: `Detect all the text in box format.`

---

## 3. What it dumps

All tensors are `.npy`, indexed by `manifest.json`. bf16 is upcast to fp32 for
storage (NumPy has no bf16); the C++ side loads/compares in fp32.

| File | Meaning | Phase |
|---|---|---|
| `pixel_values.npy` | ViT input (post image-processor) | 1 |
| `image_grid_hws.npy` | per-image `(h,w)` grid sizes (int32) | 1 |
| `input_ids.npy` | tokenized prompt incl. `<IMG_CONTEXT>` | 2 |
| `image_token_positions.npy` | flat indices where `input_ids == image_token_index` (151667) | 2 |
| `vit_embeds_raw.npy` | **raw** `vision_model` output, pre-`mlp1` (ViT-only parity) | 1 |
| `vit_embeds.npy` | **post-`mlp1`** projected features that are actually spliced | 1/2 |
| `inputs_embeds_text_only.npy` | text embeds before vision splice | 2 |
| `inputs_embeds.npy` | spliced embeds = prefill input to the LM | 2 |
| `committed_token_ids.npy` / `generated_token_ids.npy` | newly generated token ids | 3 |
| `output_token_ids.npy` | prompt + generated ids | 3 |
| `step_commit_counts.npy` | window length fed at each LM forward | 3/4 |
| `kv_cache_lengths.npy` | KV-cache length **after** each LM forward | 3 |
| `step_<i>_mtp_mask.npy` | attention mask fed at LM forward `i` | 4 |
| `step_<i>_position_ids.npy` | `position_ids` at LM forward `i` (MTP: last `n_future_tokens` get `-1`) | 4 |
| `step_<i>_input_ids.npy` | the `generated_with_mask` window at LM forward `i` | 4 |
| `parsed_boxes.npy` | parsed boxes `[n,4]` xyxy in **pixel** coords | 4 |

`manifest.json` also records: provenance (checkpoint/image/prompt, torch/
transformers/numpy versions, GPU name, dtype), all config **token ids**
(`box_start=151668`, `box_end=151669`, `coord_start=151677`,
`coord_end=152677`, `none=4064`, `default_mask=151676`, `im_end=151645`, …),
`image_token_index=151667`, `n_future_tokens=6`, the decoded answer string,
parsed boxes/points, and the source image size.

---

## 4. The exact behavior these vectors encode (for the C++ implementer)

Read directly from the canonical sources; the C++ port must reproduce:

1. **Splice** (`modeling_locateanything.py`):
   `vit = mlp1(cat(vision_model(pixel_values, grid_hws)))`, then
   `input_embeds[input_ids == 151667] = vit[:n_selected]`. `mlp1` is
   `LayerNorm(4*vit_C) -> Linear(4*vit_C, llm_C) -> GELU -> Linear(llm_C, llm_C)`.
   The merge folds a `2x2` patch block (`merge_kernel_size`) — hence `4*vit_C`.
2. **MTP window** (`generate._prepare_inputs_in_mtp`, `n_future_tokens=6`):
   `generated_with_mask = cat(generated, generated[:,-1:], 5 x mask_tok)` where
   `mask_tok = default_mask_token_id = 151676`. `position_ids` are sliced from a
   plain `arange`, then `position_ids[0, -n_future_tokens:] -= 1`. After the
   forward, the KV cache is truncated back to `generated.shape[1]`.
3. **MTP sampling / box decode** (`generate_utils.py`):
   `sample_tokens` reads the last 6 logits, `decode_bbox_avg` does a top-k
   weighted/argmax box decode with `start_thresh`/`end_thresh`/abnormal-spread
   rules; `handle_pattern` classifies `coord_box` / `point_box` / `empty_box` /
   `error_box` / `ref_object` / `im_end` and decides MTP↔AR switching (hybrid).
4. **Termination**: `im_end_token_id = 151645` (or `null_token_id = 152678`).

The per-step `step_<i>_*` dumps capture (2) numerically for the first few steps
so the C++ MTP path can be diffed window-by-window.

---

## 5. Determinism & provenance (IMPORTANT for the parity gates)

* Use `--temperature 0` (greedy). The canonical sampler is stochastic when
  `temperature > 0`; only `temperature == 0` yields reproducible golden vectors.
  (Note: production `LocateAnythingWorker.predict` defaults to `temperature=0.7,
  do_sample=True` — that is for serving, **not** for golden capture.)
* **Bitwise identity across different GPUs is NOT guaranteed** (SM arch,
  cuBLAS/cuDNN heuristics, bf16 rounding). C++ parity gates **must compare with a
  tolerance**. Suggested: bf16 feature tensors `rtol=2e-2, atol=2e-2`; token ids
  exact; box pixel coords within ~1px. Regenerate + re-baseline if you change GPU
  or dtype — both are recorded in `manifest.json`.

---

## 6. Caveats / what this script cannot do here

* It does **not** build or run any C++/TensorRT code; it only emits references.
* It cannot run in the build environment (no torch/checkpoint) — by design.
* `generated_token_ids` are recovered by **re-tokenizing the decoded answer**
  (the canonical `generate()` returns a string, not ids). If the tokenizer round
  trip is lossy for special `<box>`/coord tokens, use the per-step
  `step_<i>_input_ids.npy` dumps for the exact committed ids instead. A future
  improvement is to patch `generate()` to also return `generated` ids directly.
* Per-step capture relies on wrapping `model.language_model.forward`. If a build
  wraps that callable (e.g. `torch.compile`), the wrapper may not fire; the
  manifest then carries `step_capture_warning`.
* PyTorch↔TRT numerical parity and TRT engine behavior are out of scope — those
  are asserted by the Phase 1-4 C++ gates that *consume* these vectors.
