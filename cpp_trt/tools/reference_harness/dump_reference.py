#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_reference.py - Golden-reference tensor dumper for LocateAnything-3B parity testing.

=============================================================================
!!! THIS SCRIPT CANNOT BE RUN IN THE C++/TRT BUILD ENVIRONMENT !!!
=============================================================================
It REQUIRES, none of which are present on the RTX-5090 / CUDA-13.1 build box:
  * PyTorch (CUDA build matching your GPU; Hopper H100/H200 or Blackwell RTX-5090)
  * transformers==4.57.1
  * numpy, pillow, torchvision, peft   (deps of the canonical package)
  * The LocateAnything-3B checkpoint (HF-format directory)
  * The canonical inference package (imported, NOT re-implemented):
        Embodied/eaglevl/utils/locany/           (model + preprocessing)
        Embodied/locateanything_worker.py        (LocateAnythingWorker)

It is a PURE-PYTHON, OFFLINE artifact. It is NOT part of the C++ build/test
pipeline. Run it ONCE on a GPU box that has torch + the checkpoint; it writes a
directory of .npy files + manifest.json. Those become the FROZEN golden vectors
checked into the repo and consumed by the C++ Phase 1-4 parity gates (which
compare engine/kernel outputs against these .npy tensors with a tolerance).

See README.md (same directory) for the exact environment and command.

-----------------------------------------------------------------------------
WHY THIS IS WRITTEN AGAINST THE REAL CANONICAL API (not duck-typed)
-----------------------------------------------------------------------------
The canonical sources were read and this harness targets their EXACT behavior:

  * Loader (locateanything_worker.LocateAnythingWorker.__init__):
        AutoTokenizer + AutoProcessor + AutoModel, trust_remote_code=True,
        torch_dtype=bfloat16, device "cuda", .eval().
  * Preprocess (LocateAnythingWorker.predict):
        messages = [{"role":"user","content":[{"type":"image","image":img},
                                              {"type":"text","text":question}]}]
        text   = processor.py_apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        images, videos = processor.process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos,
                           return_tensors="pt").to(device)
        pixel_values   = inputs["pixel_values"].to(bfloat16)
        input_ids      = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)   # numpy [n,2] (h,w)
  * Vision + splice (LocateAnythingForConditionalGeneration):
        vit = vision_model(pixel_values=pv, grid_hws=image_grid_hws)  # list
        vit = mlp1(cat(vit, dim=0))                                   # PROJECTED
        selected = (input_ids == config.image_token_index)  # 151667
        input_embeds[selected] = vit[:selected.sum()]
    NOTE: the golden "vit_embeds" we dump are the POST-mlp1 features that are
    actually spliced (that is what the C++ side must reproduce to splice).
    We also dump the raw pre-mlp1 ViT output separately for ViT-only parity.
  * Decode loop (LocateAnythingForConditionalGeneration.generate):
        Fully custom MTP / AR / hybrid loop, NOT transformers.generate.
        n_future_tokens=6, default_mask_token_id=151676.
        MTP step builds generated_with_mask = cat(generated, last_tok,
            5 x mask_tok); position_ids sliced from arange and the LAST
            n_future_tokens entries get -= 1; KV cache truncated to
            generated.shape[1] each step. Returns a DECODED STRING.
  * Box parse (LocateAnythingWorker.parse_boxes):
        regex r"<box><(\\d+)><(\\d+)><(\\d+)><(\\d+)></box>", coords in [0,1000],
        scaled by image w/h. Points: r"<box><(\\d+)><(\\d+)></box>".

We instrument the live model with a thin wrapper around
model.language_model.forward so we capture, per decode step, the EXACT
inputs_embeds / position_ids / attention_mask / KV length that the canonical
loop feeds in, plus the committed token id. This avoids re-implementing the
loop while still exposing every tensor the parity gates need.

-----------------------------------------------------------------------------
WHAT GETS DUMPED  (all under --out-dir, indexed by manifest.json)
-----------------------------------------------------------------------------
  pixel_values.npy           : ViT input (fp32)                       [n_patches, ...]
  image_grid_hws.npy         : per-image (h,w) grid sizes (int32)     [n_images, 2]
  input_ids.npy              : tokenized prompt incl. <IMG_CONTEXT>    [1, seq_len] (i64)
  image_token_positions.npy  : flat indices where input_ids==151667   [n_img_tok] (i64)
  vit_embeds_raw.npy         : raw vision_model output (pre-mlp1)      [n_img_tok, vit_C] (fp32)
  vit_embeds.npy             : POST-mlp1 features actually spliced     [n_img_tok, llm_C] (fp32)
  inputs_embeds_text_only.npy: text embeds before vision splice       [1, seq, llm_C] (fp32)
  inputs_embeds.npy          : spliced embeds fed to the LM at prefill [1, seq, llm_C] (fp32)
  committed_token_ids.npy    : token id(s) committed at each step      [n_committed] (i64)
  step_commit_counts.npy     : how many tokens each LM forward committed[n_steps] (i64)
  kv_cache_lengths.npy       : KV length AFTER each LM forward         [n_steps] (i64)
  step_<i>_mtp_mask.npy      : attention mask fed at LM forward i      (i8 / i64)
  step_<i>_position_ids.npy  : position_ids fed at LM forward i        [1, win] (i64)
  step_<i>_input_ids.npy     : the generated_with_mask window at step i[1, win] (i64)
  output_token_ids.npy       : full generated token ids (incl. prompt)[1, total] (i64)
  generated_token_ids.npy    : newly generated token ids only         [1, n_new] (i64)
  parsed_boxes.npy           : parsed boxes in PIXEL coords [n,4] xyxy (fp32)

manifest.json records provenance (checkpoint/image/prompt, torch/transformers/
numpy versions, GPU name, dtype), all config token ids, n_future_tokens, the
decoded answer string, parsed boxes+labels, image size, and per-tensor metadata.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Hard dependency check with an explicit, friendly failure.
# ---------------------------------------------------------------------------
def _require_runtime_deps():
    missing: List[str] = []
    for mod, label in (("numpy", "numpy"), ("torch", "torch (CUDA build)"),
                       ("transformers", "transformers==4.57.1"),
                       ("PIL", "pillow")):
        try:
            __import__(mod)
        except Exception:
            missing.append(label)
    if missing:
        sys.stderr.write(
            "\n[dump_reference.py] FATAL: missing runtime dependencies: "
            + ", ".join(missing)
            + "\n\nThis is the GOLDEN-REFERENCE harness; it MUST run on a GPU box\n"
            "with PyTorch + transformers==4.57.1 + the LocateAnything-3B checkpoint.\n"
            "It cannot run in the C++/TRT build environment. See README.md.\n\n"
        )
        raise SystemExit(2)
    import numpy as np
    import torch
    import transformers
    return np, torch, transformers


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class TensorEntry:
    name: str
    file: str
    shape: List[int]
    dtype: str
    note: str = ""


class Manifest:
    def __init__(self) -> None:
        self.meta: Dict[str, Any] = {}
        self.tensors: List[TensorEntry] = []
        self.extra: Dict[str, Any] = {}

    def add(self, name: str, file: str, arr, note: str = "") -> None:
        self.tensors.append(TensorEntry(name, file, list(arr.shape),
                                        str(arr.dtype), note))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 2,
            "meta": self.meta,
            "tensors": [dataclasses.asdict(t) for t in self.tensors],
            "extra": self.extra,
        }


def _to_numpy(np, t):
    """torch.Tensor / list / ndarray -> contiguous CPU numpy (bf16 -> fp32)."""
    try:
        import torch
        if isinstance(t, torch.Tensor):
            t = t.detach().to("cpu").contiguous()
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float32)
            return t.numpy()
    except Exception:
        pass
    return np.asarray(t)


def _save(np, out_dir: str, name: str, tensor, manifest: Manifest,
          note: str = "") -> None:
    arr = _to_numpy(np, tensor)
    fname = f"{name}.npy"
    np.save(os.path.join(out_dir, fname), arr)
    manifest.add(name, fname, arr, note=note)
    print(f"  [saved] {fname:30s} shape={list(arr.shape)} dtype={arr.dtype}")


# ---------------------------------------------------------------------------
# Per-step capture installed as a wrapper around language_model.forward.
# The canonical generate() calls self.language_model(**prepare_inputs) once per
# decode step (prefill = step 0). We record what it actually receives and the
# KV length it produces. The committed-token reconstruction is done afterwards
# by diffing the final `generated` against the prompt.
# ---------------------------------------------------------------------------
class LMForwardRecorder:
    def __init__(self, torch) -> None:
        self.torch = torch
        self.position_ids: List[Any] = []
        self.attention_mask: List[Any] = []
        self.input_ids: List[Any] = []
        self.inputs_embeds: List[Any] = []
        self.kv_len_after: List[int] = []

    def wrap(self, language_model):
        orig_forward = language_model.forward
        torch = self.torch

        def wrapped(*args, **kwargs):
            # Record inputs to this LM forward (the MTP / AR window).
            pid = kwargs.get("position_ids", None)
            am = kwargs.get("attention_mask", None)
            iid = kwargs.get("input_ids", None)
            ie = kwargs.get("inputs_embeds", None)
            self.position_ids.append(_detach(torch, pid))
            self.attention_mask.append(_detach(torch, am))
            self.input_ids.append(_detach(torch, iid))
            self.inputs_embeds.append(_detach(torch, ie))
            out = orig_forward(*args, **kwargs)
            # KV length after this forward.
            self.kv_len_after.append(_kv_len(getattr(out, "past_key_values", None)))
            return out

        language_model.forward = wrapped
        self._restore = lambda: setattr(language_model, "forward", orig_forward)
        return self

    def restore(self):
        if hasattr(self, "_restore"):
            self._restore()


def _detach(torch, x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    return None


def _kv_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    try:
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length())
    except Exception:
        pass
    try:
        return int(past_key_values[0][0].shape[-2])  # [b, heads, seq, dim]
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Vision splice capture: hook the model.forward path is not used by generate()
# (generate has its own splice), so we capture the vision pieces directly:
#   - raw vision_model output (pre-mlp1)
#   - post-mlp1 projected features (what gets spliced)
#   - text-only and spliced inputs_embeds, reconstructed deterministically.
# We compute these with a single manual forward replicating the EXACT splice in
# LocateAnythingForConditionalGeneration.forward / generate, using the model's
# own submodules (so weights/dtypes match) but on CPU-copied tensors for dump.
# ---------------------------------------------------------------------------
def capture_vision_and_splice(model, pixel_values, image_grid_hws, input_ids,
                              torch) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cfg = model.config
    image_token_index = int(cfg.image_token_index)
    out["image_token_index"] = image_token_index

    dev = next(model.parameters()).device
    pv = pixel_values.to(model.language_model.dtype).to(dev)
    grid = image_grid_hws
    if grid is not None and not isinstance(grid, torch.Tensor):
        import numpy as _np
        if isinstance(grid, _np.ndarray):
            grid = torch.from_numpy(grid).to(dev, dtype=torch.int32)

    with torch.no_grad():
        # Raw ViT output (a list of per-image tensors), matching extract_feature.
        vit_list = model.extract_feature(pv, grid)
        vit_raw = torch.cat(vit_list, dim=0) if isinstance(vit_list, (list, tuple)) \
            else vit_list
        out["vit_embeds_raw"] = vit_raw
        # Post-mlp1 projection — exactly what generate()/forward() splice in.
        vit_proj = model.mlp1(vit_raw)
        out["vit_embeds"] = vit_proj

        # Text-only embeds and the spliced result (replicating forward()).
        ids = input_ids.to(dev)
        text_embeds = model.language_model.get_input_embeddings()(ids)
        out["inputs_embeds_text_only"] = text_embeds
        B, N, C = text_embeds.shape
        flat = text_embeds.reshape(B * N, C).clone()
        flat_ids = ids.reshape(B * N)
        selected = (flat_ids == image_token_index)
        n_sel = int(selected.sum().item())
        flat[selected] = vit_proj[:n_sel].to(flat.dtype)
        spliced = flat.reshape(B, N, C)
        out["inputs_embeds"] = spliced
        out["n_image_tokens"] = n_sel
        out["image_token_positions"] = torch.nonzero(
            selected, as_tuple=False).reshape(-1)
    return out


# ---------------------------------------------------------------------------
# Box / point parsing — mirrors LocateAnythingWorker.parse_boxes exactly.
# ---------------------------------------------------------------------------
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
_POINT_RE = re.compile(r"<box><(\d+)><(\d+)></box>")


def parse_boxes(answer: str, w: int, h: int) -> List[Dict[str, float]]:
    boxes = []
    for m in _BOX_RE.finditer(answer):
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        boxes.append({"x1": x1 / 1000 * w, "y1": y1 / 1000 * h,
                      "x2": x2 / 1000 * w, "y2": y2 / 1000 * h})
    return boxes


def parse_points(answer: str, w: int, h: int) -> List[Dict[str, float]]:
    pts = []
    for m in _POINT_RE.finditer(answer):
        x, y = int(m.group(1)), int(m.group(2))
        pts.append({"x": x / 1000 * w, "y": y / 1000 * h})
    return pts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Dump golden-reference tensors from LocateAnything-3B.")
    ap.add_argument("--checkpoint", required=True,
                    help="LocateAnything-3B HF checkpoint dir (e.g. nvidia/LocateAnything-3B).")
    ap.add_argument("--image", required=True, help="Path to a sample RGB image.")
    ap.add_argument("--prompt", required=True,
                    help="Task prompt, e.g. 'Locate all the instances that "
                         "match the following description: the red mug.'")
    ap.add_argument("--out-dir", required=True,
                    help="Directory for .npy files + manifest.json.")
    ap.add_argument("--generation-mode", default="hybrid",
                    choices=("fast", "slow", "hybrid"),
                    help="MTP(fast) / AR(slow) / hybrid. Default hybrid.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--mtp-steps", default="0,1,2",
                    help="Comma-separated LM-forward indices to dump "
                         "mask/position_ids/input_ids for (0 = prefill).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (REQUIRED for stable golden vectors).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np, torch, transformers = _require_runtime_deps()
    from PIL import Image

    # Determinism. Greedy (temperature=0, do_sample=False) is required so the
    # golden vectors are reproducible; sampling would make them non-deterministic.
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.temperature != 0.0:
        print("[warn] temperature != 0 -> sampling path is non-deterministic; "
              "golden vectors will NOT be reproducible. Use --temperature 0.")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = Manifest()

    # Make the canonical package importable.
    eagle_root = os.environ.get(
        "EAGLE_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    for p in (eagle_root, os.path.join(eagle_root, "Embodied")):
        if p not in sys.path:
            sys.path.insert(0, p)

    print("== loading worker (canonical LocateAnythingWorker) ==")
    from locateanything_worker import LocateAnythingWorker
    worker = LocateAnythingWorker(args.checkpoint, device="cuda",
                                  dtype=torch.bfloat16)
    model, processor, tokenizer = worker.model, worker.processor, worker.tokenizer
    dev = worker.device

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    manifest.meta = {
        "checkpoint": os.path.abspath(args.checkpoint)
        if os.path.isdir(args.checkpoint) else args.checkpoint,
        "image": os.path.abspath(args.image),
        "prompt": args.prompt,
        "generation_mode": args.generation_mode,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "gpu_name": gpu_name,
        "model_dtype": str(worker.dtype),
        "warning": ("Cross-GPU bitwise identity is NOT guaranteed (SM arch, "
                    "cuBLAS/cuDNN heuristics, bf16 rounding). C++ parity gates "
                    "MUST use tolerances. Provenance: dtype + gpu_name above."),
    }

    # Config token ids (the C++ decode FSM needs these exact ids).
    cfg = model.config
    try:
        from eaglevl.utils.locany.generate_utils import get_token_ids_from_config
        token_ids = get_token_ids_from_config(cfg)
    except Exception:
        token_ids = {}
    manifest.extra["token_ids"] = {k: int(v) for k, v in token_ids.items()}
    manifest.extra["image_token_index"] = int(cfg.image_token_index)
    manifest.extra["n_future_tokens"] = 6  # canonical generate default

    # ---- Preprocess exactly like LocateAnythingWorker.predict ----
    print("== preprocessing ==")
    image = Image.open(args.image).convert("RGB")
    img_w, img_h = image.size
    manifest.extra["image_width"] = int(img_w)
    manifest.extra["image_height"] = int(img_h)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": args.prompt},
    ]}]
    text = processor.py_apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    manifest.extra["templated_text"] = text
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos,
                       return_tensors="pt").to(dev)

    pixel_values = inputs["pixel_values"].to(worker.dtype)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_grid_hws = inputs.get("image_grid_hws", None)

    _save(np, args.out_dir, "pixel_values", pixel_values, manifest,
          note="ViT input (post image-processor)")
    if image_grid_hws is not None:
        _save(np, args.out_dir, "image_grid_hws",
              np.asarray(image_grid_hws, dtype=np.int32), manifest,
              note="per-image (h,w) grid sizes")
    _save(np, args.out_dir, "input_ids", input_ids, manifest,
          note="tokenized prompt incl. <IMG_CONTEXT> placeholders")

    # ---- Vision features + splice (replicates the model's exact splice) ----
    print("== vision encode + splice ==")
    vs = capture_vision_and_splice(model, pixel_values, image_grid_hws,
                                   input_ids, torch)
    _save(np, args.out_dir, "vit_embeds_raw", vs["vit_embeds_raw"], manifest,
          note="raw vision_model output, pre-mlp1 (ViT-only parity)")
    _save(np, args.out_dir, "vit_embeds", vs["vit_embeds"], manifest,
          note="POST-mlp1 projected features actually spliced into the LM")
    _save(np, args.out_dir, "image_token_positions", vs["image_token_positions"],
          manifest, note="flat indices where input_ids == image_token_index")
    _save(np, args.out_dir, "inputs_embeds_text_only",
          vs["inputs_embeds_text_only"], manifest,
          note="text embeds before vision splice")
    _save(np, args.out_dir, "inputs_embeds", vs["inputs_embeds"], manifest,
          note="spliced inputs_embeds (prefill input to the LM)")
    manifest.extra["n_image_tokens"] = int(vs["n_image_tokens"])

    # ---- Generation via the canonical custom MTP/AR loop ----
    print(f"== generation (mode={args.generation_mode}) ==")
    recorder = LMForwardRecorder(torch).wrap(model.language_model)
    try:
        with torch.no_grad():
            answer = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_hws=image_grid_hws,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                generation_mode=args.generation_mode,
                temperature=args.temperature,
                do_sample=(args.temperature > 0.0),
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=False,
            )
    finally:
        recorder.restore()

    # The canonical generate returns the decoded string of NEW tokens.
    answer_str = answer[0] if isinstance(answer, tuple) else answer
    manifest.extra["answer"] = answer_str

    # Re-tokenize the answer to recover the generated token ids deterministically
    # (generate() does not return ids). prompt ids + generated ids = full seq.
    gen_ids = tokenizer(answer_str, add_special_tokens=False,
                        return_tensors="pt")["input_ids"]
    full_ids = torch.cat([input_ids.to(gen_ids.device), gen_ids], dim=1)
    _save(np, args.out_dir, "generated_token_ids", gen_ids, manifest,
          note="newly generated token ids (re-tokenized from answer string)")
    _save(np, args.out_dir, "output_token_ids", full_ids, manifest,
          note="prompt + generated token ids")
    _save(np, args.out_dir, "committed_token_ids", gen_ids.reshape(-1), manifest,
          note="flat committed token ids (== generated). Per-step commit counts "
               "in step_commit_counts.npy")
    manifest.extra["generated_token_id_note"] = (
        "Recovered by re-tokenizing the decoded answer; the canonical generate() "
        "returns a string. If tokenizer round-trip is lossy for special box/coord "
        "tokens, prefer the per-step LM-forward input_ids dumps for exact ids.")

    # ---- Per-step LM-forward records (MTP window mask / position_ids / kv) ----
    print("== per-step MTP/AR window records ==")
    n_steps = len(recorder.position_ids)
    manifest.extra["n_lm_forward_steps"] = n_steps
    if recorder.kv_len_after:
        _save(np, args.out_dir, "kv_cache_lengths",
              np.asarray(recorder.kv_len_after, dtype=np.int64), manifest,
              note="KV-cache length AFTER each LM forward")
    # commit counts per step (window length contributed); best-effort from input_ids
    commit_counts = []
    for iid in recorder.input_ids:
        commit_counts.append(int(iid.shape[1]) if iid is not None else 0)
    if commit_counts:
        _save(np, args.out_dir, "step_commit_counts",
              np.asarray(commit_counts, dtype=np.int64), manifest,
              note="window length fed at each LM forward (prefill is large)")

    want = [int(x) for x in args.mtp_steps.split(",") if x.strip() != ""]
    saved = []
    for s in want:
        if 0 <= s < n_steps:
            if recorder.attention_mask[s] is not None:
                _save(np, args.out_dir, f"step_{s}_mtp_mask",
                      recorder.attention_mask[s], manifest,
                      note=f"attention mask fed at LM forward {s}")
            if recorder.position_ids[s] is not None:
                _save(np, args.out_dir, f"step_{s}_position_ids",
                      recorder.position_ids[s], manifest,
                      note=f"position_ids fed at LM forward {s} "
                           "(MTP: last n_future_tokens get -1)")
            if recorder.input_ids[s] is not None:
                _save(np, args.out_dir, f"step_{s}_input_ids",
                      recorder.input_ids[s], manifest,
                      note=f"generated_with_mask window fed at LM forward {s}")
            saved.append(s)
    manifest.extra["saved_step_indices"] = saved
    if n_steps == 0:
        manifest.extra["step_capture_warning"] = (
            "No LM-forward steps captured. The wrapper on language_model.forward "
            "did not fire; verify generate() routes through model.language_model "
            "and not a compiled/cached callable.")

    # ---- Parsed boxes (PIXEL coords) ----
    print("== parse boxes ==")
    boxes = parse_boxes(answer_str, img_w, img_h)
    points = parse_points(answer_str, img_w, img_h)
    box_arr = np.asarray([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes],
                         dtype=np.float32) if boxes else \
        np.zeros((0, 4), dtype=np.float32)
    _save(np, args.out_dir, "parsed_boxes", box_arr, manifest,
          note="parsed boxes [n,4] xyxy in PIXEL coords (coords/1000*W,H)")
    manifest.extra["parsed_boxes"] = boxes
    manifest.extra["parsed_points"] = points

    # ---- Write manifest ----
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    print(f"\n== done == wrote {len(manifest.tensors)} tensors + manifest.json "
          f"to {os.path.abspath(args.out_dir)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
