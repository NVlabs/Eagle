# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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

import json
import os
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModel, AutoProcessor


LOCATE_ANYTHING_MODEL_TYPES = {"locateanything", "locate_anything"}


def apply_chat_template(processor, messages, prefer_py_template=False):
    if prefer_py_template and hasattr(processor, "py_apply_chat_template"):
        return processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if hasattr(processor, "py_apply_chat_template"):
        return processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    raise AttributeError("Processor does not provide a chat template API")


def process_vision_info(processor, messages):
    if hasattr(processor, "process_vision_info"):
        return processor.process_vision_info(messages)

    raise AttributeError("Processor does not provide process_vision_info")


def _maybe_to_device(value, device):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device)
    return value


def prepare_generation_inputs(processor_inputs, device, dtype=None):
    pixel_values = _maybe_to_device(processor_inputs.get("pixel_values"), device)
    if dtype is not None and torch.is_tensor(pixel_values) and pixel_values.is_floating_point():
        # The repository Eagle Embodied model does not cast images to its dtype.
        pixel_values = pixel_values.to(dtype=dtype)
    return {
        "input_ids": processor_inputs["input_ids"].to(device),
        "attention_mask": _maybe_to_device(processor_inputs.get("attention_mask"), device),
        "pixel_values": pixel_values,
        "image_grid_hws": _maybe_to_device(processor_inputs.get("image_grid_hws"), device),
    }


def get_model_type(config) -> str:
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = getattr(config, "architectures", None) or []
    if model_type in LOCATE_ANYTHING_MODEL_TYPES or any("locateanything" in str(x).lower() for x in architectures):
        return "locateanything"
    if model_type == "eagle_embodied" or any(
        "eagleembodied" in str(x).lower() for x in architectures
    ):
        return "eagle_embodied"
    return model_type or "generic"


def has_physical_atoms(model_path) -> bool:
    """Whether a local checkpoint is an Eagle Embodied target trained with physical atoms.

    These targets (and their VFlash drafters) run with the repository model.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return False
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    return config.get("model_type") == "eagle_embodied" and isinstance(
        config.get("physical_atom_config"), dict
    )


def _patch_qwen3_rope_compat(model_config):
    text_config = getattr(model_config, "text_config", None)
    rope_parameters = getattr(text_config, "rope_parameters", None)
    if not isinstance(rope_parameters, dict):
        return

    rope_theta = rope_parameters.get("rope_theta")
    if rope_theta is not None:
        text_config.rope_theta = rope_theta

    rope_type = rope_parameters.get("rope_type") or rope_parameters.get("type")
    if (
        rope_type
        and rope_type != "default"
        and getattr(text_config, "rope_scaling", None) is None
    ):
        text_config.rope_scaling = dict(rope_parameters)
        text_config.rope_scaling.setdefault("type", rope_type)


def _set_attention_implementation(model_config, attn_impl):
    for config in (
        model_config,
        getattr(model_config, "text_config", None),
        getattr(model_config, "vision_config", None),
    ):
        if config is not None:
            config._attn_implementation = attn_impl
            config._attn_implementation_internal = attn_impl


def _bind_checkpoint_tokenizer(model_config, tokenizer):
    """Require the tokenizer the checkpoint was exported with and use its image token."""
    from eaglevl.train.constants import IMG_CONTEXT_TOKEN

    expected = int(model_config.text_config.vocab_size)
    actual = len(tokenizer)
    if actual != expected:
        raise ValueError(
            "checkpoint tokenizer/model vocabulary mismatch: "
            f"tokenizer={actual}, config.text_config.vocab_size={expected}. "
            "Export a self-contained target checkpoint before evaluation."
        )
    model_config.image_token_index = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)


def load_model_and_processor(
    model_path: str,
    device="cuda",
    dtype=torch.bfloat16,
    use_fast: bool = True,
    local_code: bool = False,
    local_processor: bool = False,
    attn_implementation: Optional[str] = None,
    training_runtime: bool = False,
    draft_model_path: Optional[str] = None,
):
    """Load a checkpoint and its processor for inference; returns ``(model, processor)``.

    By default the checkpoint's remote code builds both (LocateAnything and
    Eagle Embodied checkpoints).

    - ``local_code`` builds the repository ``EagleEmbodiedForConditionalGeneration``
      instead; VFlash drafters (``draft_model_path``) attach only to it.
    - ``local_processor`` builds the repository processor around the checkpoint
      tokenizer, loaded with ``add_eos_token=False``. ``use_fast`` selects the
      fast or the slow tokenizer in both cases.
    - ``attn_implementation`` is written into the configuration of the
      repository model before it is built.
    - ``training_runtime`` applies the trainer's process settings, including the
      Liger kernels and the PIL image limits, before the target is built.
    """
    if draft_model_path and not local_code:
        raise ValueError("VFlash drafters attach to the repository model; use local_code=True")
    if (attn_implementation is not None or local_processor) and not local_code:
        raise ValueError("attn_implementation and local_processor require local_code=True")
    if training_runtime:
        # The drafter module binds the Hugging Face Qwen3 MLP and RMSNorm at
        # import, so it keeps them after the Liger kernels replace them for the target.
        import eaglevl.model.vflash.modeling_dflash  # noqa: F401
        from dotenv import load_dotenv
        from packaging import version
        from PIL import Image, ImageFile, PngImagePlugin

        from eaglevl.patch import replace_liger_fused_ops

        # The import-time settings of the Eagle Embodied trainer.
        load_dotenv()
        if version.parse(torch.__version__) >= version.parse("2.4.0"):
            import numpy as np

            torch.serialization.add_safe_globals(
                [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))])
        replace_liger_fused_ops()
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        PngImagePlugin.MAX_TEXT_CHUNK = 1024 * 2 ** 20
        warnings.filterwarnings("ignore")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if local_processor:
        from transformers import AutoTokenizer

        from eaglevl.train.tools import build_local_processor

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, add_eos_token=False, trust_remote_code=True, use_fast=use_fast
        )
        processor = build_local_processor(model_path, tokenizer)
    else:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=use_fast,
        )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"

    if not local_code:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device).eval()
        return model, processor

    from eaglevl.model.eagle_embodied.configuration_eagle_embodied import EagleEmbodiedConfig
    from eaglevl.model.eagle_embodied.modeling_eagle_embodied import (
        EagleEmbodiedForConditionalGeneration,
    )

    model_kwargs = {"torch_dtype": dtype}
    if attn_implementation is not None or local_processor:
        config = EagleEmbodiedConfig.from_pretrained(model_path)
        _patch_qwen3_rope_compat(config)
        if attn_implementation is not None:
            _set_attention_implementation(config, attn_implementation)
            model_kwargs["attn_implementation"] = attn_implementation
        if local_processor:
            _bind_checkpoint_tokenizer(config, tokenizer)
        model_kwargs["config"] = config
    model = EagleEmbodiedForConditionalGeneration.from_pretrained(model_path, **model_kwargs).eval()
    model = model.to(device)
    if draft_model_path:
        # The drafter must also be bound to this tokenizer's vocabulary.
        model.load_vflash_draft(draft_model_path, tokenizer=tokenizer)
    return model, processor


def build_generate_kwargs(
    prepared_inputs,
    processor,
    generation_mode,
    max_new_tokens,
    include_eos_token=False,
    model_type=None,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    verbose=False,
):
    """Without ``model_type`` return the LocateAnything evaluation kwargs below.

    ``EagleEmbodiedInferenceAdapter`` passes the checkpoint family from
    ``get_model_type``; the sampling arguments apply only to that path.
    """
    if model_type is not None:
        return _build_family_generate_kwargs(
            prepared_inputs, processor, generation_mode, max_new_tokens, include_eos_token,
            model_type, temperature, top_p, repetition_penalty, verbose,
        )

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None and hasattr(processor, "batch_decode"):
        tokenizer = processor

    generate_kwargs = dict(
        pixel_values=prepared_inputs["pixel_values"],
        input_ids=prepared_inputs["input_ids"],
        attention_mask=prepared_inputs["attention_mask"],
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        generation_mode=generation_mode,
    )

    if prepared_inputs["image_grid_hws"] is not None:
        generate_kwargs["image_grid_hws"] = prepared_inputs["image_grid_hws"]

    if include_eos_token and tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

    if generation_mode in ("fast", "hybrid"):
        generate_kwargs["n_future_tokens"] = 6

    return generate_kwargs


def _build_family_generate_kwargs(
    prepared_inputs,
    processor,
    generation_mode,
    max_new_tokens,
    include_eos_token,
    model_type,
    temperature,
    top_p,
    repetition_penalty,
    verbose,
):
    """Build model-family-aware kwargs without leaking LocateAnything-only args."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None and hasattr(processor, "batch_decode"):
        tokenizer = processor

    do_sample = temperature is not None and temperature > 0
    generate_kwargs = {
        "pixel_values": prepared_inputs["pixel_values"],
        "input_ids": prepared_inputs["input_ids"],
        "attention_mask": prepared_inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
    }
    if do_sample:
        generate_kwargs.update(temperature=temperature, top_p=top_p)

    if prepared_inputs["image_grid_hws"] is not None:
        generate_kwargs["image_grid_hws"] = prepared_inputs["image_grid_hws"]

    if include_eos_token and tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

    if model_type == "locateanything":
        generate_kwargs.update(
            tokenizer=tokenizer,
            generation_mode=generation_mode,
            verbose=verbose,
        )
        if generation_mode in ("fast", "hybrid"):
            generate_kwargs["n_future_tokens"] = 6
    elif model_type == "eagle_embodied" and generation_mode == "vflash":
        if do_sample:
            raise ValueError("VFlash decoding keeps the Eagle target greedy; use temperature=0")
        from evaluation.vflash.generation import vflash_generate_kwargs

        generate_kwargs.update(vflash_generate_kwargs(tokenizer, max_new_tokens))

    return generate_kwargs


def split_generation_output(raw_output) -> Tuple[Any, Optional[Any], Optional[Any]]:
    if not isinstance(raw_output, tuple):
        return raw_output, None, None
    payload = raw_output[0] if raw_output else ""
    history = raw_output[1] if len(raw_output) > 1 else None
    stats = raw_output[2] if len(raw_output) > 2 else None
    return payload, history, stats


def decode_generation_output(raw_output, input_ids, processor, strip_prompt=None, strip_end_tokens=False):
    """Decode the first generated sequence.

    ``strip_prompt=None`` drops the first ``len(input_ids)`` tokens of a long
    enough token output; ``True`` drops them only when they echo
    ``input_ids``; ``False`` keeps them. ``strip_end_tokens`` removes the
    ``<|im_end|>`` and ``<|endoftext|>`` markers.
    """
    if strip_end_tokens:
        text = decode_generation_output(raw_output, input_ids, processor, strip_prompt)
        return text.replace("<|im_end|>", "").replace("<|endoftext|>", "")

    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]

    if isinstance(raw_output, str):
        return raw_output

    if isinstance(raw_output, list):
        if not raw_output:
            return ""
        if isinstance(raw_output[0], str):
            return raw_output[0]

    if torch.is_tensor(raw_output):
        generated_ids = raw_output
        if (
            strip_prompt is not False
            and generated_ids.ndim == 2
            and input_ids is not None
            and generated_ids.shape[1] >= input_ids.shape[1]
            and (
                strip_prompt is None
                or torch.equal(generated_ids[:1, : input_ids.shape[1]].cpu(), input_ids[:1].cpu())
            )
        ):
            generated_ids = generated_ids[:, input_ids.shape[1]:]
        generated_ids = generated_ids.detach().cpu()

        if hasattr(processor, "post_process_image_text_to_text"):
            decoded = processor.post_process_image_text_to_text(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if isinstance(decoded, list):
                return decoded[0]
            return decoded

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None and hasattr(processor, "batch_decode"):
            tokenizer = processor
        decoded = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if isinstance(decoded, list) else decoded

    return str(raw_output)


class EagleEmbodiedInferenceAdapter:
    """Load one checkpoint and normalize its processor/generation interface."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype=torch.bfloat16,
        generation_mode: str = "hybrid",
        use_fast: bool = True,
        draft_model_path: Optional[str] = None,
        bbox_hybrid=None,
    ):
        """``generation_mode="vflash"`` decodes an Eagle target through the
        ``draft_model_path`` drafter; ``bbox_hybrid`` (a
        ``eaglevl.model.vflash.bbox_hybrid.BBoxHybridConfig``) selects the bbox
        hybrid decoder for visual-primitive box questions.
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.generation_mode = generation_mode
        self.bbox_hybrid = bbox_hybrid
        if generation_mode == "vflash" and not draft_model_path:
            raise ValueError("VFlash decoding requires draft_model_path")
        # VFlash targets carry physical atoms and run with the repository model.
        self.physical_atoms = has_physical_atoms(model_path)
        self.model, self.processor = load_model_and_processor(
            model_path,
            device=device,
            dtype=dtype,
            use_fast=use_fast,
            local_code=self.physical_atoms,
            draft_model_path=draft_model_path if generation_mode == "vflash" else None,
        )
        if generation_mode == "vflash":
            # The benchmark protocol casts the whole drafter to the model dtype,
            # which also rounds its float32 rotary inverse frequencies.
            self.model.vflash_draft.to(dtype=dtype)
        self.model_type = get_model_type(self.model.config)

    @torch.inference_mode()
    def generate(
        self,
        messages,
        max_new_tokens: int = 4096,
        generation_mode: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        include_eos_token: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        text_list = [apply_chat_template(self.processor, messages, prefer_py_template=True)]
        image_inputs, video_inputs = process_vision_info(self.processor, messages)
        processor_inputs = self.processor(
            text=text_list,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        prepared_inputs = prepare_generation_inputs(
            processor_inputs,
            self.device,
            self.dtype if self.physical_atoms else None,
        )
        generate_kwargs = build_generate_kwargs(
            prepared_inputs,
            self.processor,
            generation_mode=generation_mode or self.generation_mode,
            max_new_tokens=max_new_tokens,
            include_eos_token=include_eos_token,
            model_type=self.model_type,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            verbose=verbose,
        )
        if self.bbox_hybrid is not None and generate_kwargs.get("generation_mode") == "vflash":
            from evaluation.vflash.generation import bbox_hybrid_prefix_ids

            generate_kwargs.update(
                bbox_hybrid=self.bbox_hybrid,
                initial_prefix_token_ids=bbox_hybrid_prefix_ids(messages, self.processor.tokenizer),
            )
        raw_output = self.model.generate(**generate_kwargs)
        payload, history, stats = split_generation_output(raw_output)
        answer = decode_generation_output(
            payload,
            prepared_inputs["input_ids"],
            self.processor,
            strip_prompt=False,
        )
        result = {"answer": answer}
        if history is not None:
            result["history"] = history
        if stats is not None:
            result["stats"] = stats
        return result


class BenchmarkInferenceWorker:
    """Detection prompt adapter for Eagle Embodied (VFlash) checkpoints."""

    def __init__(
        self,
        model_path,
        device="cuda",
        generation_mode="hybrid",
        include_eos_token=False,
        draft_model_path=None,
        bbox_hybrid=None,
    ):
        self.include_eos_token = include_eos_token
        self.adapter = EagleEmbodiedInferenceAdapter(
            model_path,
            device=device,
            generation_mode=generation_mode,
            draft_model_path=draft_model_path,
            bbox_hybrid=bbox_hybrid,
        )

    def _generate_answer(self, messages, max_new_tokens):
        kwargs = dict(max_new_tokens=max_new_tokens, include_eos_token=self.include_eos_token)
        if self.adapter.physical_atoms:
            # VFlash benchmarks decode greedily; the adapter defaults sample.
            kwargs.update(temperature=0.0, top_p=1.0, repetition_penalty=1.0)
        return self.adapter.generate(messages, **kwargs)["answer"]

    def _image_text(self, question):
        # VFlash training puts a newline between the image and the text.
        return ("\n" if self.adapter.physical_atoms else "") + question

    @torch.inference_mode()
    def generate(self, image, query, max_new_tokens=4096, question_override=None):
        if isinstance(query, str):
            question = query
            return_question = False
        else:
            categories = [str(category) for category in query]
            if question_override is not None:
                question = question_override + ", ".join(categories) + "."
            else:
                category_text = "</c>".join(categories)
                question = (
                    "Locate all the instances that matches the following "
                    f"description: {category_text}."
                )
            return_question = True

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self._image_text(question)},
            ],
        }]
        answer = self._generate_answer(messages, max_new_tokens)
        if return_question:
            return answer, question
        return answer

    @torch.inference_mode()
    def generate_multiturn(self, image, questions, max_new_tokens=4096):
        """Ask ``questions`` in one conversation; the image is sent with the first.

        Each answer is fed back as the assistant turn without its end marker.
        """
        messages, answers = [], []
        for index, question in enumerate(questions):
            content = [{"type": "text", "text": str(question)}]
            if index == 0:
                content = [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._image_text(str(question))},
                ]
            messages.append({"role": "user", "content": content})
            answer = self._generate_answer(messages, max_new_tokens)
            answers.append(answer)
            history = answer.rstrip()
            if history.endswith("<|im_end|>"):
                history = history[: -len("<|im_end|>")].rstrip()
            messages.append({"role": "assistant", "content": [{"type": "text", "text": history}]})
        return answers
