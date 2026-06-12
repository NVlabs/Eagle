# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
locateanything_worker.py - A reusable worker for LocateAnything inference.
"""
import os
import re
from typing import Optional, Union

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


class LocateAnythingWorker:
    """Stateful worker that loads the model once and serves perception queries."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype=torch.bfloat16,
        use_batch_runtime: bool = False,
        attn: str = "la_flash",
        vision_attn: str = "auto",
        scheduler: str = "pipeline",
        group_size: int = 0,
        strict_attn: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.use_batch_runtime = use_batch_runtime
        self.scheduler = scheduler
        self.group_size = group_size

        if use_batch_runtime:
            self._init_batch_runtime(
                model_path=model_path,
                attn=attn,
                vision_attn=vision_attn,
                scheduler=scheduler,
                group_size=group_size,
                strict_attn=strict_attn,
            )
            return

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

    def _init_batch_runtime(
        self,
        model_path: str,
        attn: str,
        vision_attn: str,
        scheduler: str,
        group_size: int,
        strict_attn: bool,
    ) -> None:
        """Initialize the optional HF-release batch runtime.

        The batch runtime is shipped with the Hugging Face model repository as
        `batch_utils` and `kernel_utils`. It uses FlashAttention varlen sparse
        range plans for `attn="la_flash"` and does not build a local CUDA
        extension.
        """
        os.environ["LA_FLASH_MODEL"] = model_path
        os.environ["LA_FLASH_ATTN"] = attn
        os.environ["LA_FLASH_VISION_ATTN"] = vision_attn
        os.environ["LA_FLASH_HYBRID_SCHEDULER"] = scheduler
        os.environ["LA_FLASH_HYBRID_GROUP_SIZE"] = str(group_size)
        if strict_attn:
            os.environ["LA_FLASH_STRICT_ATTN"] = "1"

        try:
            from batch_utils import generate_batch_hybrid, get_last_hybrid_stats, load
        except ImportError as exc:
            raise ImportError(
                "Batch inference requires the Hugging Face release files "
                "`batch_utils/` and `kernel_utils/` on PYTHONPATH. Download "
                "nvidia/LocateAnything-3B and run from that directory, or add "
                "the model directory to PYTHONPATH."
            ) from exc

        self._batch_generate = generate_batch_hybrid
        self._batch_stats = get_last_hybrid_stats
        self.tokenizer, self.processor, self.model = load()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        repetition_penalty: float = 1.1,
        verbose: bool = True,
    ) -> dict:
        """
        Run a single perception query.

        Args:
            image: PIL Image (RGB).
            question: The task prompt (see supported prompts below).
            generation_mode: "fast" (MTP) | "slow" (NTP) | "hybrid".
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling cutoff; 0 disables top-k.
            repetition_penalty: Repetition penalty used by the model sampler.
            verbose: If True, return timing statistics.

        Returns:
            dict with keys: "answer", "stats" (optional), "history" (optional).
        """
        if self.use_batch_runtime:
            return self.predict_batch(
                [(image, question)],
                generation_mode=generation_mode,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                verbose=verbose,
            )[0]

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            verbose=verbose,
        )

        result = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result

    @torch.no_grad()
    def predict_batch(
        self,
        requests: list[Union[tuple[Image.Image, str], dict]],
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        repetition_penalty: float = 1.1,
        scheduler: Optional[str] = None,
        group_size: Optional[int] = None,
        verbose: bool = True,
    ) -> list[dict]:
        """Run a batch of `(image, question)` perception queries.

        When `use_batch_runtime=True`, this uses the released `batch_utils`
        hybrid scheduler. Otherwise it falls back to serial calls to
        `predict()` for compatibility.
        """
        pairs = []
        for item in requests:
            if isinstance(item, dict):
                image = item["image"]
                question = item.get("question", item.get("prompt"))
                if question is None:
                    raise ValueError("batch request dict must contain `question` or `prompt`")
            else:
                image, question = item
            pairs.append((image, question))

        if not self.use_batch_runtime:
            return [
                self.predict(
                    image,
                    question,
                    generation_mode=generation_mode,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    verbose=verbose,
                )
                for image, question in pairs
            ]

        if generation_mode != "hybrid":
            raise ValueError("batch runtime currently supports generation_mode='hybrid'")

        answers = self._batch_generate(
            pairs,
            temperature=temperature,
            top_p=None if top_p < 0 else top_p,
            top_k=None if top_k <= 0 else top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            scheduler=self.scheduler if scheduler is None else scheduler,
            group_size=self.group_size if group_size is None else group_size,
        )
        stats = self._batch_stats() if verbose else None
        results = []
        for answer in answers:
            row = {"answer": answer}
            if stats is not None:
                row["stats"] = stats
            results.append(row)
        return results

    # ---- Convenience methods for each task ----

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        """Object detection / document layout analysis."""
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def detect_batch(self, requests: list[tuple[Image.Image, Union[list[str], str]]], **kwargs) -> list[dict]:
        """Batch object detection.

        Args:
            requests: list of `(image, categories)` pairs. `categories` can be
                either a list of labels or a pre-joined `"</c>"` string.
        """
        pairs = []
        for image, categories in requests:
            cats = categories if isinstance(categories, str) else "</c>".join(categories)
            prompt = f"Locate all the instances that matches the following description: {cats}."
            pairs.append((image, prompt))
        return self.predict_batch(pairs, **kwargs)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — single instance."""
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — multiple instances."""
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_text(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Text grounding."""
        prompt = f"Please locate the text referred as {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_text(self, image: Image.Image, **kwargs) -> dict:
        """Scene text detection."""
        prompt = "Detect all the text in box format."
        return self.predict(image, prompt, **kwargs)

    def ground_gui(self, image: Image.Image, phrase: str, output_type: str = "box", **kwargs) -> dict:
        """GUI grounding (box or point)."""
        if output_type == "point":
            prompt = f"Point to: {phrase}."
        else:
            prompt = f"Locate the region that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Pointing."""
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    # ---- Utility: parse model output ----

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate bounding boxes.

        Coordinates in model output are normalized integers in [0, 1000].
        """
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({
                "x1": x1 / 1000 * image_width,
                "y1": y1 / 1000 * image_height,
                "x2": x2 / 1000 * image_width,
                "y2": y2 / 1000 * image_height,
            })
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate points."""
        points = []
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append({
                "x": x / 1000 * image_width,
                "y": y / 1000 * image_height,
            })
        return points


# --------------- Usage Example ---------------
if __name__ == "__main__":
    worker = LocateAnythingWorker("nvidia/LocateAnything-3B")
    img = Image.open("example.jpg").convert("RGB")

    # Object Detection
    result = worker.detect(img, ["person", "car", "bicycle"])
    print("Detection:", result["answer"])

    # Phrase Grounding (multiple)
    result = worker.ground_multi(img, "people wearing red shirts")
    print("Grounding:", result["answer"])

    # Scene Text Detection
    result = worker.detect_text(img)
    print("Text Detection:", result["answer"])

    # Pointing
    result = worker.point(img, "the traffic light")
    print("Pointing:", result["answer"])

    # GUI Grounding (point)
    result = worker.ground_gui(img, "the search button", output_type="point")
    print("GUI Point:", result["answer"])

    # Parse structured output
    w, h = img.size
    boxes = LocateAnythingWorker.parse_boxes(result["answer"], w, h)
    print("Parsed boxes:", boxes)

    # Optional batch runtime from the Hugging Face release. Run from the
    # downloaded model directory, or add that directory to PYTHONPATH.
    batch_worker = LocateAnythingWorker(
        "nvidia/LocateAnything-3B",
        use_batch_runtime=True,
        attn="la_flash",
        scheduler="pipeline",
    )
    batch_results = batch_worker.detect_batch([
        (img, ["person", "car"]),
        (img, ["traffic light"]),
    ])
    print("Batch Detection:", [row["answer"] for row in batch_results])
