# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# The "Recipe-driven photometric augmentation" section is partly adopted from the
# ImageNet-C corruption code of the robustness project
# (https://github.com/hendrycks/robustness), licensed under the Apache License,
# Version 2.0, as redistributed in LIBERO-plus (https://github.com/sylvestf/LIBERO-plus).
#
# Modifications © 2026 NVIDIA CORPORATION & AFFILIATES, licensed under
# the Apache License, Version 2.0.
#
# --------------------------------------------------------
# ImageNet-C corruptions (robustness), Dan Hendrycks
# Licensed under the Apache License, Version 2.0
# --------------------------------------------------------

# -*- coding: utf-8 -*-
"""
Image augmentation module for training-time data augmentation.
"""

import math
import random
from dataclasses import dataclass
from PIL import Image
from typing import Union, List
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch


def resize_image_keep_aspect_ratio(
    image: Image.Image, 
    target_long_edge: int
) -> Image.Image:
    """
    Resize the image to the specified long edge length while preserving aspect ratio.
    
    Args:
        image: PIL Image
        target_long_edge: Target long edge length
    
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    long_edge = max(width, height)
    
    if long_edge == target_long_edge:
        return image
    
    scale = target_long_edge / long_edge
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return image.resize((new_width, new_height), Image.LANCZOS)


def apply_resize_augmentation(
    image: Image.Image,
    data_augment: bool = True,
    min_long_edge: int = 640,
    max_long_edge: int = 2048,
    augment_prob: float = 0.5
) -> Image.Image:
    """
    Apply resize augmentation.
    
    Rules:
    - If data_augment=True, no processing is done and the original image is returned.
    - If data_augment=False, with augment_prob probability the original image is kept,
      and with (1-augment_prob) probability the long edge is resized to a random value
      in [min_long_edge, max_long_edge].
    
    Args:
        image: PIL Image
        data_augment: The data_augment value from dataset config
        min_long_edge: Minimum long edge length
        max_long_edge: Maximum long edge length
        augment_prob: Probability of keeping the original image (default 50%)
    
    Returns:
        Processed PIL Image
    """
    if not data_augment:
        return image
    
    if random.random() < augment_prob:
        return image
    
    width, height = image.size
    current_long_edge = max(width, height)
    
    target_long_edge = random.randint(min_long_edge, max_long_edge)
    
    if target_long_edge == current_long_edge:
        return image
    
    return resize_image_keep_aspect_ratio(image, target_long_edge)


def apply_resize_augmentation_to_list(
    images: List[Image.Image],
    data_augment: bool = True,
    min_long_edge: int = 640,
    max_long_edge: int = 2048,
    augment_prob: float = 0.5
) -> List[Image.Image]:
    """
    Apply resize augmentation to a list of images.
    
    Note: Each image independently decides whether to augment and the target size.
    
    Args:
        images: List of PIL Images
        data_augment: The data_augment value from dataset config
        min_long_edge: Minimum long edge length
        max_long_edge: Maximum long edge length
        augment_prob: Probability of keeping the original image
    
    Returns:
        List of processed PIL Images
    """
    return [
        apply_resize_augmentation(
            img, 
            data_augment=data_augment,
            min_long_edge=min_long_edge,
            max_long_edge=max_long_edge,
            augment_prob=augment_prob
        )
        for img in images
    ]


# =========================================================================== #
# Recipe-driven photometric augmentation
# =========================================================================== #
# Online photometric augmentation for multi-view training samples.
#
# Reproduces the LIBERO-plus "Sensor Noise" family -- motion blur, gaussian blur,
# zoom blur, fog, glass blur, each with the 10 severity tables copied from
# ``LIBERO-plus/libero/libero/envs/env_wrapper.py`` -- plus a ``light`` op
# (brightness / contrast / gamma / white-balance jitter).  Everything is numpy +
# cv2 so the training container needs no ImageMagick / skimage / scipy.
#
# Recipe usage (any JSONL entry; absent means no augmentation)::
#
#     "visual_aug": {
#         "prob": 0.3,                 # per-sample chance that ONE op is applied
#         "views": [0],                # image / video indexes to corrupt (0 = primary camera)
#         "ops": ["motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur", "light"],
#         "severity": [1, 10],         # inclusive range, uniform
#         "zoom_max_severity": 5       # zoom blur drags content towards the centre; cap it
#     }
#
# Semantics
# ---------
# * One coin flip per sample.  On a hit, one op and one severity are drawn and
#   applied to every listed view index: the PIL image at that index and/or the
#   video clip at that index (every frame of the clip).  Per-frame randomness
#   (blur angle, glass shuffle, fog map) is redrawn for each frame; the ``light``
#   parameters stay fixed for the whole clip so a clip does not flicker.
# * Spatial constants in LIBERO-plus are tuned for 256x256 frames.  They are
#   scaled by ``max(1, min(H, W) / 256)`` so a 640x480 frame gets the same
#   relative blur; LIBERO frames (256) use the exact LIBERO-plus numbers.
# * Randomness comes only from the ``random.Random``/``random`` instance handed
#   to :meth:`VisualAugmenter.draw`; the trainer seeds it per sample index, so
#   augmentation is reproducible across resumes.
# * Geometry is untouched (blur / fog / glass are pixel-local), so 2D trajectory
#   labels stay valid.  Zoom blur ghosts content radially, hence the cap.

# --------------------------------------------------------------------------- #
# Severity tables (verbatim from LIBERO-plus env_wrapper.py, 10 levels each)  #
# --------------------------------------------------------------------------- #
MOTION_BLUR_C = [
    (5, 2),
    (8, 3),
    (10, 4),
    (12, 5),
    (15, 6),
    (18, 8),
    (20, 10),
    (25, 12),
    (30, 15),
    (35, 20),
]
GAUSSIAN_BLUR_C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ZOOM_BLUR_C = [
    np.arange(1, 1.11, 0.01),
    np.arange(1, 1.16, 0.01),
    np.arange(1, 1.21, 0.02),
    np.arange(1, 1.26, 0.02),
    np.arange(1, 1.31, 0.03),
    np.arange(1, 1.36, 0.01),
    np.arange(1, 1.41, 0.01),
    np.arange(1, 1.46, 0.02),
    np.arange(1, 1.51, 0.02),
    np.arange(1, 1.56, 0.03),
]
FOG_C = [
    (0.5, 3),
    (1.0, 2.8),
    (1.5, 2.5),
    (2.0, 2.2),
    (2.5, 2.0),
    (3.0, 1.8),
    (3.5, 1.6),
    (4.0, 1.5),
    (4.5, 1.4),
    (5.0, 1.3),
]
GLASS_BLUR_C = [  # (sigma, max_delta, iterations)
    (0.5, 1, 3),
    (0.7, 1, 3),
    (0.9, 2, 3),
    (1.0, 2, 2),
    (1.1, 3, 2),
    (1.3, 3, 2),
    (1.5, 4, 2),
    (1.8, 4, 2),
    (2.2, 5, 1),
    (2.5, 5, 1),
]

ALL_OPS = ("motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur", "light")
DEFAULT_CONFIG: Dict[str, Any] = {
    "prob": 0.3,
    "views": [0],
    "ops": list(ALL_OPS),
    "severity": [1, 10],
    "zoom_max_severity": 5,
}
REFERENCE_SHORT_EDGE = 256.0


def validate_visual_aug_config(cfg: Any, where: str = "recipe") -> Dict[str, Any]:
    """Return a normalised config dict or raise ``ValueError``."""
    if cfg is None or cfg is False:
        return None
    if cfg is True:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f'{where}: "visual_aug" must be a JSON object, got {type(cfg).__name__}')
    unknown = sorted(set(cfg) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(
            f'{where}: "visual_aug" has unknown fields {unknown}; allowed {sorted(DEFAULT_CONFIG)}'
        )
    out = dict(DEFAULT_CONFIG)
    out.update(cfg)
    prob = out["prob"]
    if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not (0.0 <= float(prob) <= 1.0):
        raise ValueError(f'{where}: "visual_aug.prob" must be a number in [0, 1], got {prob!r}')
    out["prob"] = float(prob)
    views = out["views"]
    if views != "all" and (
        not isinstance(views, (list, tuple))
        or not views
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in views)
    ):
        raise ValueError(
            f'{where}: "visual_aug.views" must be a non-empty list of non-negative ints, got {views!r}'
        )
    out["views"] = "all" if views == "all" else [int(v) for v in views]
    ops = out["ops"]
    if not isinstance(ops, (list, tuple)) or not ops or any(o not in ALL_OPS for o in ops):
        raise ValueError(f'{where}: "visual_aug.ops" must be a non-empty subset of {ALL_OPS}, got {ops!r}')
    out["ops"] = [str(o) for o in ops]
    sev = out["severity"]
    if (
        not isinstance(sev, (list, tuple))
        or len(sev) != 2
        or any(isinstance(s, bool) or not isinstance(s, int) for s in sev)
        or not (1 <= sev[0] <= sev[1] <= 10)
    ):
        raise ValueError(
            f'{where}: "visual_aug.severity" must be [lo, hi] with 1 <= lo <= hi <= 10, got {sev!r}'
        )
    out["severity"] = [int(sev[0]), int(sev[1])]
    zcap = out["zoom_max_severity"]
    if isinstance(zcap, bool) or not isinstance(zcap, int) or not (1 <= zcap <= 10):
        raise ValueError(f'{where}: "visual_aug.zoom_max_severity" must be an int in [1, 10], got {zcap!r}')
    return out


# --------------------------------------------------------------------------- #
# Ops.  All take/return float32 RGB arrays in [0, 1] of shape (H, W, 3).       #
# --------------------------------------------------------------------------- #
def _spatial_scale(h: int, w: int) -> float:
    return max(1.0, min(h, w) / REFERENCE_SHORT_EDGE)


def _gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    return cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def motion_blur(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """One-sided directional blur (ImageMagick MotionBlurImage analogue)."""
    h, w = x.shape[:2]
    scale = _spatial_scale(h, w)
    radius, sigma = MOTION_BLUR_C[severity - 1]
    radius, sigma = radius * scale, sigma * scale
    angle = math.radians(float(rng.uniform(-45, 45)))
    k = max(1, int(math.ceil(radius)))
    size = 2 * k + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    offsets = np.arange(0, k + 1, dtype=np.float32)
    weights = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    weights /= weights.sum()
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    for off, wgt in zip(offsets, weights):
        dx = int(round(off * cos_a))
        dy = int(round(off * sin_a))
        kernel[k + dy, k + dx] += wgt
    return cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REFLECT)


def gaussian_blur(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    h, w = x.shape[:2]
    sigma = GAUSSIAN_BLUR_C[severity - 1] * _spatial_scale(h, w)
    return np.clip(_gaussian(x, sigma), 0, 1)


def _center_zoom(img: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Crop the centre 1/zoom of the image and resize it back (non-square safe)."""
    h, w = img.shape[:2]
    ch = max(1, int(math.ceil(h / zoom_factor)))
    cw = max(1, int(math.ceil(w / zoom_factor)))
    top = (h - ch) // 2
    left = (w - cw) // 2
    crop = img[top : top + ch, left : left + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def zoom_blur(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    factors = ZOOM_BLUR_C[severity - 1]
    out = np.zeros_like(x)
    for z in factors:
        out += _center_zoom(x, float(z))
    return np.clip((x + out) / (len(factors) + 1), 0, 1)


def plasma_fractal(rng: np.random.Generator, mapsize: int = 256, wibbledecay: float = 3.0) -> np.ndarray:
    """Diamond-square heightmap in [0, 1] (port of the LIBERO-plus helper with an explicit RNG)."""
    assert mapsize & (mapsize - 1) == 0, "mapsize must be a power of two"
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100.0

    def wibbledmean(array):
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(
            squareaccum
        )

    def filldiamonds():
        drgrid = maparray[stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(ldrsum + lulsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(tdrsum + tulsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return (maparray / max(maparray.max(), 1e-12)).astype(np.float32)


def fog(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    h, w = x.shape[:2]
    c = FOG_C[severity - 1]
    max_val = float(x.max())
    frac = plasma_fractal(rng, mapsize=256, wibbledecay=c[1])
    frac = cv2.resize(frac, (w, h), interpolation=cv2.INTER_LINEAR)  # same fog structure at any resolution
    out = x + c[0] * frac[..., None]
    return np.clip(out * max_val / (max_val + c[0]), 0, 1)


def glass_blur(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian blur, local pixel shuffle (vectorised gather), gaussian blur."""
    h, w = x.shape[:2]
    scale = _spatial_scale(h, w)
    sigma, max_delta, iterations = GLASS_BLUR_C[severity - 1]
    sigma *= scale
    d = max(1, int(round(max_delta * scale)))
    out = _gaussian(x, sigma)
    grid_h, grid_w = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for _ in range(int(iterations)):
        dy = rng.integers(-d, d + 1, size=(h, w))
        dx = rng.integers(-d, d + 1, size=(h, w))
        src_h = np.clip(grid_h + dy, 0, h - 1)
        src_w = np.clip(grid_w + dx, 0, w - 1)
        out = out[src_h, src_w]
    return np.clip(_gaussian(out, sigma), 0, 1)


def draw_light_params(severity: int, rng: np.random.Generator) -> Dict[str, float]:
    """Brightness / contrast / gamma / white balance, amplitude grows with severity."""
    s = severity / 10.0
    return {
        "brightness": float(rng.uniform(1 - 0.5 * s, 1 + 0.5 * s)),
        "contrast": float(rng.uniform(1 - 0.4 * s, 1 + 0.4 * s)),
        "gamma": float(math.exp(rng.uniform(-0.6 * s, 0.6 * s))),
        "gain_r": float(rng.uniform(1 - 0.25 * s, 1 + 0.25 * s)),
        "gain_b": float(rng.uniform(1 - 0.25 * s, 1 + 0.25 * s)),
    }


def light(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    gains = np.array([params["gain_r"], 1.0, params["gain_b"]], dtype=np.float32)
    out = np.clip(x * gains, 0, 1)
    out = np.power(out, params["gamma"], dtype=np.float32)
    out = (out - 0.5) * params["contrast"] + 0.5
    out = out * params["brightness"]
    return np.clip(out, 0, 1)


NOISE_OPS = {
    "motion_blur": motion_blur,
    "gaussian_blur": gaussian_blur,
    "zoom_blur": zoom_blur,
    "fog": fog,
    "glass_blur": glass_blur,
}


# --------------------------------------------------------------------------- #
# Plan + augmenter                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class AugPlan:
    op: str
    severity: int
    seed: int
    light_params: Optional[Dict[str, float]] = None


def apply_plan_to_array(rgb_uint8: np.ndarray, plan: AugPlan, frame_index: int = 0) -> np.ndarray:
    """Apply ``plan`` to one HxWx3 uint8 RGB frame; returns uint8."""
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frame, got shape {rgb_uint8.shape}")
    x = rgb_uint8.astype(np.float32) / 255.0
    rng = np.random.default_rng(np.uint64(plan.seed) + np.uint64(frame_index) * np.uint64(7919))
    if plan.op == "light":
        params = plan.light_params or draw_light_params(plan.severity, rng)
        y = light(x, params)
    else:
        y = NOISE_OPS[plan.op](x, plan.severity, rng)
    return np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)


class VisualAugmenter:
    """Per-dataset augmenter built from the recipe's ``visual_aug`` field."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = validate_visual_aug_config(cfg, where="visual_aug")
        self.prob = self.cfg["prob"]
        self.views = self.cfg["views"]
        self.ops = self.cfg["ops"]
        self.sev_lo, self.sev_hi = self.cfg["severity"]
        self.zoom_cap = self.cfg["zoom_max_severity"]

    @classmethod
    def from_recipe(cls, meta: Dict[str, Any]) -> Optional["VisualAugmenter"]:
        cfg = meta.get("visual_aug") if isinstance(meta, dict) else None
        if cfg is None or cfg is False:
            return None
        return cls(cfg)

    def draw(self, py_rng: Union[random.Random, Any] = random) -> Optional[AugPlan]:
        """Coin flip + op/severity draw.  Returns ``None`` when the sample is left clean."""
        if self.prob <= 0.0 or py_rng.random() >= self.prob:
            return None
        op = py_rng.choice(self.ops)
        severity = py_rng.randint(self.sev_lo, self.sev_hi)
        if op == "zoom_blur":
            severity = min(severity, self.zoom_cap)
        seed = py_rng.getrandbits(32)
        plan = AugPlan(op=op, severity=severity, seed=seed)
        if op == "light":
            plan.light_params = draw_light_params(severity, np.random.default_rng(seed))
        return plan

    # -- application ---------------------------------------------------------
    def apply_images(self, images: List[Image.Image], plan: AugPlan) -> List[Image.Image]:
        out = list(images)
        for vi in (range(len(out)) if self.views == "all" else self.views):
            if vi < len(out) and out[vi] is not None:
                img = out[vi]
                mode = img.mode
                arr = np.asarray(img.convert("RGB"))
                aug = apply_plan_to_array(arr, plan, frame_index=vi if self.views == "all" else 0)
                new = Image.fromarray(aug)
                if mode not in ("RGB",):
                    new = new.convert("RGB")  # keep the processor input RGB
                out[vi] = new
        return out

    def apply_videos(self, videos: List[Any], plan: AugPlan) -> List[Any]:
        out = list(videos)
        for vi in (range(len(out)) if self.views == "all" else self.views):
            if vi < len(out) and out[vi] is not None:
                out[vi] = self._apply_video(out[vi], plan)
        return out

    @staticmethod
    def _apply_video(video: Any, plan: AugPlan) -> Any:
        if isinstance(video, torch.Tensor):
            if video.ndim != 4:
                raise ValueError(f"expected a TCHW video tensor, got shape {tuple(video.shape)}")
            frames = video.detach().cpu()
            if frames.dtype != torch.uint8:
                frames = frames.clamp(0, 255).round().to(torch.uint8)
            arr = frames.permute(0, 2, 3, 1).numpy()  # T H W C
            aug = np.stack([apply_plan_to_array(arr[t], plan, frame_index=t) for t in range(arr.shape[0])])
            result = torch.from_numpy(aug).permute(0, 3, 1, 2).contiguous()
            return result.to(video.dtype) if video.dtype != torch.uint8 else result
        if isinstance(video, (list, tuple)):
            return [
                (
                    Image.fromarray(apply_plan_to_array(np.asarray(f.convert("RGB")), plan, frame_index=t))
                    if isinstance(f, Image.Image)
                    else f
                )
                for t, f in enumerate(video)
            ]
        raise TypeError(f"unsupported video container {type(video).__name__}")
