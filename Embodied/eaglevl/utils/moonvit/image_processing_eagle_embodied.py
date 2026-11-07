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
#
# This file is adopted from the Kimi-VL project
# (https://github.com/MoonshotAI/Kimi-VL), licensed under the MIT License.
#
# Modifications © 2026 NVIDIA CORPORATION & AFFILIATES, licensed under
# the Apache License, Version 2.0.
#
# --------------------------------------------------------
# Kimi-VL
# Copyright (c) 2025 Moonshot AI
# Licensed under The MIT License
# --------------------------------------------------------

"""Image processor class for KimiVL."""

import math
import numpy as np
from PIL import Image
from typing import Optional, Union

import torch
from torchvision.transforms import functional as TF
from transformers.image_utils import ImageInput, make_list_of_images, valid_images
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import TensorType
from transformers import AutoImageProcessor

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class EagleEmbodiedImageProcessor(BaseImageProcessor):
    model_type = "eagle_embodied"

    def __init__(
        self,
        patch_size: int = 14,
        image_mean: tuple[float, float, float] = MEAN,
        image_std: tuple[float, float, float] = STD,
        in_token_limit: int = 25600,
        merge_kernel_size: list[int, int] = [2, 2],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_token_limit = in_token_limit
        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.merge_kernel_size = merge_kernel_size

    def rescale(
        self, image: Image.Image, merge_kernel_size: list[int, int] = [2, 2]
    ) -> Image.Image:
        w, h = image.size
        patch_size = self.patch_size

        if (w // patch_size) * (h // patch_size) > self.in_token_limit:
            scale = math.sqrt(self.in_token_limit / ((w // patch_size) * (h // patch_size)))
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        new_w, new_h = image.size
        pad_size_h = merge_kernel_size[0] * patch_size
        pad_size_w = merge_kernel_size[1] * patch_size

        target_w = math.ceil(new_w / pad_size_w) * pad_size_w
        target_h = math.ceil(new_h / pad_size_h) * pad_size_h

        if target_w != new_w or target_h != new_h:
            image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)

        w, h = image.size
        if w // patch_size >= 512 or h // patch_size >= 512:
            raise ValueError("Exceed pos emb")

        return image

    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        return TF.to_tensor(image.convert("RGB"))

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        return TF.normalize(image, self.image_mean, self.image_std)

    def patchify(self, image: torch.Tensor) -> tuple[torch.Tensor, list[int, int]]:
        patch_size = self.patch_size
        C, H, W = image.shape
        patches = image.reshape(C, H // patch_size, patch_size, W // patch_size, patch_size)
        patches = patches.permute(1, 3, 0, 2, 4)
        patches = patches.contiguous().view(-1, C, patch_size, patch_size)
        grid_hw = (H // patch_size, W // patch_size)
        return patches, grid_hw

    def _preprocess(self, image: ImageInput) -> tuple[torch.Tensor, list[int, int]]:
        """
        Preprocess image and patchify it.
        Args:
            image (`ImageInput`):
                Image to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
        Returns:
            patches: torch.Tensor
            grid_hw: list[int, int]
        """
        image = self.rescale(image, self.merge_kernel_size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        patches, grid_hw = self.patchify(image)
        return patches, grid_hw

    def preprocess_video_tensor(self, video: "torch.Tensor") -> BatchFeature:
        """
        Batch-process a video tensor directly, bypassing PIL conversion.

        Args:
            video: [T, C, H, W] float tensor with pixel values in [0, 255].
        Returns:
            BatchFeature with pixel_values and image_grid_hws.
        """
        T, C, H, W = video.shape
        patch_size = self.patch_size
        merge_h, merge_w = self.merge_kernel_size

        tokens_per_frame = (W // patch_size) * (H // patch_size)
        if tokens_per_frame > self.in_token_limit:
            scale = math.sqrt(self.in_token_limit / tokens_per_frame)
            H_new, W_new = int(H * scale), int(W * scale)
            video = TF.resize(
                video, [H_new, W_new],
                interpolation=TF.InterpolationMode.BICUBIC, antialias=True,
            )
            _, _, H, W = video.shape

        pad_h = merge_h * patch_size
        pad_w = merge_w * patch_size
        target_h = math.ceil(H / pad_h) * pad_h
        target_w = math.ceil(W / pad_w) * pad_w

        if target_h != H or target_w != W:
            video = TF.resize(
                video, [target_h, target_w],
                interpolation=TF.InterpolationMode.BICUBIC, antialias=True,
            )
            _, _, H, W = video.shape

        if W // patch_size >= 512 or H // patch_size >= 512:
            raise ValueError("Exceed pos emb")

        video = video.float() / 255.0
        mean = torch.tensor(self.image_mean, dtype=video.dtype, device=video.device).view(1, 3, 1, 1)
        std = torch.tensor(self.image_std, dtype=video.dtype, device=video.device).view(1, 3, 1, 1)
        video = (video - mean) / std

        grid_h, grid_w = H // patch_size, W // patch_size
        patches = video.reshape(T, C, grid_h, patch_size, grid_w, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(T * grid_h * grid_w, C, patch_size, patch_size)

        image_grid_hws = np.tile(np.array([[grid_h, grid_w]]), (T, 1))
        return BatchFeature(data={"pixel_values": patches, "image_grid_hws": image_grid_hws})

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        pixel_values, image_grid_hws = [], []
        for image in images:
            patches, image_grid_hw = self._preprocess(image)
            pixel_values.append(patches)
            image_grid_hws.append(image_grid_hw)
        pixel_values = torch.concat(pixel_values, dim=0)
        image_grid_hws = np.array(image_grid_hws)
        data = {"pixel_values": pixel_values, "image_grid_hws": image_grid_hws}

        return BatchFeature(data=data, tensor_type=return_tensors)

AutoImageProcessor.register("EagleEmbodiedImageProcessor", EagleEmbodiedImageProcessor)
