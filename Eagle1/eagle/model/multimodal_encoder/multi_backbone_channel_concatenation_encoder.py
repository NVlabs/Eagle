# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .convnext_encoder import ConvNextVisionTower
from .hr_clip_encoder import HRCLIPVisionTower
from .vision_models.eva_vit import EVAVITVisionTower
from .sam_encoder import SAMVisionTower
from .pix2struct_encoder import Pix2StructLargeVisionTower
from .timm_encoder import TimmVisionTower
from .qwen2_5vl_encoder import QwenVisionTower
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from copy import deepcopy
import random
import math

class MultiBackboneChannelConcatenationVisionTower(nn.Module):
    def __init__(self,
                 vision_tower,
                 args,
                 grid_size=32):
        
        super().__init__()

        self.is_loaded = False
        self.grid_size = grid_size
        self.num_tokens = self.grid_size ** 2
        
        vision_tower_name_list = vision_tower.split(";")
        self.input_image_size = 1024 # hardcode
        self.load_vision_towers(vision_tower_name_list, args)

      
    def load_vision_towers(self, vision_tower_name_list, args):
        self.vision_towers = nn.ModuleList()
        for name in vision_tower_name_list:
            if name == 'det-1024':
                det_args = deepcopy(args)
                det_args.input_image_size = 1024
                det_args.freeze_vision = False
                det_args.vision_tower_pretrained_from = 'checkpoints/pretrained_models/eva02_L_coco_det_sys_o365.pth'
                det_vision_tower = EVAVITVisionTower("eva02-l-16", det_args)     
                det_vision_tower.load_model()
                self.vision_towers.append(det_vision_tower)

            elif name == 'convnext-1024':
                ## ConvNeXt
                convnext_args = deepcopy(args)
                convnext_args.freeze_vision = False
                convnext_args.input_image_size = 1024
                convnext_args.vision_tower_pretrained_from = '/mnt/models/convnext/pytorch_model.bin'
                convnext_vision_tower = "convnext_xxlarge.clip_laion2b_soup" # hardcode
                convnext_vision_tower = ConvNextVisionTower(convnext_vision_tower, 
                                                                convnext_args)
                convnext_vision_tower.load_model()      
                self.vision_towers.append(convnext_vision_tower)
            
            elif name == "sam-1024":
                sam_args = deepcopy(args)
                sam_args.freeze_vision = False
                sam_args.input_image_size = 1024
                sam_args.add_pixel_shuffle = True
                sam_vision_tower = SAMVisionTower("SAM-L", sam_args)
                sam_vision_tower.load_model()
                self.vision_towers.append(sam_vision_tower)

            elif name == 'pix2struct-1024':
                pix_args = deepcopy(args)
                #pix_args.freeze_vision = True
                pix_args.input_image_size = 1024
                pix_args.freeze_vision = False
                pix_args.do_resize = True
                pix_args.de_normalize = True
                pix_vision_tower = Pix2StructLargeVisionTower("pix2struct-large", pix_args)     
                pix_vision_tower.load_model()
                self.vision_towers.append(pix_vision_tower)

            elif name == 'clip-448':
                clip_args = deepcopy(args)
                clip_args.input_image_size = 336 # actually 448, will have no effect
                clip_args.freeze_vision = False
                clip_vision_tower = HRCLIPVisionTower("openai/clip-vit-large-patch14-336", clip_args)     
                clip_vision_tower.load_model()
                self.vision_towers.append(clip_vision_tower)

            elif name == 'timm-vit':
                timm_args = deepcopy(args)
                timm_args.freeze_vision = False
                timm_args.input_image_size = 1024
                timm_vision_tower = TimmVisionTower("uni_v1", timm_args)
                timm_vision_tower.load_model(PYTORCH_CKPT_PATH='/mnt/models/UNI/pytorch_model.bin')
                self.vision_towers.append(timm_vision_tower)

            elif name == 'qwen2_5-vl':
                qwen_args = deepcopy(args)
                qwen_args.freeze_vision = False
                qwen_args.input_image_size = 1024

                qwen_vision_tower = QwenVisionTower(
                    "qwen2_5-vl",
                    qwen_args
                )

                qwen_vision_tower.load_model()
                self.vision_towers.append(qwen_vision_tower)
                self.qwen_processor = qwen_vision_tower.image_processor

        # a hardcode here, so we always use convnext in the vision encoder mixture
        self.image_processor = convnext_vision_tower.image_processor
        self.is_loaded = True

    def load_model(self):
        assert self.is_loaded, "All the vision encoders should be loaded during initialization!"

    def forward(self, x, xq=None):
        features = []
        if isinstance(self.vision_towers[-1], QwenVisionTower):
            self.qwen2_5_vision_tower = self.vision_towers[-1]
            self.vision_towers = self.vision_towers[:-1]
        else:
            self.qwen2_5_vision_tower = None

        for vision_tower in self.vision_towers:
            if vision_tower.input_image_size != self.input_image_size:
                resized_x = F.interpolate(x.float(), 
                                          size=(vision_tower.input_image_size, vision_tower.input_image_size), 
                                          mode='bilinear', 
                                          align_corners=True).to(dtype=x.dtype)
            else:
                resized_x = x
            feature = vision_tower(resized_x)
            if len(feature.shape) == 3: # b, n, c
                b, n, c = feature.shape
                if n == self.num_tokens:
                    features.append(feature)
                    continue

                w = h = int(n**0.5)
                feature = feature.transpose(1,2).reshape(b, c, h, w)
            else:
                b, c, h, w = feature.shape

            if w != self.grid_size:
                feature = F.interpolate(feature.float(), size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=True).to(dtype=x.dtype)
            features.append(feature.flatten(2,3).transpose(1,2))
        
        features = torch.cat(features, dim=-1)

        if self.qwen2_5_vision_tower is not None:
            qwen2_5_images_features = self.qwen2_5_vision_tower(xq)
        else:
            qwen2_5_images_features = None

        return {
            "fusion_feature": features,
            "qwen2_5_images_features": qwen2_5_images_features
        }
        
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.clip_vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.clip_vision_tower.parameters()).device

    @property
    def config(self):
        assert NotImplementedError
        pass

    @property
    def hidden_size(self):
        return sum([_.hidden_size for _ in self.vision_towers])

    @property
    def num_patches(self):
        return self.num_tokens
