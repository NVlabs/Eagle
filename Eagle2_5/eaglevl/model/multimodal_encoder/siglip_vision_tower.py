import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..siglip import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig

import math
import torch
import torch.nn.functional as F
from typing import List, Optional
import os
from transformers import AutoConfig

class SiglipVisionTower(nn.Module):
    # We use the same wrapper as the default clip encoder. 
    # See `clip_encoder.py` in the same folder
    def __init__(self, vision_tower, args, delay_load=False, raw_config=None):
        super().__init__()

        self.is_loaded = False
        self.freeze_vision=args.freeze_vision
        self.input_image_size=args.input_image_size
        self.vision_tower_name = vision_tower
        # self.select_layer = args.mm_vision_select_layer
        self.name = 'siglip'
        # self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.delay_load = delay_load
        self.raw_config = raw_config
        if not delay_load:
            self.load_model()
        else:
            if os.path.isfile(self.vision_tower_name):
                self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name, local_files_only=True)
            elif isinstance(self.raw_config.vision_config.siglip_vision_config, SiglipVisionConfig):
                self.cfg_only = self.raw_config.vision_config.siglip_vision_config
            elif isinstance(self.raw_config.vision_config.siglip_vision_config, dict):
                self.cfg_only = SiglipVisionConfig(**self.raw_config.vision_config.siglip_vision_config)
            else:
                raise ValueError(f"Invalid type for siglip_vision_config: {type(self.raw_config.vision_config.siglip_vision_config)}")

    def load_model(self):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = SiglipImageProcessor(size=1024)
        # self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, local_files_only=True, torch_dtype=torch.bfloat16)
        if self.delay_load:
            # cfg = SiglipVisionConfig.from_pretrained(self.vision_tower_name, local_files_only=True)
            self.vision_tower = SiglipVisionModel(self.cfg_only)
        else:    
            self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, local_files_only=True)

        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

        self.vision_tower.vision_model.encoder.gradient_checkpointing = True
        self.is_loaded = True

    def forward(self, images):
        return self.vision_tower(
                pixel_values=images,
                output_hidden_states=False,
                return_dict=True).last_hidden_state


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
