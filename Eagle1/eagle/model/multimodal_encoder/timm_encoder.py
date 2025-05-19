import os
import torch
import torch.nn as nn

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login, hf_hub_download

class TimmVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower = None # Initialize vision_tower attribute
        self.image_processor = None # Initialize image_processor attribute
        self.freeze_vision = args.freeze_vision
        self.input_image_size = args.input_image_size
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.timm_kwargs = {}

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False): # Assuming 'unfreeze_mm_vision_tower'
            self.load_model()

    def load_model(self, PYTORCH_CKPT_PATH='/mnt/models/UNI/pytorch_model.bin'):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, load_model called again, skipping.')
            return

        if self.vision_tower_name == 'uni_v1':
            self.timm_kwargs = {
                'model_name': 'vit_large_patch16_224',
                'img_size': 224,
                'patch_size': 16,
                'init_values': 1e-5,
                'num_classes': 0,
                'dynamic_img_size': True
            }
            self.vision_tower = timm.create_model(pretrained=False, **self.timm_kwargs)
            self.is_loaded = True
        elif self.vision_tower_name == 'virchow':
            self.timm_kwargs = {
                'model_name': 'vit_huge_patch14_224',
                'img_size': 224,
                'patch_size': 14,
                'init_values': 1e-5,
                'mlp_ratio': 5.3375,
                'num_classes': 0,
                'mlp_layer': timm.layers.SwiGLUPacked, # Corrected from 'mip_layer' if that was a typo
                'act_layer': torch.nn.SiLU,
                'dynamic_img_size': True,
                'global_pool': ""
            }
            self.vision_tower = timm.create_model(pretrained=False, **self.timm_kwargs)
            self.vision_tower.load_state_dict(torch.load(PYTORCH_CKPT_PATH), strict=True)
            self.is_loaded = True
        elif self.vision_tower_name == 'virchow2':
            self.timm_kwargs = {
                'model_name': 'vit_huge_patch14_224',
                'img_size': 224,
                'patch_size': 14,
                'init_values': 1e-5,
                'mlp_ratio': 5.3375, # Corrected from 'mip_ratio'
                'num_classes': 0,
                'mlp_layer': timm.layers.SwiGLUPacked,
                'act_layer': torch.nn.SiLU,
                'reg_tokens': 4,
                'dynamic_img_size': True,
                'global_pool': ""
            }
            self.vision_tower = timm.create_model(pretrained=False, **self.timm_kwargs)
            # self.vision_tower.load_state_dict(torch.load(PYTORCH_CKPT_PATH), strict=True)
            self.is_loaded = True
        elif self.vision_tower_name == 'uni_v2':
            self.timm_kwargs = {
                'model_name': 'vit_giant_patch14_224',
                'img_size': 224,
                'patch_size': 14,
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5,
                'embed_dim': 1536,
                'mlp_ratio': 2.66667 * 2,
                'num_classes': 0,
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked,
                'act_layer': torch.nn.SiLU,
                'reg_tokens': 8,
                'dynamic_img_size': True
            }
            self.vision_tower = timm.create_model(pretrained=False, **self.timm_kwargs)
            # self.vision_tower.load_state_dict(torch.load(PYTORCH_CKPT_PATH), strict=True)
            self.is_loaded = True
        else:
            print(f"Unknown vision tower name: {self.vision_tower_name}. Model not loaded.")
            self.is_loaded = False
            return # Exit if no model is matched

        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

        config = resolve_data_config(self.vision_tower.pretrained_cfg, model=self.vision_tower)
        self.image_processor = create_transform(**config)

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_features = self.vision_tower.forward_features(image.to(device=self.divice, dtype=self.dtype))[:, 1:]
                image_features.append(image_features)
        else:
            image_features = self.vision_tower.forward_features(images)[:, 1:]

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
    
    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype
    
    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.timm_kwargs
        else:
            return None

    @property
    def hidden_size(self):
        return self.vision_tower.num_features

    @property
    def num_patches_per_side(self):
        return self.input_image_size // self.timm_kwargs['patch_size']


    @property
    def num_patches(self):
        return (self.input_image_size // self.timm_kwargs['patch_size']) ** 2