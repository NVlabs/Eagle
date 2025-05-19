import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoImageProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

class QwenVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.freeze_vision = args.freeze_vision
        self.vision_tower_name = vision_tower
        self.vision_tower_model_path = args.qwen_visual_tower_path

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            # Load config only if model loading is delayed and not unfreezing
            self.cfg_only = Qwen2_5_VLVisionConfig.from_pretrained(self.vision_tower_model_path)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, load_model called again, skipping.')
            return

        # 加载图像处理器
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_model_path)

        # 加载完整模型并提取视觉部分
        full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.vision_tower_model_path,
            torch_dtype="auto",
            device_map=device_map,
            attn_implementation="sdpa"
        )

        self.vision_tower = full_model.visual
        # If you need the vision_config specifically:
        # self.vision_config = full_model.config.vision_config

        # 释放完整模型
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        """
        处理输入图像并返回视觉特征
        Args:
            images: 包含 pixel_values 和 image_grid_thw 的字典
        """
        hidden_states = images['pixel_values']
        grid_thws = images['image_grid_thw'] # Corrected key from ' image_grid_thw' to 'image_grid_thw'

        if isinstance(hidden_states, list): # Prefer isinstance over type()
            image_features_list = []
            for hs_item, gt_item in zip(hidden_states, grid_thws):
                image_feature = self.vision_tower(
                    hidden_states=hs_item.to(device=self.device, dtype=self.dtype),
                    grid_thw=gt_item.to(device=self.device, dtype=torch.int),
                )
                image_features_list.append(image_feature)
            image_features = torch.stack(image_features_list)
        else:
            hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
            # Assuming grid_thws is singular if hidden_states is not a list
            # or if it's already a batch, it should be named grid_thws not grid_thw
            current_grid_thw = grid_thws.to(device=self.device, dtype=torch.int)
            image_features = self.vision_tower(
                hidden_states=hidden_states,
                grid_thw=current_grid_thw
            )
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
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        # Qwen默认使用32x32的特征网格
        # This might also be derivable from config: config.image_size // config.patch_size
        # For Qwen2.5-VL, it is indeed 32 (e.g. 448x448 image, 14x14 patch size from ViT -> 32x32 patches)
        return 32

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2