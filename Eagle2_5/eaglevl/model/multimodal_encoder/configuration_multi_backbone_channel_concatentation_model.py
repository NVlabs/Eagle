# --------------------------------------------------------
# EagleVL
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from ..siglip import SiglipVisionConfig
logger = logging.get_logger(__name__)


class MultiBackboneChannelConcatenationVisionModelConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MultiBackboneChannelConcatenationVisionModelConfig`]. It is used to
    instantiate a vision encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_path (str): Path to the vision model or its configuration.
        mm_vision_select_layer (int, optional): The layer to select from the vision model
                                                for multi-modal processing. Defaults to -2.
        grid_size (int, optional): The size of the grid for vision processing. Defaults to 32.
        **kwargs: Additional keyword arguments to be passed to the parent PretrainedConfig.
        
    """

    model_type = 'MOB'

    def __init__(
            self,
            vision_path,
            mm_vision_select_layer=-1,
            grid_size=32,  
            hidden_size='lazy_calculation',
            freeze_backbones=None,
            moe_version_type='convnext_512_siglip_448',
            delay_load=False,
            convnext_img_size=512,
            radio_img_size=512,
            siglip_img_size=448,
            vision_tower_siglip_path=None,
            vision_tower_convnext_path='convnext_xxlarge.clip_laion2b_soup',
            vision_tower_radio_path=None,
            normalize_type='siglip',
            image_size=dict(
                siglip=448,
                convnext=512,
                radio=448
            ),
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.normalize_type = normalize_type
        self.vision_path = vision_path
        self.mm_vision_select_layer = mm_vision_select_layer
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.freeze_backbones = freeze_backbones
        self.moe_version_type = moe_version_type
        self.delay_load = delay_load
        self.convnext_img_size = convnext_img_size
        self.radio_img_size = radio_img_size
        self.siglip_img_size = siglip_img_size
        # other args. to make it compatable with eagle-next
        self.vision_tower_siglip_path = vision_tower_siglip_path
        self.vision_tower_radio_path = vision_tower_radio_path
        if os.path.exists(vision_tower_siglip_path):
            self.siglip_vision_config = SiglipVisionConfig.from_pretrained(self.vision_tower_siglip_path, local_files_only=True)
        if vision_tower_radio_path is not None and os.path.exists(vision_tower_radio_path):
            self.radio_vision_config = AutoConfig.from_pretrained(vision_tower_radio_path, trust_remote_code=True)
        self.vision_tower_convnext_path = vision_tower_convnext_path
        self.vision_tower = self.vision_path[4:] # remove `MOB:` prefix

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'vision_config' in config_dict:
            config_dict = config_dict['vision_config']

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)
