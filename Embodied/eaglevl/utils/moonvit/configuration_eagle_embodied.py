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
# This file is adopted from the InternVL project
# (https://github.com/OpenGVLab/InternVL), licensed under the MIT License.
#
# Modifications © 2026 NVIDIA CORPORATION & AFFILIATES, licensed under
# the Apache License, Version 2.0.
#
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
logger = logging.get_logger(__name__)

class MoonViTConfig(PretrainedConfig):
    model_type = "moonvit"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        merge_kernel_size: tuple[int, int] = (2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        # Positional embedding config
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        # Transformer config
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # Patch merger config
        self.merge_kernel_size = merge_kernel_size


class EagleEmbodiedConfig(PretrainedConfig):
    model_type = 'eagle_embodied'
    is_composition = True
    sub_configs = {"vision_config": MoonViTConfig, "text_config": PretrainedConfig}
    def __init__(
            self,
            vision_config=None,
            text_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            downsample_ratio=0.5,
            template=None,
            loss_version='v1',
            mlp_checkpoint=False,
            image_token_index=151667,
            initializer_range=0.01,
            **kwargs):
        if isinstance(vision_config, dict):
            vision_config = vision_config.copy()
            vision_model_type = vision_config.get("model_type", "moonvit")
            if vision_model_type != "moonvit":
                raise ValueError(
                    'Unsupported model_type: {}. Only moonvit is supported.'.format(vision_model_type)
                )
            vision_config = MoonViTConfig(**vision_config)
        elif vision_config is None:
            vision_config = MoonViTConfig()
        elif not isinstance(vision_config, PretrainedConfig):
            raise TypeError(f"vision_config must be a dict or PretrainedConfig, got {type(vision_config)!r}")

        if isinstance(text_config, dict):
            text_config = text_config.copy()
            text_model_type = text_config.get("model_type")
            text_arch = text_config.get("architectures", [None])[0]
            if text_arch == 'Qwen3VLForConditionalGeneration' or text_model_type == 'qwen3_vl_text':
                text_config = Qwen3VLTextConfig(**text_config)
            elif text_model_type == "qwen3" or text_arch == "Qwen3ForCausalLM":
                text_config = Qwen3Config(**text_config)
            elif text_model_type in (None, "qwen2") or text_arch in (None, "Qwen2ForCausalLM"):
                text_config = Qwen2Config(**text_config)
            else:
                raise ValueError(
                    'Unsupported architecture: {}. '
                    'Supported: Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3VLForConditionalGeneration.'.format(
                        text_arch
                    )
                )
        elif text_config is None:
            text_config = Qwen2Config()
        elif not isinstance(text_config, PretrainedConfig):
            raise TypeError(f"text_config must be a dict or PretrainedConfig, got {type(text_config)!r}")

        self.vision_config = vision_config
        self.text_config = text_config

        super().__init__(**kwargs)
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.mlp_checkpoint = mlp_checkpoint
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.loss_version = loss_version
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.image_token_index = image_token_index
        self.initializer_range = initializer_range

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['text_config'] = self.text_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['image_token_index'] = self.image_token_index
        output['initializer_range'] = self.initializer_range
        for attr in ('_attn_implementation', '_attn_implementation_autoset', '_attn_implementation_internal'):
            val = getattr(self, attr, None)
            if val is not None:
                output[attr] = val
        return output
