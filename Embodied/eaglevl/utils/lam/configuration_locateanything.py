# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
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


class LocateAnythingConfig(PretrainedConfig):
    model_type = 'locateanything'
    is_composition = True
    sub_configs = {"vision_config": MoonViTConfig, "text_config": Qwen2Config}
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
            box_start_token_id=151668,
            box_end_token_id=151669,
            coord_start_token_id=151677,
            coord_end_token_id=152677,
            ref_start_token_id=151672,
            ref_end_token_id=151673,
            none_token_id=4064,
            **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {'model_type': 'moonvit'}
            logger.info('vision_config is None. Initializing the MoonViTConfig with default values.')

        if text_config is None:
            text_config = {'architectures': ['Qwen2ForCausalLM']}
            logger.info('text_config is None. Initializing the Qwen2Config config with default values.')

        if vision_config['model_type'] == 'moonvit':
            self.vision_config = MoonViTConfig(**vision_config)
        else:
            raise ValueError('Unsupported model_type: {}. Only moonvit is supported.'.format(vision_config['model_type']))


        if text_config['architectures'][0] == 'Qwen2ForCausalLM':
            self.text_config = Qwen2Config(**text_config)
        elif text_config['architectures'][0] == 'Qwen3ForCausalLM':
            self.text_config = Qwen3Config(**text_config)
        else:
            raise ValueError('Unsupported architecture: {}. Only Qwen2ForCausalLM and Qwen3ForCausalLM are supported.'.format(text_config['architectures'][0]))
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.mlp_checkpoint = mlp_checkpoint
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.loss_version = loss_version
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.image_token_index = image_token_index
        self.box_start_token_id = box_start_token_id
        self.box_end_token_id = box_end_token_id
        self.coord_start_token_id = coord_start_token_id
        self.coord_end_token_id = coord_end_token_id
        self.ref_start_token_id = ref_start_token_id
        self.ref_end_token_id = ref_end_token_id
        self.none_token_id = none_token_id

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
        output['box_start_token_id'] = self.box_start_token_id
        output['box_end_token_id'] = self.box_end_token_id
        output['coord_start_token_id'] = self.coord_start_token_id
        output['coord_end_token_id'] = self.coord_end_token_id
        output['ref_start_token_id'] = self.ref_start_token_id
        output['ref_end_token_id'] = self.ref_end_token_id
        output['none_token_id'] = self.none_token_id
        output['_attn_implementation'] = self._attn_implementation
        if hasattr(self, '_attn_implementation_autoset'):
            output['_attn_implementation_autoset'] = self._attn_implementation_autoset
        return output
