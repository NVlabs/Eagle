from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM decoder.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP layers of the model.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the backbone model. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT model. Default is 0.'},
    )
    loss_version: str = field(
        default='v1',
        metadata={'help': 'loss version'}
    )
    save_every_n_hours: int = field(
        default=4,
    )
    pre_feature_reduction: bool = field(
        default=False
    )
    freeze_backbones: Optional[str] = field(
        default=None
    )
    lr_scale: Optional[str] = field(
        default=None,
        metadata={'help': "available: [None, 'vision_model: 0.1, mlp: 1.0, llm: 1.0']"}
    )
    use_fp8: bool = field(
        default=False,
        metadata={'help': 'Set to True to use fp8.'}
    )
    use_pixel_shuffle: bool = field(
        default=True,
        metadata={'help': 'Set to True to use pixel shuffle.'}
    )
    mlp_connector_layers: int = field(
        default=2,
        metadata={'help': 'Set the number of MLP connector layers. Default is 2.'}
    )
    use_vit_windows_attn: bool = field(
        default=True,
        metadata={'help': 'Set to True to use ViT Windows Attention.'}
    )
    use_vit_rope: bool = field(
        default=True,
        metadata={'help': 'Set to True to use ViT Rope.'}
    )
    auto_thinking_prompt: bool = field(
        default=False,
        metadata={'help': 'Set to True to use auto thinking prompt.'}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=224,
        metadata={'help': 'Set the desired size for the image. Default is 224.'},
    )
    down_sample_ratio: Optional[float] = field(
        default=1.0,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 1.0.'},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='qwen2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic image size.'},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image.'},
    )
    min_dynamic_tiles: Optional[int] = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_tiles: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 6.'},
    )
    neftune_alpha: Optional[float] = field(
        default=None,
        metadata={'help': 'The noise_alpha value for NEFTune. Default is None.'},
    )
    normalize_type: Optional[str] = field(
        default='imagenet',
        metadata={'help': 'The normalize type for the image. Default is imagenet.'},
    )
    sequence_parallel_degree: int = field(
        default=1,
        metadata={'help': 'Sequence parallelism degree. Default is 1.'}
    )
    ring_sequence_parallel_degree: int = field(
        default=1,
        metadata={'help': 'Sequence parallelism ring degree. Default is 1.'}
    )
    sample_length_div: int = field(
        default=1,
        metadata={'help': 'Specify the division of sample length. Default is 1.'}
    )
    padding_last_sample: bool = field(
        default=False,
        metadata={'help': 'Set to True to pad the last sample to the maximum length.'}
    )
    zip_json: bool = field(
        default=False,
        metadata={'help': 'Set to True to use zip json.'}
    )
    use_onelogger: bool = field(   
        default=False,
        metadata={'help': 'Set to True to use one logger.'}
    )
    use_online_packing: bool = field(
        default=True,
        metadata={'help': 'Set to True to use online packing.'}
    )
