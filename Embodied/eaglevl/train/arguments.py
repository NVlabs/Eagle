# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={'help': 'Use reentrant autograd checkpointing. Non-reentrant is recommended for long contexts.'},
    )
    rope_scaling_type: Optional[str] = field(
        default='auto',
        metadata={'help': 'RoPE scaling for long context: auto, yarn, dynamic, linear, or none.'},
    )
    rope_scaling_factor: Optional[float] = field(
        default=None,
        metadata={'help': 'Optional RoPE scaling factor. By default it is derived from max_seq_length.'},
    )
    rope_original_max_position_embeddings: Optional[int] = field(
        default=None,
        metadata={'help': 'Original pretrained context length used to derive RoPE scaling.'},
    )
    save_every_n_hours: int = field(
        default=4,
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
    mlp_connector_layers: int = field(
        default=2,
        metadata={'help': 'Set the number of MLP connector layers. Default is 2.'}
    )
    chat_template_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to chat template file.'}
    )
    processor_config_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to processor config file.'}
    )
    preprocessor_config_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to preprocessor config file.'}
    )
    block_size: int = field(
        default=4,
        metadata={'help': 'block size of mask token.'},
    )
    causal_attn: bool = field(
        default=False,
        metadata={'help': 'causal attention or not. default to False.'},
    )
    attn_implementation: Optional[str] = field(
        default='magi',
        metadata={'help': 'attention implementation: magi, flash_attention_2, sdpa, eager.'},
    )
    expected_mask_repeat_times: int = field(
        default=3,
        metadata={'help': 'expected mask repeat time.'},
    )
    # VFlash options. They apply only when training_mode is set; the defaults
    # are those of the reference runs.
    training_mode: Optional[str] = field(
        default=None,
        metadata={
            'help': (
                'VFlash objective. Unset trains Eagle Embodied as usual; "ar" trains '
                'the VFlash target with physical-atom tokens; "vflash" freezes that '
                'target and trains a separate VFlash drafter.'
            )
        },
    )
    oof_loss_scale: float = field(
        default=0.1,
        metadata={'help': 'Loss weight of every <oof> target relative to other tokens.'},
    )
    vflash_anchors_per_sequence: int = field(
        default=512,
        metadata={'help': 'Random ordinary-text response anchors sampled per clean sequence.'},
    )
    vflash_loss_gamma: float = field(
        default=7.0,
        metadata={
            'help': (
                'Position-weight decay gamma of ordinary-text B16 blocks. '
                'Physical-atom blocks are always uniform.'
            )
        },
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
    conv_style: Optional[str] = field(
        default='qwen2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    jsonl_index_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': (
                'Shared working directory for per-launch JSONL byte-offset indexes. '
                'Training rebuilds every index on each launch. '
                'Defaults to .jsonl_index next to meta_path.'
            )
        },
    )
    jsonl_index_workers: int = field(
        default=4,
        metadata={'help': 'Concurrent index builders per distributed rank.'},
    )
    jsonl_index_chunk_size_mb: int = field(
        default=4,
        metadata={'help': 'Chunk size in MiB used while scanning JSONL files.'},
    )
    neftune_alpha: Optional[float] = field(
        default=None,
        metadata={'help': 'The noise_alpha value for NEFTune. Default is None.'},
    )
    n_frames: Optional[int] = field(
        default=16,
        metadata={'help': 'The number of frames for the video. Default is 16.'},
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
    use_onelogger: bool = field(   
        default=False,
        metadata={'help': 'Set to True to use one logger.'}
    )
    use_online_packing: bool = field(
        default=True,
        metadata={'help': 'Set to True to use online packing.'}
    )
    video_total_pixels: int = field(
        default=16384 * 28 * 28 * 0.8,
        metadata={'help': 'The total number of pixels for the video. Default is 32000 * 28 * 28 * 0.9.'}
    )
    max_frames: int = field(
        default=64,
        metadata={'help': 'The maximum number of frames for the video. Default is 16.'}
    )
    target_fps: int = field(
        default=2,
        metadata={'help': 'The target FPS for the video. Default is 2.'}
    )
    max_num_tokens_per_sample: int = field(
        default=32768,
        metadata={'help': 'Maximum tokens allowed in one raw sample; longer samples are skipped.'}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={'help': 'Hard limit on tokens in a packed batch; flush if adding a sample would exceed it.'}
    )
    packing_buffer_size: int = field(
        default=32,
        metadata={'help': 'The buffer size for the packing. Default is 32.'}
    )
    balance_media_tokens: bool = field(
        default=False,
        metadata={'help': 'Align media-cost quantiles across data-parallel ranks with bounded lookahead.'}
    )
    balance_window_size: int = field(
        default=8,
        metadata={'help': 'Number of completed packs in one media-balance lookahead window.'}
    )
    balance_seed: Optional[int] = field(
        default=None,
        metadata={'help': 'Shared media-balance seed. Defaults to the training seed on every rank.'}
    )
    debug_assert_sp_batch_sync: bool = field(
        default=False,
        metadata={'help': 'Hash and compare initial packed batches inside each sequence-parallel group.'}
    )
    sample_log_interval: int = field(
        default=100,
        metadata={'help': 'Log image count every N steps. Default is 100.'}
    )
    strict_data: bool = field(
        default=False,
        metadata={
            'help': (
                'Abort on any data error instead of skipping the sample, so the trained '
                'stream never silently diverges. A transient error such as a video '
                'read timeout then ends the job; resume it from the last checkpoint.'
            )
        },
    )
    packing_log_interval: int = field(
        default=10000,
        metadata={'help': 'Log packing occupancy every N packs on the first data worker.'}
    )
    vflash_max_samples_per_pack: int = field(
        default=8,
        metadata={
            'help': (
                'A VFlash pack closes once it holds this many clean samples. With '
                '--vflash_train_sample_epochs, max_num_tokens must fit that many '
                'samples of max_num_tokens_per_sample.'
            )
        },
    )
    vflash_train_sample_epochs: Optional[float] = field(
        default=None,
        metadata={
            'help': (
                'Set max_steps from this many raw-sample epochs of the recipe '
                '(--training_mode vflash only).'
            )
        },
    )
    auto_thinking_handler: bool = field(
        default=False,
        metadata={'help': 'Set to True to auto handle thinking mode format. Default is False.'}
    )