import os
import os.path as osp
import gc
import logging
import math
import random
import sys
import warnings
import numpy as np
from copy import deepcopy
from datasets import load_dataset
from typing import Dict, Optional

import json
import time
import torch
import torch.distributed as dist
import transformers
import socket
from eaglevl.dist_utils import init_dist

# Check transformers version to determine import path
import importlib
import packaging.version as version
from eaglevl.model.siglip.modeling_siglip import SiglipVisionModel as CustomSiglipVisionModel
transformers_version = importlib.import_module("transformers").__version__
if version.parse(transformers_version) < version.parse("4.40.0"):
    # implenmenting flash attention 2 in custom siglip vision model
    transformers.models.siglip.modeling_siglip.SiglipVisionModel = CustomSiglipVisionModel
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from eaglevl.patch import ( 
                            replace_train_sampler,
                            replace_train_sampler_for_online_packing,
                            replace_liger_fused_ops,
                            get_collator,
                            patch_packing_attention)
from eaglevl.model.eagle2_5 import Eagle2_5_VLForConditionalGeneration, Eagle2_5_VLConfig
from eaglevl.sp_utils import set_pg_manager, get_pg_manager
from eaglevl.train.constants import special_tokens_list, IMG_CONTEXT_TOKEN 
from eaglevl.train.dataset import (ConcatDatasetForOnlinePacking, build_transform,
                                    dynamic_preprocess, PROCESS_FUNCTIONS)
from eaglevl.train.arguments import ModelArguments, DataTrainingArguments
from eaglevl.train.loading_frames import get_frames_for_multiple_videos_and_images
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed, AutoProcessor)
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
from eaglevl.train.tools import SaveCheckpointCallback, MemoryLoggerCallback, get_last_checkpoint_guard

from eaglevl.model.c_radio.radio_model import RADIOModel, RADIOConfig
from eaglevl.train.one_logger import create_onelogger_config, warp_onelogger_trainer
from dotenv import load_dotenv
load_dotenv()

if version.parse(torch.__version__) >= version.parse("2.4.0"):
    torch.serialization.add_safe_globals(
        [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))])


#============ Patch ============
replace_liger_fused_ops() # To save memory
#========================

#============ for loading large images ============
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
# ==================================================

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, template_name, meta, tokenizer, tcs_loader, num_image_token,
                 image_size=224, is_train=True, pad2square=False, group_by_length=False,
                 dynamic_image_size=False, use_thumbnail=False, min_dynamic_tiles=1,
                 max_dynamic_tiles=6, repeat_time=1, normalize_type='imagenet', sample_length_div=1,
                 zip_json=True, use_online_packing=False):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.sample_length_div = sample_length_div
        self.max_length = self.tokenizer.model_max_length
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.zip_json = zip_json
        self.use_online_packing = use_online_packing
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_tiles: {min_dynamic_tiles}, max_dynamic_tiles: {max_dynamic_tiles}')
        logger.info(f'[Dataset] use_online_packing: {use_online_packing}')
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        logger.info('Formatting inputs...Skip in lazy mode')
        # assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        self.raw_data = load_dataset("parquet", data_files=meta['annotation'])['train']

        if repeat_time < 1:
            len_partial_data = int(len(self.raw_data) * repeat_time)
            random.seed(10086)  # fixed seed for all ranks
            self.raw_data = self.raw_data.select(
                indices=random.sample(range(len(self.raw_data)), len_partial_data)
            )

        # divide raw_data into N ranks, and only keep the data for current rank
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.raw_data = self.raw_data.select(
            indices=range(self.rank, len(self.raw_data), self.world_size)
        )
        
        gc.collect()
        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_tiles = min_dynamic_tiles
        self.max_dynamic_tiles = max_dynamic_tiles
        self.normalize_type = normalize_type
        self.length = self.raw_data['length']
        self.weights = [1] * len(self.raw_data)

    def __len__(self):
        return len(self.raw_data)

    def multi_modal_get_item(self, data_item):
        # prepare data
        additional_data = data_item['data']
        unified_frame_list = data_item['unified_frame_list']
        pre_num_tiles = data_item['num_tiles_list']
        pre_image_original_size_list = data_item['image_original_size_list']
        pre_image_target_aspect_ratio_list = data_item['image_target_aspect_ratio_list']
        pre_num_all_tiles = data_item['num_all_tiles']
        conversations = data_item['conversations']
        pre_length = data_item['length']
        pre_label_length = data_item['label_length']

        # prepare preprocess function
        preprocess_function = PROCESS_FUNCTIONS[self.template_name]
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                    pad2square=self.pad2square, normalize_type=self.normalize_type)

        if len(unified_frame_list) > 0:
            pil_image_or_frame_list = get_frames_for_multiple_videos_and_images(unified_frame_list, )
            all_tiles = []
            num_tiles_list = []
            for i, pil_image in enumerate(pil_image_or_frame_list):
                tiles = dynamic_preprocess(pil_image, min_num=self.min_dynamic_tiles, max_num=self.max_dynamic_tiles,
                            image_size=self.image_size, use_thumbnail=self.use_thumbnail, target_aspect_ratio=pre_image_target_aspect_ratio_list[i])
                num_tiles = len(tiles)
                assert num_tiles == pre_num_tiles[i]
                all_tiles.extend(tiles)
                num_tiles_list.append(num_tiles)
            num_all_tiles = len(all_tiles)
                
            assert num_all_tiles == pre_num_all_tiles == sum(num_tiles_list), f'num_all_tiles: {num_all_tiles}, pre_num_all_tiles: {pre_num_all_tiles}, sum(num_tiles): {sum(num_tiles)}, image_object_list: {len(image_object_list)}'
            num_tiles_list = np.array(num_tiles_list)
        
            pixel_values = [transform(tile) for tile in all_tiles]
            pixel_values = torch.stack(pixel_values)
            image_flags = torch.tensor([1] * num_all_tiles, dtype=torch.long)
            text_only = False
        else:
            if not self.use_online_packing:
                pixel_values = torch.zeros((1, 3, self.image_size, self.image_size), dtype=torch.float32)
                image_flags = torch.tensor([0], dtype=torch.long)
            else:
                # For packing case, we do not need to add dummy image for each pure-text sample
                # we add dummy image only the whole packed sample does not have any image.
                # see ConcatDatasetForOnlinePacking.get_dummy_image()
                pixel_values = torch.zeros((0, 3, self.image_size, self.image_size), dtype=torch.float32)
                image_flags = torch.tensor([], dtype=torch.long)
            num_tiles_list = 0
            text_only = True
    
        ret = preprocess_function(self.template_name, [deepcopy(conversations)],
            self.tokenizer, self.num_image_token * num_tiles_list, text_only=text_only,
            group_by_length=self.group_by_length, ds_name=self.ds_name, replace_special_tokens=True, placeholder='frame')
        assert ret is not None

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            image_flags=image_flags,
            pixel_values=pixel_values
        )
        # check if pre-computed length is equal to the length of input_ids
        if ret['input_ids'].shape[0] != pre_length:
            logger.info(f'online length not equal to pre-computed length. Dataset: {self.ds_name}, online length: {ret["input_ids"].shape[0]}, pre-computed length: {pre_length}')

        return ret

    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.raw_data)
        
        while True:
            # load data
            ori_data_item = self.raw_data[i]
            data_item = {
                'data': json.loads(ori_data_item['data']),
                'unified_frame_list': json.loads(ori_data_item['unified_frame_list']),
                'image_original_size_list': json.loads(ori_data_item['image_original_size_list']),
                'image_target_aspect_ratio_list': json.loads(ori_data_item['image_target_aspect_ratio_list']),
                'num_tokens_per_image_list': json.loads(ori_data_item['num_tokens_per_image_list']),
                'num_tiles_list': json.loads(ori_data_item['num_tiles_list']),
                'num_all_tiles': ori_data_item['num_all_tiles'],
                'conversations': json.loads(ori_data_item['conversations']),
                'length': ori_data_item['length'],
                'label_length': ori_data_item['label_length'],
            }
            try:
                ret = self.multi_modal_get_item(data_item)
                break
            except Exception as e:
                # traceback.print_exc()
                error_file = e.__traceback__.tb_frame.f_globals['__file__']
                error_file_line = e.__traceback__.tb_lineno
                print(f'Failed to load image, error is {e} of file {error_file}-line {error_file_line}, the dataset is: {self.ds_name}', "data_item is", data_item["conversations"], flush=True)
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_datasets(data_args, tokenizer, tcs_loader, model, group_by_length=False,
                   dynamic_image_size=False, use_thumbnail=False, min_dynamic_tiles=1,
                   max_dynamic_tiles=6, normalize_type='imagenet'):
    datasets = []
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_name in ds_collections.keys():
        repeat_time = ds_collections[ds_name].get('repeat_time', 1)
        if 'max_dynamic_tiles' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_tiles']
            logger.info(f'max_dynamic_tiles is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_tiles
        try:
            dataset = LazySupervisedDataset(
                data_args.conv_style, ds_collections[ds_name],
                tokenizer,
                tcs_loader,
                num_image_token=model.num_image_token,
                image_size=data_args.force_image_size,
                is_train=ds_collections[ds_name].get('data_augment', False),
                pad2square=data_args.pad2square,
                group_by_length=group_by_length,
                dynamic_image_size=dynamic_image_size,
                use_thumbnail=use_thumbnail,
                min_dynamic_tiles=min_dynamic_tiles,
                max_dynamic_tiles=max_num,
                repeat_time=repeat_time,
                normalize_type=normalize_type,
                sample_length_div=data_args.sample_length_div,
                zip_json=data_args.zip_json,
                use_online_packing=data_args.use_online_packing
            )
        except Exception as e:
            logger.info(f'Error in loading dataset: {ds_name}, {e}')
            exit()
        dataset.ds_name = ds_name
        repeat_time = 1 if repeat_time < 1 else repeat_time  # don't repeat if repeat_time is less than 1
        for i in range(repeat_time):
            logger.info(f'Add dataset:{ds_name}_{i} with length: {len(dataset)}')
            datasets.append(dataset)
            if data_args.use_data_resampling:
                lengths.append(math.sqrt(len(dataset)))
            else:
                lengths.append(len(dataset))
    
    train_dataset = ConcatDatasetForOnlinePacking(datasets)
    return train_dataset


def main():
    
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if os.path.exists(osp.join(training_args.output_dir, 'done.txt')):
        logger.info("The training is done since `done.txt` exists in this directory, exit!")
        return
    
    if data_args.use_online_packing:
        patch_packing_attention()
    # from IPython import embed; embed()
    # exit()
    
    if data_args.use_onelogger:
        # pip install --index-url=https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple --upgrade one-logger-utils
        one_logger_callback_utils = create_onelogger_config(training_args, data_args)


    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    if data_args.use_onelogger:
        one_logger_callback_utils.on_model_init_start()

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint_guard(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            logger.info(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
            
    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    num_new_tokens = tokenizer.add_tokens(special_tokens_list, special_tokens=True)
    
    # special case for assistant token
    if len(tokenizer.encode("assistant")) > 1:
        tokenizer.add_tokens(["assistant"], special_tokens=False)
        num_new_tokens += 1
        
    image_token_index = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader =  None 

    if model_args.model_name_or_path is not None:
        logger.info('Loading Eagle2_5_VLForConditionalGeneration...')
        config = Eagle2_5_VLConfig.from_pretrained(model_args.model_name_or_path)
        config._attn_implementation = 'flash_attention_2'
        config._attn_implementation_autoset = False
        config.text_config._attn_implementation = 'flash_attention_2'
        config.vision_config._attn_implementation = 'flash_attention_2'
        logger.info('Using flash_attention_2')
        
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.loss_version = model_args.loss_version
        config.min_dynamic_tiles = data_args.min_dynamic_tiles
        config.max_dynamic_tiles = data_args.max_dynamic_tiles
        config.image_token_index = image_token_index
        model = Eagle2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, 
            torch_dtype=torch.bfloat16, config=config, 
            attn_implementation='flash_attention_2'
            )
        try:
            processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=True)
            processor.tokenizer = tokenizer
        except Exception as e:
            logger.info(f'Error in loading processor: {e}')
            processor = None
    else:
        logger.info("Loading vision backbone(s) from/with {}".format(model_args.vision_path))
        # a hack to support mixture of encoders
        if model_args.vision_path.startswith("MOB:"):
            assert False, "MOB is deprecated, For Eagle2.5 we only support Siglip/Siglip2"
        else:
            vision_config = AutoConfig.from_pretrained(model_args.vision_path, trust_remote_code=True)

            if vision_config.model_type == 'intern_vit_6b':
                logger.info('Loading ViT-6B...')
                vision_config.drop_path_rate = model_args.drop_path_rate
                vision_model = InternVisionModel.from_pretrained(
                    model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
            elif vision_config.model_type == 'siglip':
                logger.info('Loading Siglip...')
                vision_config.vision_config._attn_implementation = 'flash_attention_2'
                vision_model = SiglipVisionModel.from_pretrained(
                    model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config.vision_config )
                vision_config =  vision_config.vision_config  
            elif vision_config.model_type == 'siglip_vision_model':
                logger.info('Loading siglip_vision_model...')
                vision_config._attn_implementation = 'flash_attention_2'
                vision_model = SiglipVisionModel.from_pretrained(
                    model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
            elif vision_config.model_type == 'radio':
                logger.info('Loading radio...')
                vision_model = RADIOModel.from_pretrained(
                    model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config, trust_remote_code=True)
            else:
                raise ValueError(f"Unsupported vision model type: {vision_config.model_type}")
        logger.info('Loading LLM...')
        text_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        text_config._attn_implementation = 'flash_attention_2' 
        logger.info('Using flash_attention_2 for LLM')

        llm = AutoModelForCausalLM.from_pretrained(
            model_args.llm_path, torch_dtype=torch.bfloat16,
            config=text_config, trust_remote_code=True)
        logger.info('Building Eagle2_5_VLConfig...')
        eagle_2_5_vl_config = Eagle2_5_VLConfig(
            vision_config.to_dict(), text_config.to_dict(), downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square, template=data_args.conv_style,
            select_layer=model_args.vision_select_layer, dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail, loss_version=model_args.loss_version,
            min_dynamic_tiles=data_args.min_dynamic_tiles, max_dynamic_tiles=data_args.max_dynamic_tiles,
            image_token_index=image_token_index, use_pixel_shuffle=model_args.use_pixel_shuffle,
            mlp_connector_layers=model_args.mlp_connector_layers)
        eagle_2_5_vl_config.force_image_size = data_args.force_image_size
        eagle_2_5_vl_config._attn_implementation = 'flash_attention_2'
        
        logger.info('Building Eagle2_5_VLForConditionalGeneration...')
        model = Eagle2_5_VLForConditionalGeneration(eagle_2_5_vl_config, vision_model, llm)
        processor = None
        
    model.neftune_alpha = data_args.neftune_alpha
    
    logger.info(model)
    
    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    if data_args.use_onelogger:
        one_logger_callback_utils.on_model_init_end()
        
    if hasattr(model.config.vision_config, 'patch_size'):
        patch_size = model.config.vision_config.patch_size
        logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
        logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
        if hasattr(model.config.vision_config, "image_size") and model.config.vision_config.image_size and model.config.vision_config.image_size != data_args.force_image_size:
            logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
            logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
            model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                    new_size=data_args.force_image_size,
                                                    patch_size=patch_size)
            model.config.vision_config.image_size = data_args.force_image_size
        model.config.force_image_size = data_args.force_image_size
        if model_args.use_pixel_shuffle:
            model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))
        else:
            model.num_image_token = int((data_args.force_image_size // patch_size) ** 2)
    else:
        assert False, "MOB is deprecated, For Eagle2.5 we only support Siglip/Siglip2"

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.text_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True

    if model.config.vision_config.model_type == 'siglip_vision_model':
         model.vision_model.gradient_checkpointing_enable({"use_reentrant": True})
         model.vision_model.vision_model.encoder.gradient_checkpointing = True
         
    elif model.config.vision_config.model_type == 'MOB' or model.config.vision_config.model_type == 'radio':
        pass
    else:
        model.vision_model.encoder.gradient_checkpointing = True

    if model_args.grad_checkpoint:
       model.language_model._set_gradient_checkpointing()
    logger.info("model init done")

    logger.info("setting sequence parallelism")
    # set sequence parallelism
    if data_args.sequence_parallel_degree > 1:
        logger.info("setting pg manager")
        set_pg_manager(model, data_args.sequence_parallel_degree)
        logger.info(f'Sequence parallelism is enabled, SP = {data_args.sequence_parallel_degree}')

    # print information of all ranks
    logger.info("multi-node distribution")
    hostnames = [None] * dist.get_world_size()
    if get_pg_manager() is not None:
        local_info = f"sp_rank: {get_pg_manager().sequence_parallel_rank}, ulysses_rank: {get_pg_manager().ulysses_sequence_parallel_rank}, ring_rank: {get_pg_manager().ring_sequence_parallel_rank}, host-name: {socket.gethostname()}"
    else:
        local_info = f"sp_rank: None, ulysses_rank: None, ring_rank: None, host-name: {socket.gethostname()}"
    dist.all_gather_object(hostnames, local_info)
    hostnames[dist.get_rank()] = local_info
    
    if dist.get_rank() == 0:
        for i, info in enumerate(hostnames):
            logger.info(f"global rank[{i}], {info}")

    logger.info("starting build train dataset")
    train_dataset = build_datasets(
        data_args, tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_tiles=data_args.min_dynamic_tiles, max_dynamic_tiles=data_args.max_dynamic_tiles,
        normalize_type=data_args.normalize_type)
    logger.info("build train dataset done")

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    if data_args.use_online_packing:
        replace_train_sampler_for_online_packing()
    else:
        replace_train_sampler()
    
    # do we need default_data_collator?
    my_callbacks = [SaveCheckpointCallback(initial_interval_hours=model_args.save_every_n_hours, save_interval_minutes=5)] if model_args.save_every_n_hours > 0 else []
    my_callbacks.append(MemoryLoggerCallback())
    if data_args.use_onelogger:
        CustomTrainer = warp_onelogger_trainer(one_logger_callback_utils)
    else:
        CustomTrainer = Trainer

    if processor is not None:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=None,
            data_collator=get_collator(),
            callbacks=my_callbacks,
            processing_class=processor
        )
    else:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=get_collator(),
            callbacks=my_callbacks,
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics['train_samples'] = len(train_dataset)

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
    with open(osp.join(training_args.output_dir, 'done.txt'), 'w') as f:
        f.write('done: ' + time.ctime())

if __name__ == '__main__':
    main()