import io

from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
from typing import Dict, Union
import numpy as np
import torch
import torchvision.transforms as T
import transformers
from eaglevl.conversation import get_conv_template
from eaglevl.patch.train_sampler_patch import Packer
from PIL import Image
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from torchvision.transforms.functional import InterpolationMode
import os
import bisect
from eaglevl.train.constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD)



try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
except ImportError as E:
    print('please install petrel_client')
import sys


def replace_special_tokens(text, special_tokens):
    for special_token in special_tokens:
        text = text.replace(special_token, '')
    return text


class ConcatDatasetForOnlinePacking(ConcatDataset):

    def __getitem_for_int_idx__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][sample_idx]
    
    def __get_raw_data_for_int_idx__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx].get_raw_data(sample_idx)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.__getitem_for_int_idx__(idx)
        elif isinstance(idx, Packer):
            idx_list = idx.items
            ret_list = [self.__getitem_for_int_idx__(index) for index in idx_list]
            return self.pack_data(ret_list)
        else:
            raise ValueError(f'Unsupported index type: {type(idx)}')

    def get_raw_data(self, idx):
        if isinstance(idx, Packer):
            return [self.__get_raw_data_for_int_idx__(index) for index in idx.items]
        else:
            raise ValueError(f'Unsupported index type: {type(idx)}')

    def get_dummy_image(self):
        image = Image.new('RGB', (self.datasets[0].image_size, self.datasets[0].image_size), (255, 255, 255))
        images = dynamic_preprocess(image, min_num=self.datasets[0].min_dynamic_tiles, max_num=self.datasets[0].max_dynamic_tiles,
                                    image_size=self.datasets[0].image_size, use_thumbnail=self.datasets[0].use_thumbnail)
        transform = build_transform(is_train=self.datasets[0].is_train, input_size=self.datasets[0].image_size,
                                    pad2square=self.datasets[0].pad2square, normalize_type=self.datasets[0].normalize_type)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_tiles = pixel_values.size(0)
        return pixel_values

    def pack_data(self, ret_list):
        try:
            input_ids=torch.cat([each['input_ids'] for each in ret_list], dim=0)
        except:
            for each in ret_list:
                print('input_ids.shape', each['input_ids'].shape)
            raise ValueError
        labels=torch.cat([each['labels'] for each in ret_list], dim=0)
        pixel_values = torch.cat([each['pixel_values'] for each in ret_list], dim=0)
        image_flags = torch.cat([each['image_flags'] for each in ret_list], dim=0)
        sub_sample_lengths = [each['input_ids'].shape[0] for each in ret_list]
        attention_masks = []
        for i, ret in enumerate(ret_list):
            attention_masks += [i] * ret['input_ids'].size(0)  # start from 0
        attention_masks = torch.tensor(attention_masks, device=pixel_values.device)
        model_max_length = self.datasets[0].tokenizer.model_max_length

        if len(attention_masks) < model_max_length:
            pad_length = model_max_length - len(input_ids)
            pad_input_ids =  torch.tensor([self.datasets[0].tokenizer.pad_token_id] * pad_length)
            pad_labels =  torch.tensor([IGNORE_TOKEN_ID] * pad_length)
            pad_attention_masks = torch.tensor([len(ret_list)] * pad_length)
            input_ids = torch.cat([input_ids, pad_input_ids], dim = 0)
            labels = torch.cat([labels, pad_labels], dim = 0)
            attention_masks = torch.cat([attention_masks, pad_attention_masks], dim = 0)
            sub_sample_lengths.append(model_max_length-sum(sub_sample_lengths))
        elif len(attention_masks) > model_max_length and len(ret_list) == 1:
            print(f'warning: len(attention_masks) > model_max_length, {len(attention_masks)} > {model_max_length}', flush=True)
            input_ids = input_ids[:model_max_length]
            labels = labels[:model_max_length]
            attention_masks = attention_masks[:model_max_length]
            sub_sample_lengths = [model_max_length]
        elif len(attention_masks) > model_max_length and len(ret_list) > 1:
            return self.pack_data(ret_list[:-1])

        sub_sample_lengths = torch.tensor(sub_sample_lengths)
        
        if pixel_values.size(0)==0:
            pixel_values = self.get_dummy_image()
            image_flags = torch.tensor([0])
        assert attention_masks is not None, f'attention_masks is None, input_ids.shape={input_ids.shape}, sub_sample_lengths={sub_sample_lengths}, image_flags={image_flags}'
        new_ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_masks,
            pixel_values=pixel_values,
            image_flags=image_flags,
            sub_sample_lengths=sub_sample_lengths,
        )
        return new_ret

class ConcatDatasetForOnlinePacking_AnyRes(ConcatDataset):

    def __getitem_for_int_idx__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][sample_idx]
    
    def __get_raw_data_for_int_idx__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx].get_raw_data(sample_idx)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.__getitem_for_int_idx__(idx)
        elif isinstance(idx, Packer):
            idx_list = idx.items
            ret_list = [self.__getitem_for_int_idx__(index) for index in idx_list]
            assert len(ret_list) == len(idx_list) and len(ret_list) > 0, f'len(ret_list): {len(ret_list)}, len(idx_list): {len(idx_list)}, idx_list: {idx_list}'
            return self.pack_data(ret_list)
        else:
            raise ValueError(f'Unsupported index type: {type(idx)}')

    def get_raw_data(self, idx):
        if isinstance(idx, Packer):
            return [self.__get_raw_data_for_int_idx__(index) for index in idx.items]
        else:
            raise ValueError(f'Unsupported index type: {type(idx)}')

    def get_dummy_image(self):
        pixel_values = torch.zeros(1, 3, 28, 28)
        return pixel_values

    def pack_data(self, ret_list):
        try:
            input_ids=torch.cat([each['input_ids'] for each in ret_list], dim=0)
        except:
            for each in ret_list:
                print('input_ids.shape', each['input_ids'].shape)
            raise ValueError
        labels=torch.cat([each['labels'] for each in ret_list], dim=0)
        pixel_values = [image for each in ret_list for image in each['pixel_values']]
        image_flags = torch.cat([torch.tensor(each['image_flags']) for each in ret_list], dim=0)
        sub_sample_lengths = [each['input_ids'].shape[0] for each in ret_list]
        attention_masks = []
        for i, ret in enumerate(ret_list):
            attention_masks += [i] * ret['input_ids'].size(0)  # start from 0
        attention_masks = torch.tensor(attention_masks, device=labels.device)
        model_max_length = self.datasets[0].processor.tokenizer.model_max_length

        if len(attention_masks) < model_max_length:
            pad_length = model_max_length - len(input_ids)
            pad_input_ids =  torch.tensor([self.datasets[0].processor.tokenizer.pad_token_id] * pad_length)
            pad_labels =  torch.tensor([IGNORE_TOKEN_ID] * pad_length)
            pad_attention_masks = torch.tensor([len(ret_list)] * pad_length)
            input_ids = torch.cat([input_ids, pad_input_ids], dim = 0)
            labels = torch.cat([labels, pad_labels], dim = 0)
            attention_masks = torch.cat([attention_masks, pad_attention_masks], dim = 0)
            sub_sample_lengths.append(model_max_length-sum(sub_sample_lengths))
        elif len(attention_masks) > model_max_length and len(ret_list) == 1:
            print(f'warning: len(attention_masks) > model_max_length, {len(attention_masks)} > {model_max_length}', flush=True)
            # TODO: this should not happen for one single sample case, but it did happen, need to check why
            len_sample = len(input_ids)
            input_ids = input_ids[len_sample-model_max_length:]
            labels = labels[len_sample-model_max_length:]
            attention_masks = attention_masks[len_sample-model_max_length:]
        elif len(attention_masks) > model_max_length and len(ret_list) > 1:
            return self.pack_data(ret_list[:-1])

        sub_sample_lengths = torch.tensor(sub_sample_lengths)
        
        if len(pixel_values)==0:
            pixel_values = [self.get_dummy_image()]
            image_flags = torch.tensor([0])
        assert attention_masks is not None, f'attention_masks is None, input_ids.shape={input_ids.shape}, sub_sample_lengths={sub_sample_lengths}, image_flags={image_flags}'
        new_ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_masks,
            pixel_values=pixel_values,
            image_flags=image_flags,
            sub_sample_lengths=sub_sample_lengths,
        )
        return new_ret
    

class WeightedConcatDataset(ConcatDataset):
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = torch.DoubleTensor(weights)
        self.total_size = sum(len(d) for d in datasets)
        self.sampler = WeightedRandomSampler(weights=self.weights, num_samples=self.total_size, replacement=True)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_size


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert('RGB')


class TCSLoader(object):

    def __init__(self, conf_path, sc_config_key='sensecore'):
        print(f'[TCSLoader] config_path: {conf_path}')
        print('--> before Client(conf_path)')
        self.client = Client(conf_path)
        self.sc_config_key = sc_config_key
        print('--> after Client(conf_path)')

    def __call__(self, fn):
        img_value_str = self.client.get(fn)
        img = pil_loader(img_value_str)
        return img


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg
    return jpeg_degrade


# Define the JPEG compression quality range, pre-create all JPEG compression functions
qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}


def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])

    return transform


def preprocess(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token: int,
        text_only: bool = False,
        group_by_length: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        replace_special_tokens=True,
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
    if not text_only:
        new_conversations = []
        for conversation in conversations:
            conversation = conversation.replace('<image>', image_tokens, num_image)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ': '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == '':
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            logger.info(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_mpt(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token: Union[int, np.ndarray],
        text_only: bool = False,
        group_by_length: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        replace_special_tokens=True,
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            # break the pad token '<s>' -> s> # TODO: how about <<s>
            sentence['value'] = str(sentence['value']).strip()
            if replace_special_tokens: 
                sentence['value'] = sentence['value'].replace(tokenizer.pad_token, tokenizer.pad_token[1:]) 
            if sentence['value'][0] == '\n':
                sentence['value'] = sentence['value'][1:]
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    if type(num_image_token) == int:
        image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
        if not text_only:
            new_conversations = []
            for conversation in conversations:
                conversation = conversation.replace('<image>', image_tokens, num_image)
                new_conversations.append(conversation)
            conversations = new_conversations
    elif type(num_image_token) == np.ndarray:
        image_tokens_list = []
        new_conversations = []
        for idx, num_token_per_image in enumerate(num_image_token):
            image_tokens_list.append(f'<image {idx+1}>{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * int(num_token_per_image)}{IMG_END_TOKEN}')
            
        for conversation in conversations:
            for idx, image_tokens in enumerate(image_tokens_list):
                conversation = conversation.replace(f'<image-{idx+1}>', image_tokens, num_image)
            new_conversations.append(conversation)
        conversations = new_conversations
    else:
        assert False, 'not support'

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    
    # if input_ids[0].size(0)<256: print(conversations)
    
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1]  # <|im_end|><|im_start|>assistant\n
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]  # system + user + gpt
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(re_turns):
            if turn == '':
                break
            turn_len = len(tokenizer(turn).input_ids) + 1
            if turn_len>=tokenizer.model_max_length:
                return None
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer(parts[0]).input_ids)
            if instruction_len>=tokenizer.model_max_length:
                return None

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            # print(f'[question {i}]', tokenizer.decode(input_ids[:, cur_len: cur_len + instruction_len][0]))
            # print(f'[answer {i}]', tokenizer.decode(input_ids[:, cur_len + instruction_len: cur_len + turn_len][0]))
            # print(f'[label {i}]', target[cur_len + instruction_len: cur_len + turn_len])
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_phi3(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token: int,
        text_only: bool = False,
        group_by_length: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        replace_special_tokens=True,
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            if replace_special_tokens: 
                sentence['value'] = str(sentence['value']).replace(tokenizer.pad_token, tokenizer.pad_token[1:]) 
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
    if not text_only:
        new_conversations = []
        for conversation in conversations:
            conversation = conversation.replace('<image>', image_tokens, num_image)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    tokenizer.padding_side = 'right'
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1]  # <|end|>\n<|assistant|>
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(int(tokenizer.pad_token_id)).sum())

        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]  # system + user + gpt
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        endoftext_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        target[target == endoftext_id] = IGNORE_TOKEN_ID

        for i, turn in enumerate(re_turns):
            if turn == '':
                break
            if i == 0:
                turn_len = len(tokenizer(turn).input_ids)
            else:
                turn_len = len(tokenizer(turn).input_ids) - 1
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if i == 0:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            else:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            # print(f'[question {i}]', tokenizer.decode(input_ids[:, cur_len: cur_len + instruction_len][0]))
            # print(f'[answer {i}]', tokenizer.decode(input_ids[:, cur_len + instruction_len: cur_len + turn_len][0]))
            # print(f'[label {i}]', target[cur_len + instruction_len: cur_len + turn_len])
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )




def preprocess_qwen2(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token: Union[int, np.ndarray],
        text_only: bool = False,
        group_by_length: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        replace_special_tokens=True,
        placeholder: str = 'image',
) -> Dict:
    
    conversations = []
    roles = dict(
        human='user',
        gpt='assistant',
        observation='observation',
    )
    for source in sources:
        if source[0]['from'] == 'system':
            system_prompt = source[0]['value']
            source = source[1:]
        else:
            system_prompt = "You are an AI assistant whose name is Eagle-Next."
        conversation = [
            {"role": "system", "content": system_prompt},
        ]
        for i, msg in enumerate(source):
            msg['value'] = str(msg['value']).rstrip().lstrip()
 
            conversation.append(
                {
                    'role':roles[msg['from']],
                    'content': msg['value']
                }
            ) 
        conversations.append(conversation)

    qwen2_chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    conversations = [tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, chat_template=qwen2_chat_template,
    ) for conversation in conversations]
    num_image_token_pre_compute = 0
    for i, conversation in enumerate(conversations):
        if type(num_image_token) == int:
            if not text_only:
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
                conversations[i] = conversations[i].replace(f'<{placeholder}>', image_tokens, num_image)
                num_image_token_pre_compute += num_image_token
        elif type(num_image_token) == np.ndarray:
            # num_image_token is a numpy array, which contains the number of image tokens for each image
            num_image_token_list = num_image_token.tolist()
            image_tokens_list = []
            for idx, num_token_per_image in enumerate(num_image_token_list):
                image_tokens_list.append(f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * int(num_token_per_image)}{IMG_END_TOKEN}')
            for idx, image_tokens in enumerate(image_tokens_list):
                if not text_only:
                    conversations[i] = conversations[i].replace(f'<{placeholder}-{idx+1}>', image_tokens, num_image)
                    num_image_token_pre_compute += num_image_token_list[idx]

    # Tokenize conversations
    group_by_length = True
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    IMG_CONTEXT_TOKEN_ID = tokenizer.encode(IMG_CONTEXT_TOKEN)[0]
    assert num_image_token_pre_compute == ((input_ids==IMG_CONTEXT_TOKEN_ID).sum()), f'Precompute image token number: {num_image_token_pre_compute} vs. Actual: {(input_ids==IMG_CONTEXT_TOKEN_ID).sum()}'
    
    targets_flag = input_ids.clone() * 0

    start_header_idxs = torch.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|im_start|>")
    )[1]
    assistant_idxs = torch.where(
        input_ids == tokenizer.convert_tokens_to_ids("assistant")
    )[1]
    eot_idxs = torch.where(
        input_ids == tokenizer.convert_tokens_to_ids('<|im_end|>'))[1]

    # context = np.ones_like(input_ids, dtype=np.int8)
    assert targets_flag.size(0) == 1
    
    for assistant_idx in assistant_idxs:
        sets = list(set(start_header_idxs + 1))
        sets = [each.item() for each in sets]
        if assistant_idx.item() in sets:
            st = assistant_idx + 1  # assistant\n
            for eot_idx in eot_idxs:
                if eot_idx > st:
                    targets_flag[:, st+1: eot_idx + 1] = 1
                    break
    targets = input_ids.clone()
    assert targets_flag.sum() > 0, 'should train some label'
    targets[targets_flag==0] = IGNORE_TOKEN_ID
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_nm5(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token: Union[int, np.ndarray],
        text_only: bool = False,
        group_by_length: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        replace_special_tokens=True,
) -> Dict:
    
    conversations = []
    roles = dict(
        human='user',
        gpt='assistant'
    )
    for source in sources:
        conversation = [
            {"role": "system", "content": "You are an AI assistant whose name is Eagle-Next."},
        ]
        for i, msg in enumerate(source):
            msg['value'] = str(msg['value']).strip()
            if msg['value'][0] == '\n':
                msg['value'] = msg['value'][1:] 
            conversation.append(
                {
                    'role':roles[msg['from']],
                    'content': msg['value']
                }
            ) 
        conversations.append(conversation)
             
    nm5_chat_template =  "{{'<SPECIAL_10>System\\\\n'}}{% for message in messages %}{% if message['role'] == 'system' %}{{message['content'].strip()}}{% endif %}{% endfor %}{{'\\\\n'}}{% for message in messages %}{% if message['role'] == 'user' %}{{ '<SPECIAL_11>User\\\\n' + message['content'].strip() + '\\\\n<SPECIAL_11>Assistant\\\\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'].strip() }}{% endif %}{% endfor %}"
    conversations = [tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, chat_template=nm5_chat_template,
    ) for conversation in conversations]
    num_image_token_pre_compute = 0
    for i, conversation in enumerate(conversations):
        if type(num_image_token) == int:
            if not text_only:
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
                conversations[i] = conversations[i].replace('<image>', image_tokens, num_image)
                num_image_token_pre_compute += num_image_token
        elif type(num_image_token) == np.ndarray:
            image_tokens_list = []
            for idx, num_token_per_image in enumerate(num_image_token):
                image_tokens_list.append(f'<image {idx+1}>{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * int(num_token_per_image)}{IMG_END_TOKEN}')
            for idx, image_tokens in enumerate(image_tokens_list):
                if not text_only:
                    conversations[i] = conversations[i].replace(f'<image-{idx+1}>', image_tokens, num_image)
                    num_image_token_pre_compute += num_image_token[idx]

    # Tokenize conversations
    group_by_length = True
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    IMG_CONTEXT_TOKEN_ID = tokenizer.encode(IMG_CONTEXT_TOKEN)[-1]

    assert num_image_token_pre_compute == ((input_ids==IMG_CONTEXT_TOKEN_ID).sum())
    
    targets_flag = input_ids.clone() * 0

    
    
    start_header_idxs = torch.where(input_ids == tokenizer.convert_tokens_to_ids("<SPECIAL_11>"))[1]
    assistant_idxs = torch.where(input_ids == tokenizer.convert_tokens_to_ids("Assistant"))[1]
    # print((input_ids == tokenizer.convert_tokens_to_ids("Assistant")).sum())
    eot_idxs = [idx.item()-1 for idx in start_header_idxs]
    eot_idxs.append(input_ids.size(1)-1)
    

    # context = np.ones_like(input_ids, dtype=np.int8)
    assert targets_flag.size(0) == 1
    
    for assistant_idx in assistant_idxs:
        sets = list(set(start_header_idxs + 1))
        sets = [each.item() for each in sets]
        # print(assistant_idx.item(), sets)
        if assistant_idx.item() in sets:
            st = assistant_idx + 1  # Assistant\n
            for eot_idx in eot_idxs:
                if eot_idx > st:
    
                    targets_flag[:, st: eot_idx + 1] = 1
                    break
    # from IPython import embed; embed(); exit()
    targets = input_ids.clone()
    assert targets_flag.sum() > 0, 'should train some label'
    targets[targets_flag==0] = IGNORE_TOKEN_ID
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
    

def preprocess_llama3(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token: Union[int, np.ndarray],
        text_only: bool = False,
        group_by_length: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        per_tile_token: int = 256,
        replace_special_tokens=True,
        tile_tag = False
) -> Dict:
    
    conversations = []
    roles = dict(
        human='user',
        gpt='assistant'
    )
    for source in sources:
        if source[0]['from'] == 'system':
            system_prompt = source[0]['value']
            source = source[1:]
        else:
            system_prompt = "You are an AI assistant whose name is Eagle-Next."
        conversation = [
            {"role": "system", "content": system_prompt},
        ]
        for i, msg in enumerate(source):
            msg['value'] = str(msg['value']).strip()
            if msg['value'][0] == '\n':
                msg['value'] = msg['value'][1:] 
            conversation.append(
                {
                    'role':roles[msg['from']],
                    'content': msg['value']
                }
            ) 
        conversations.append(conversation)
             
    llama3_chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"
    conversations = [tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, chat_template=llama3_chat_template,
    ) for conversation in conversations]
    num_image_token_pre_compute = 0
    for i, conversation in enumerate(conversations):
        if type(num_image_token) == int:
            if not text_only:
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
                conversations[i] = conversations[i].replace('<image>', image_tokens, num_image)
                num_image_token_pre_compute += num_image_token
        elif type(num_image_token) == np.ndarray:
            image_tokens_list = []
            for idx, num_token_per_image in enumerate(num_image_token):
                image_tokens_list.append(f'<image {idx+1}>{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * int(num_token_per_image)}{IMG_END_TOKEN}')
            for idx, image_tokens in enumerate(image_tokens_list):
                if not text_only:
                    conversations[i] = conversations[i].replace(f'<image-{idx+1}>', image_tokens, num_image)
                    num_image_token_pre_compute += num_image_token[idx]

    # Tokenize conversations
    tokenizer.padding_side = 'right'
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    IMG_CONTEXT_TOKEN_ID = tokenizer.encode(IMG_CONTEXT_TOKEN)[-1]
    
    assert num_image_token_pre_compute == ((input_ids==IMG_CONTEXT_TOKEN_ID).sum())
    
    targets_flag = input_ids.clone() * 0

    start_header_idxs = torch.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    )[1]
    assistant_idxs = torch.where(
        input_ids == tokenizer.convert_tokens_to_ids("assistant")
    )[1]
    end_header_idxs = torch.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    )[1]
    eot_idxs = torch.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|eot_id|>"))[1]

    # context = np.ones_like(input_ids, dtype=np.int8)
    assert targets_flag.size(0) == 1
    
    for assistant_idx in assistant_idxs:
        sets = list(set((start_header_idxs + end_header_idxs) // 2))
        sets = [each.item() for each in sets]
        if assistant_idx.item() in sets:
            st = assistant_idx + 3  # assistant<|end_header_id|>\n\n
            for eot_idx in eot_idxs:
                if eot_idx > st:
                    targets_flag[:, st+1: eot_idx + 1] = 1
                    break
    targets = input_ids.clone()
    if targets_flag.sum() == 0:
        targets_flag[:, st:] = 1
        
    assert targets_flag.sum() > 0, 'should train some label'
    targets[targets_flag==0] = IGNORE_TOKEN_ID
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
    
def preprocess_internlm(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token: Union[int, np.ndarray],
        text_only: bool = False,
        group_by_length: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        replace_special_tokens=True,
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'

            sentence['value'] = str(sentence['value']).strip()
            # break the pad token '<s>' -> s> # TODO: how about <<s>
            if replace_special_tokens: 
                sentence['value'] = sentence['value'].replace(tokenizer.pad_token, tokenizer.pad_token[1:]) 
            if sentence['value'][0] == '\n':
                sentence['value'] = sentence['value'][1:]
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())
    num_image_token_pre_compute = 0
    if type(num_image_token) == int:
        image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
        if not text_only:
            new_conversations = []
            for conversation in conversations:    
                conversation = conversation.replace('<image>', image_tokens, num_image)
                num_image_token_pre_compute += num_image_token
                new_conversations.append(conversation)
            conversations = new_conversations
            
    elif type(num_image_token) == np.ndarray:
        image_tokens_list = []
        new_conversations = []
        for idx, num_token_per_image in enumerate(num_image_token):
            image_tokens_list.append(f'<image {idx+1}>{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * int(num_token_per_image)}{IMG_END_TOKEN}')
            
        for conversation in conversations:
            for idx, image_tokens in enumerate(image_tokens_list):
                conversation = conversation.replace(f'<image-{idx+1}>', image_tokens, num_image)
                num_image_token_pre_compute += int(num_image_token[idx])
            new_conversations.append(conversation)
            
        conversations = new_conversations
    else:
        assert False, 'not support'
    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    IMG_CONTEXT_TOKEN_ID = tokenizer.encode(IMG_CONTEXT_TOKEN)[1]
    assert num_image_token_pre_compute == ((input_ids==IMG_CONTEXT_TOKEN_ID).sum())
    targets = input_ids.clone()

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 浦语里面 pad_token_id = eos_token_id
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID  # <s>
        parts = conversation.split(conv.roles[1])  # [UNUSED_TOKEN_146]assistant\n
        info = parts[0] + conv.roles[1]
        temp_len = len(tokenizer(info).input_ids) - 1  # 去除tokenizer的<s>
        if temp_len>=tokenizer.model_max_length:
            return None
        target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
        cur_len = cur_len + temp_len

        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            temp_len = len(tokenizer(part1).input_ids) - 1
            if temp_len>=tokenizer.model_max_length:
                return None
            cur_len = cur_len + temp_len
            part = conv.roles[0] + part2 + conv.roles[1]
            temp_len = len(tokenizer(part).input_ids) - 1
            if temp_len>=tokenizer.model_max_length:
                return None
            target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
            cur_len = cur_len + temp_len
        last_info = parts[-1]
        temp_len = len(tokenizer(last_info).input_ids) - 1
        if temp_len>=tokenizer.model_max_length:
                return None
        cur_len = cur_len + temp_len

        target[cur_len:] = IGNORE_TOKEN_ID
        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}. This dataset is {ds_name}.')
                sys.stdout.flush()
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio

def find_closest_aspect_ratio_v2(aspect_ratio, target_ratios, width, height, image_size):
    """
    previous version mainly foucs on ratio.
    We also consider area ratio here.
    """
    best_factor = float('-inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        area_ratio = (ratio[0]*ratio[1]*image_size*image_size)/ area
        
        """
        new area > 60% of original image area is enough.
        """
        factor_based_on_area_n_ratio = min((ratio[0]*ratio[1]*image_size*image_size)/ area, 0.6)* \
                                     min(target_aspect_ratio/aspect_ratio, aspect_ratio/target_aspect_ratio)
        
        if factor_based_on_area_n_ratio > best_factor:
            best_factor = factor_based_on_area_n_ratio
            best_ratio = ratio
        
    return best_ratio



# we can specify the target aspect ratio here.
def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False, target_aspect_ratio=None):
    orig_width, orig_height = image.size

    if os.environ.get('RESIZE_IMAGE_TO_HALF', False):
        image = image.resize((orig_width//2, orig_height//2))
        orig_width, orig_height = image.size
        
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    if target_aspect_ratio is None:
        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio_v2(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

PROCESS_FUNCTIONS = {
    'internlm-chat': preprocess_internlm,
    'llama3-chat': preprocess_llama3,
    'qwen2-chat': preprocess_qwen2,
    'qwen3-chat': preprocess_qwen2,
    'nm5-chat': preprocess_nm5,
    'phi3-chat': preprocess_phi3,
    "smollm2-chat": preprocess_qwen2, # smollm2 has the same conv template as qwen2-chat
}

if __name__ == '__main__':
    from PIL import Image
    image = Image.new('RGB', (224, 224), (255, 255, 255))
    images = dynamic_preprocess(image, min_num=1, max_num=12,
                                    image_size=448, use_thumbnail=True)
    print(len(images))