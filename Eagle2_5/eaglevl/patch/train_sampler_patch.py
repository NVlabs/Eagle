from typing import List, Optional, Sized, Iterator
import torch
import math
import transformers
from typing import List, Optional, Sized, Iterator

from torch.utils.data import Sampler
from torch.utils.data import Dataset, RandomSampler
import torch.distributed as dist

import accelerate
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer import (LengthGroupedSampler, RandomSampler,
                                  has_length, logger)
from tqdm import tqdm
import numpy as np
import random

from eaglevl.sp_utils import get_pg_manager


# copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L38
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float('inf')

    return chunks


# copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L88
def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


# Packer is a list that also supports comparison with integers
class Packer:
    def __init__(self, items):
        self.items = items
        self.total = 0

    def __len__(self):
        return len(self.items)

    def append(self, item):
        self.items.append(item)
        self.total += item

    def __lt__(self, other):
        if isinstance(other, int):
            return self.total < other
        elif isinstance(other, Packer):
            return self.total < other.total
        else:
            raise TypeError("Unsupported operand type for <: '{}' and '{}'".format(type(self).__name__, type(other).__name__))

    def __gt__(self, other):
        if isinstance(other, int):
            return self.total > other
        elif isinstance(other, Packer):
            return self.total > other.total
        else:
            raise TypeError("Unsupported operand type for >: '{}' and '{}'".format(type(self).__name__, type(other).__name__))

    def __eq__(self, other):
        if isinstance(other, int):
            return self.total == other
        elif isinstance(other, Packer):
            return self.total == other.total
        else:
            return False

    def __iter__(self):
        return iter(self.items)
    

def split_to_even_chunks_for_online_packing(indices, lengths, model_max_length=8192, label_lengths=None, P=32):

    indices = [indices[i] for i in range(len(indices)) if lengths[indices[i]] <= model_max_length]
  
      
    # assert len(indices) % P == 0
    bin_size = len(indices) // P
    bins = [indices[i*bin_size : (i+1)*bin_size] for i in range(P)]
    # if label_lengths is not None:
    #     label_bins = [label_lengths[i*bin_size : (i+1)*bin_size] for i in range(P)]
    # Handle any remaining samples
    remainder = len(indices) % P
    if remainder:
        bins[-1].extend(indices[-remainder:])
        if label_lengths is not None:
            label_bins[-1].extend(label_lengths[-remainder:])
    
    total_length = sum([lengths[i] for i in indices])
    min_packages = (total_length + model_max_length - 1) // model_max_length + 20 # Round up division
    min_packages = int(min_packages)
    # Step 4: Initialize packages
    packages = [[] for _ in range(min_packages)]
    package_lengths = [0] * min_packages
    
    package_index = 0
    for bin in bins: 
        sample_index = 0
        count = 0
        while sample_index < len(bin):
            length = lengths[bin[sample_index]]
            # Try to fit the sample into the current package
            if package_lengths[package_index] + length <= model_max_length:
                packages[package_index].append(bin[sample_index])
                package_lengths[package_index] += length
                sample_index += 1
            else:
                packages.append([])
                package_lengths.append(0)
            package_index = np.argmin(package_lengths)

    return [Packer(pack) for pack in packages if len(pack) > 0]



def split_to_even_chunks_for_online_packing_heap(indices, lengths, model_max_length=8192, label_lengths=None, P=32):
    """
    Optimized version of split_to_even_chunks_for_online_packing for better performance with large indices.
    """
    # Filter indices that are too long
    indices = [idx for idx in indices if lengths[idx] <= model_max_length]
    
    # Divide indices into bins
    bin_size = len(indices) // P
    bins = [indices[i*bin_size : (i+1)*bin_size] for i in range(P)]
    
    # Handle any remaining samples
    remainder = len(indices) % P
    if remainder:
        bins[-1].extend(indices[-remainder:])
    
    # Calculate minimum packages needed
    total_length = sum(lengths[i] for i in indices)
    min_packages = (total_length + model_max_length - 1) // model_max_length + 20  # Add some buffer
    min_packages = int(min_packages)
    
    # Initialize packages
    packages = [[] for _ in range(min_packages)]
    package_lengths = [0] * min_packages
    
    # Use a min heap (priority queue) for finding the package with minimum length
    import heapq
    package_heap = [(0, i) for i in range(min_packages)]
    heapq.heapify(package_heap)
    
    # Process in larger chunks for progress bar efficiency
    total_samples = len(indices)
    with tqdm(total=total_samples, desc=f'split_to_even_chunks_for_online_packing, total:{total_samples}') as bar:
        for bin_idx, bin_samples in enumerate(bins):
            # Update progress bar less frequently
            update_freq = max(1, len(bin_samples) // 10)
            samples_processed = 0
            
            for sample_idx, idx in enumerate(bin_samples):
                length = lengths[idx]
                
                # Get the package with minimum current length
                current_length, package_idx = heapq.heappop(package_heap)
                
                # If it fits, add it
                if current_length + length <= model_max_length:
                    packages[package_idx].append(idx)
                    new_length = current_length + length
                    heapq.heappush(package_heap, (new_length, package_idx))
                else:
                    # If it doesn't fit, put back the package and create a new one
                    heapq.heappush(package_heap, (current_length, package_idx))
                    
                    # Create a new package
                    new_package_idx = len(packages)
                    packages.append([idx])
                    heapq.heappush(package_heap, (length, new_package_idx))
                
                samples_processed += 1
                # Update progress bar less frequently to reduce overhead
                if samples_processed % update_freq == 0:
                    bar.update(update_freq)
            
            # Update any remaining progress
            remaining = samples_processed % update_freq
            if remaining > 0:
                bar.update(remaining)
    
    return [Packer(pack) for pack in packages if len(pack) > 0]

def gather_tensor_list(tensor):
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    lengths = [torch.tensor(0, device=tensor.device, dtype=torch.int32) for _ in range(world_size)]
    # collect lengths
    dist.all_gather(lengths, torch.tensor(tensor.size(0), device=tensor.device, dtype=torch.int32))
    tensor_list = [torch.zeros(lengths[i], device=tensor.device, dtype=tensor.dtype) for i in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    tensor_list[dist.get_rank()] = tensor
    tensor = torch.cat(tensor_list, dim=0)
    return tensor

def weighted_shuffle_np(elements, weights):
    elements = np.array(elements)
    weights = np.array(weights)
    U = np.random.uniform(0, 1, size=len(elements))
    s = U ** (1 / weights)
    idx = np.argsort(s)
    shuffled_elements = elements[idx]
    return shuffled_elements


def get_length_grouped_indices_for_online_packing(lengths, batch_size, world_size, generator=None, merge=True, accumulation_steps=1, megabatch_size=4000, model_max_length=8192, weights=None, default_rank=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    rank = torch.distributed.get_rank() if default_rank is None else default_rank
    device = torch.cuda.current_device()

    if max(weights) > 1:
        indices = np.arange(len(lengths))
        # local_indices = indices[rank:: world_size]
        # local_weights = weights[rank:: world_size]
        local_indices = indices
        local_weights = weights
        local_indices = weighted_shuffle_np(local_indices, local_weights)
    else:
        indices = torch.randperm(len(lengths), generator=generator)
        local_indices = indices
        # local_indices = indices[rank:: world_size]
    if megabatch_size == -1: megabatch_size = len(local_indices)
    megabatches = [local_indices[i : i + megabatch_size].tolist() for i in range(0, len(local_indices), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks_for_online_packing_heap(megabatch, lengths, model_max_length=model_max_length) for megabatch in tqdm(megabatches, desc='split_to_even_chunks_for_online_packing')]

    iter_batch_size = batch_size // accumulation_steps
    local_megabatches = [batch for megabatch in megabatches for batch in megabatch]
    
    if default_rank is None:
        len_local_megabatches = torch.tensor(len(local_megabatches), dtype=torch.int64, device=device)
        len_megabatches = [torch.zeros_like(len_local_megabatches) for _ in range(world_size)]
        torch.distributed.all_gather(len_megabatches, len_local_megabatches)
        max_local_megabatches = max(len_megabatches)
    else:
        max_local_megabatches = len(local_megabatches)
    
    if len(local_megabatches) < max_local_megabatches:
        pad_length = max_local_megabatches - len(local_megabatches)
        pad_megabatches = random.choices(local_megabatches, k=pad_length)
        local_megabatches.extend(pad_megabatches)
    
    ret_list = [Packer([-1]) for _ in range(len(local_megabatches)*world_size)]
    local_idx = 0
    random.shuffle(local_megabatches)
    for i in range(rank * iter_batch_size, len(ret_list), world_size*iter_batch_size):
        for j in range(i, i + iter_batch_size):
            if local_idx < len(local_megabatches):
                ret_list[j] = local_megabatches[local_idx]
                local_idx += 1
            else:
                break

    return ret_list



# modified from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L99
class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
        generator=None,
    ):
        if dataset is None and lengths is None:
            raise ValueError('One of dataset and lengths must be provided.')

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else 'input_ids'
            if (
                    not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                    or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    'Can only automatically infer lengths for datasets whose items are dictionaries with an '
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                'If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]...'
            )
            lengths = lengths.tolist()
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        
    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)



class SequenceParallelRandomSampler(Sampler):

    def __init__(self,
                 dataset: Sized,
                 data_parallel_rank: int,
                 data_parallel_world_size : int,
                 generator=None,
                 round_up: bool = True) -> None:
        """
        This is a random sampler for sequence parallel training.
        It will sample indices from the dataset randomly.
        round_up: whether to add extra samples to make it evenly divisible.
        """
        self.rank = data_parallel_rank
        self.world_size = data_parallel_world_size
        print(self.world_size, self.rank)

        self.dataset = dataset
        self.generator = generator
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / self.world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.rank) / self.world_size)
            self.total_size = len(self.dataset)
            
            
        
    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples
    
    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        indices = torch.randperm(len(self.dataset), generator=self.generator).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)


class SequenceParallelLengthGroupedSampler(Sampler):
    def __init__(self,
                 batch_size: int,
                 data_parallel_rank: int,
                 data_parallel_world_size: int,
                 gradient_accumulation_steps: int,
                 dataset: Optional[Dataset] = None,
                 lengths: Optional[List[int]] = None,
                 model_input_name: Optional[str] = None,
                 generator=None):
        if dataset is None and lengths is None:
            raise ValueError('One of dataset and lengths must be provided.')

        self.batch_size = batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_world_size = data_parallel_world_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.total_world_size = data_parallel_world_size * gradient_accumulation_steps
        self.generator = generator

        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else 'input_ids'
            if (
                    not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                    or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    'Can only automatically infer lengths for datasets whose items are dictionaries with an '
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                'If lengths is a torch.Tensor, SequenceParallelLengthGroupedSampler will be slow. Converting lengths to List[int]...'
            )
            lengths = lengths.tolist()
        self.lengths = lengths

        print(len(self.lengths), self.batch_size, self.data_parallel_world_size)
        self.num_samples = math.ceil(len(self.lengths) / self.data_parallel_world_size)
        self.total_size = self.num_samples * self.data_parallel_world_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, self.total_world_size, generator=self.generator)
        
        # add extra samples to make it evenly divisible
        indices = (indices * int(self.total_size / len(indices) + 1))[:self.total_size]
            
        
        # print(len(indices))
        # subsample
        indices = indices[self.data_parallel_rank:self.total_size:self.data_parallel_world_size]

        return iter(indices)

# modified from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L99
class OnlinePackingGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        weights: Optional[List] = None,
        model_input_name: Optional[str] = None,
        generator=None,
        accumulation_steps: int = 1,
        model_max_length: int = 8192,
        packing_megabatch_size: int = 4000,
        default_rank: int = None,
    ):
        if dataset is None and lengths is None:
            raise ValueError('One of dataset and lengths must be provided.')

        self.batch_size = batch_size
        self.model_max_length = model_max_length
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else 'input_ids'
            if (
                    not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                    or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    'Can only automatically infer lengths for datasets whose items are dictionaries with an '
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                'If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]...'
            )
            lengths = lengths.tolist()
        self.rank = dist.get_rank() if default_rank is None else default_rank
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.accumulation_steps = accumulation_steps
        self.indices = get_length_grouped_indices_for_online_packing(self.lengths, self.batch_size, self.world_size, generator=self.generator, accumulation_steps=self.accumulation_steps, model_max_length=self.model_max_length, weights=weights, default_rank=default_rank, megabatch_size=packing_megabatch_size)
    
    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        
        # current_rank = torch.distributed.get_rank()
        # print(f'rank {current_rank}, len of indices: {len(indices)}, indices: {indices}')
        return iter(self.indices)

# patch trainer
def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    if self.train_dataset is None or not has_length(self.train_dataset):
        return None
    # Build the sampler.
    if self.args.group_by_length:
        lengths = []
        for dataset in self.train_dataset.datasets:
            lengths = lengths + dataset.length
        if self.processing_class is not None:
            try:
                model_input_name = self.processing_class.tokenizer.model_input_names[0]
            except:
                model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        else:
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        print("Using LengthGroupedSampler")
        return LengthGroupedSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            # self.args.train_batch_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )
    else:
        print("Using RandomSampler")
        return RandomSampler(self.train_dataset)



def _get_sequence_parallel_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    if self.train_dataset is None or not has_length(self.train_dataset):
        return None
    
    data_parallel_rank = get_pg_manager().data_parallel_rank
    data_parallel_world_size = get_pg_manager().data_parallel_world_size
    # Build the sampler.
    if self.args.group_by_length:
        lengths = []
        for dataset in self.train_dataset.datasets:
            lengths = lengths + dataset.length
        if self.processing_class is not None:
            try:
                model_input_name = self.processing_class.tokenizer.model_input_names[0]
            except:
                model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        else:
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        print("Using SequenceParallelLengthGroupedSampler")
        return SequenceParallelLengthGroupedSampler(
            self.args.train_batch_size, # now always 1
            data_parallel_rank=data_parallel_rank,
            data_parallel_world_size=data_parallel_world_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )
    else:
        print("Using SequenceParallelRandomSampler")
        return SequenceParallelRandomSampler(
            self.train_dataset,
            data_parallel_rank=data_parallel_rank,
            data_parallel_world_size=data_parallel_world_size,
        )
        

# patch trainer
def _get_train_sampler_for_online_packing(self) -> Optional[torch.utils.data.Sampler]:
    if self.train_dataset is None or not has_length(self.train_dataset):
        return None
    # Build the sampler.
    lengths = []
    weights = []
    for dataset in self.train_dataset.datasets:
        lengths = lengths + dataset.length
        weights = weights + dataset.weights

    if self.processing_class is not None:
        try:
            model_input_name = self.processing_class.tokenizer.model_input_names[0]
            model_max_length = self.processing_class.tokenizer.model_max_length
        except:
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            model_max_length = self.tokenizer.model_max_length
    else:
        model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        model_max_length = self.tokenizer.model_max_length

    generator = torch.Generator()
    generator.manual_seed(self.args.seed)
    print(f'generator.manual_seed for online packing: {self.args.seed}')
    return OnlinePackingGroupedSampler(
        self.args.train_batch_size * self.args.gradient_accumulation_steps,
        world_size=self.args.world_size,
        dataset=self.train_dataset,
        lengths=lengths,
        weights=weights,
        model_input_name=model_input_name,
        accumulation_steps=self.args.gradient_accumulation_steps,
        model_max_length=model_max_length,
        packing_megabatch_size=-1, # non-random global packing
    )



def __len__(self):
    return len(self.batch_sampler)


def __iter__(self):
    return self.batch_sampler.__iter__()

def replace_train_sampler():
    if get_pg_manager() is not None:
        transformers.Trainer._get_train_sampler = _get_sequence_parallel_train_sampler
        accelerate.data_loader.BatchSamplerShard.__len__ = __len__
        accelerate.data_loader.BatchSamplerShard.__iter__ = __iter__
        print('Replace train sequence parallel sampler!!')
    else:
        transformers.Trainer._get_train_sampler = _get_train_sampler
        print('Replace train sampler!!')


def replace_train_sampler_for_online_packing():
    transformers.Trainer._get_train_sampler = _get_train_sampler_for_online_packing
    print('Replace train sampler!!')
