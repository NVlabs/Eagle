# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os

import math
import torch.distributed as dist
import torch


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if not self.__initialized:
            self.__initialized = True


class ProcessGroupManager(Singleton):
    """
    sp_degree = sp_ring_degree x sp_ulysses_degree
    """
    @property
    def sequence_parallel_rank(self):
        return dist.get_rank(self.sequence_parallel_group)
    
    @property
    def sequence_parallel_world_size(self):
        return dist.get_world_size(self.sequence_parallel_group)
    
    @property
    def data_parallel_world_size(self):
        return dist.get_world_size(self.data_parallel_group)
    
    @property
    def data_parallel_rank(self):
        return dist.get_rank(self.data_parallel_group)
    
    @property
    def ring_sequence_parallel_rank(self):
        return dist.get_rank(self.ring_sequence_parallel_group)
    
    @property
    def ring_sequence_parallel_world_size(self):
        return dist.get_world_size(self.ring_sequence_parallel_group)
    
    @property
    def ulysses_sequence_parallel_rank(self):
        return dist.get_rank(self.ulysses_sequence_parallel_group)
    
    @property
    def ulysses_sequence_parallel_world_size(self):
        return dist.get_world_size(self.ulysses_sequence_parallel_group)
    
    
    

    def __init__(self, sequence_parallel_size, ulysses_sequence_parallel_size, ring_sequence_parallel_size):
        if not hasattr(self, "__initialized"):
            super().__init__()
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.sequence_parallel_size = sequence_parallel_size
            self.ulysses_sequence_parallel_size = ulysses_sequence_parallel_size
            self.ring_sequence_parallel_size = ring_sequence_parallel_size 
            assert self.sequence_parallel_size // self.ulysses_sequence_parallel_size == self.ring_sequence_parallel_size, f"sequence_parallel_size {self.sequence_parallel_size} // ulysses_sequence_parallel_size {self.ulysses_sequence_parallel_size} != ring_sequence_parallel_size {self.ring_sequence_parallel_size}"
            assert self.sequence_parallel_size % self.ring_sequence_parallel_size == 0, f"sequence_parallel_size {self.sequence_parallel_size} % ring_sequence_parallel_size {self.ring_sequence_parallel_size} != 0"
            self.data_parallel_size = self.world_size // self.sequence_parallel_size
            self.num_sequence_parallel_groups = self.world_size // self.sequence_parallel_size
            self.num_ulysses_sequence_parallel_groups = self.sequence_parallel_size // self.ulysses_sequence_parallel_size
            self.num_ring_sequence_parallel_groups = self.sequence_parallel_size // self.ring_sequence_parallel_size
            

            for i in range(self.data_parallel_size):
                ranks = list(range(i * self.sequence_parallel_size, (i + 1) * self.sequence_parallel_size))
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self.sequence_parallel_group = group

            for j in range(self.sequence_parallel_size):
                ranks = list(range(j, self.world_size, self.sequence_parallel_size))
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self.data_parallel_group = group
            
            for dp_rank in range(self.data_parallel_size):
                offset = dp_rank * self.sequence_parallel_size
                for i in range(self.num_ulysses_sequence_parallel_groups):
                    ulysses_ranks = list(
                        range(
                            i * self.ulysses_sequence_parallel_size + offset,
                            (i + 1) * self.ulysses_sequence_parallel_size + offset,
                        )
                    )
                    group = dist.new_group(ulysses_ranks)
                    if self.rank in ulysses_ranks:
                        self.ulysses_sequence_parallel_group = group

                for i in range(self.num_ring_sequence_parallel_groups):
                    ring_ranks = list(range(i + offset, self.sequence_parallel_size + offset, self.num_ring_sequence_parallel_groups))
                    group = dist.new_group(ring_ranks)
                    if self.rank in ring_ranks:
                        self.ring_sequence_parallel_group = group
            


            print("--------------ProcessGroupManager Initialized---------------------")
            print("Sequence Parallel Group: ", self.sequence_parallel_group)
            print("Sequence Parallel Rank: ", self.sequence_parallel_rank)
            print("Sequence Parallel World Size: ", self.sequence_parallel_world_size)
            print("Data Parallel Group: ", self.data_parallel_group)
            print("Data Parallel Rank: ", self.data_parallel_rank)
            print("Data Parallel World Size: ", self.data_parallel_world_size)
            print("Ulysses Sequence Parallel Group: ", self.ulysses_sequence_parallel_group)
            print("Ulysses Sequence Parallel Rank: ", self.ulysses_sequence_parallel_rank)
            print("Ulysses Sequence Parallel World Size: ", self.ulysses_sequence_parallel_world_size)
            print("Ring Sequence Parallel Group: ", self.ring_sequence_parallel_group)
            print("Ring Sequence Parallel Rank: ", self.ring_sequence_parallel_rank)
            print("Ring Sequence Parallel World Size: ", self.ring_sequence_parallel_world_size)
            print("--------------ProcessGroupManager Initialized---------------------")


            
PROCESS_GROUP_MANAGER = None


def get_llm_num_heads(model):
    # now check qwen2
    return model.language_model.config.num_attention_heads

def set_pg_manager(model, sequence_parallel_size, ring_sequence_parallel_size=1):
    """
    Set the process group manager for sequence parallelism.
    sp_degree = sp_ring_degree x sp_ulysses_degree
    """
    print("setting pg manager", flush=True)
    # first check torch distributed group init and set device accordingly;
    # (DL) TODO: Whether this can be skipped in DeepSpeed.
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(
                "torch distributed is already initialized, " "skipping initialization ...",
                flush=True,
            )
    else:
        raise RuntimeError("torch distributed is not initialized")

    world_size = dist.get_world_size()

    assert sequence_parallel_size <= world_size, f"sequence_parallel_size {sequence_parallel_size} > world_size {world_size}"
    assert world_size % sequence_parallel_size == 0, f"world_size {world_size} % sequence_parallel_size {sequence_parallel_size} != 0"

    if ring_sequence_parallel_size < 1:
        ring_sequence_parallel_size = 1
    
    num_heads = get_llm_num_heads(model)
    ulysses_sequence_parallel_size = math.gcd(num_heads, sequence_parallel_size)
    ring_sequence_parallel_size = sequence_parallel_size // ulysses_sequence_parallel_size

    assert sequence_parallel_size % ring_sequence_parallel_size == 0, f"sequence_parallel_size {sequence_parallel_size} % ring_sequence_parallel_size {ring_sequence_parallel_size} != 0"

    # Init the process group manager
    global PROCESS_GROUP_MANAGER
    PROCESS_GROUP_MANAGER = ProcessGroupManager(sequence_parallel_size, ulysses_sequence_parallel_size, ring_sequence_parallel_size)


def get_pg_manager():
    return PROCESS_GROUP_MANAGER

