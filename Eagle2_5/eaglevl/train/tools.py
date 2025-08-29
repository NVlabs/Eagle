# Copyright 2024 the LlamaFactory team.
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

from enum import Enum, unique
from typing import  Dict, List, Sequence, Set, Union
from datasets import concatenate_datasets, interleave_datasets
import bisect
import numpy as np


import time
from transformers import TrainerCallback, Trainer, TrainingArguments
import glob
import os
from transformers.trainer_utils import get_last_checkpoint
import torch.distributed as dist
import shutil

def get_last_checkpoint_guard(folder):
    while True:
        last_checkpoint = get_last_checkpoint(folder)
        if last_checkpoint is None:
            break
        
        world_size = dist.get_world_size()
        if len(glob.glob(os.path.join(last_checkpoint, "*.pth"))) != world_size:
            # incomplete xxx.pth
            shutil.rmtree(last_checkpoint)
        else:
            break

    return last_checkpoint

class SaveCheckpointCallback(TrainerCallback):
    def __init__(self, initial_interval_hours, save_interval_minutes):
        super().__init__()
        self.initial_interval_seconds = initial_interval_hours * 3600 - 15 * 60
        self.save_interval_seconds = save_interval_minutes * 60
        self.start_time = None
        self.first_save_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is None:
            return control

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Check if the initial interval has passed
        if self.first_save_time is None and elapsed_time >= self.initial_interval_seconds:
            self.first_save_time = current_time
            control.should_save = True
        # Check if the subsequent save interval has passed
        elif self.first_save_time is not None and (current_time - self.first_save_time) >= self.save_interval_seconds:
            self.first_save_time = current_time
            control.should_save = True

        return control



import torch
from transformers import TrainerCallback
import torch.distributed as dist
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown, NVML_TEMPERATURE_GPU, nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature

class MemoryLoggerCallback(TrainerCallback):
    def __init__(self):
        nvmlInit()  
        self.rank = dist.get_rank() if torch.distributed.is_initialized() else 0
        self.device_id = torch.cuda.current_device()

    def log_gpu_info(self, step):
        
        handle = nvmlDeviceGetHandleByIndex(self.device_id)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        power_usage = nvmlDeviceGetPowerUsage(handle) / 1000
       
        print(f"[Step {step} | Rank {self.rank} / GPU {self.device_id}] "
              f"Memory: {mem_info.used / 1024**2:.2f} MB, "
              f"Temperature: {temperature}°C, "
              f"Power: {power_usage:.2f} W, ")

    def on_step_end(self, args, state, control, **kwargs):
        if self.rank % 32 == 0:
            self.log_gpu_info(state.global_step)

    def __del__(self):
        nvmlShutdown() 



