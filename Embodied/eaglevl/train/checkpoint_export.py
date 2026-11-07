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

"""Export self-contained Eagle Embodied remote code into trained checkpoints."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


_LAYOUTS = {
    "moonvit": {
        "files": (
            "chat_template.json",
            "configuration_eagle_embodied.py",
            "image_processing_eagle_embodied.py",
            "modeling_eagle_embodied.py",
            "modeling_vit.py",
            "preprocessor_config.json",
            "processing_eagle_embodied.py",
            "processor_config.json",
        ),
        "architecture": "EagleEmbodiedForConditionalGeneration",
        "model_type": "eagle_embodied",
        "auto_map": {
            "AutoConfig": "configuration_eagle_embodied.EagleEmbodiedConfig",
            "AutoModel": "modeling_eagle_embodied.EagleEmbodiedForConditionalGeneration",
            "AutoModelForCausalLM": "modeling_eagle_embodied.EagleEmbodiedForConditionalGeneration",
            "AutoModelForImageTextToText": "modeling_eagle_embodied.EagleEmbodiedForConditionalGeneration",
        },
    },
}


def _rewrite_config(config_path, updates):
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(updates)

    temporary_path = config_path.with_name("config.json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_path, config_path)


def export_eagle_embodied_checkpoint(output_dir, variant="moonvit"):
    """Copy the selected remote-code ABI without replacing trained config data."""
    try:
        layout = _LAYOUTS[variant]
    except KeyError as exc:
        raise ValueError(f"Unsupported Eagle Embodied checkpoint variant: {variant}") from exc

    output_path = Path(output_dir)
    config_path = output_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Trainer did not save a config: {config_path}")

    source_dir = Path(__file__).resolve().parents[1] / "utils" / variant
    sources = [source_dir / filename for filename in layout["files"]]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing remote-code sources: " + ", ".join(missing))

    for source in sources:
        shutil.copy2(source, output_path / source.name)

    _rewrite_config(
        config_path,
        {
            "architectures": [layout["architecture"]],
            "model_type": layout["model_type"],
            "auto_map": layout["auto_map"],
        },
    )
    return tuple(path.name for path in sources)


def export_vflash_draft_checkpoint(model, output_dir):
    """Export the standalone Qwen3 VFlash remote-code artifact.

    Writes ``model.config`` (the VFlash drafter's) and the drafter's modeling
    file into ``output_dir``. The frozen Eagle target is intentionally absent.
    Binding metadata in the draft config identifies the exact target/tokenizer
    this artifact expects.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "model"
        / "vflash"
        / "modeling_dflash.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Missing VFlash remote-code source: {source}")

    output_path = Path(output_dir)
    model.config.save_pretrained(output_path)
    shutil.copy2(source, output_path / source.name)

    _rewrite_config(
        output_path / "config.json",
        {
            "architectures": ["DFlashDraftModel"],
            "auto_map": {"AutoModel": "modeling_dflash.DFlashDraftModel"},
        },
    )
    return (source.name,)
