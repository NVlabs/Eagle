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

"""Binding between a frozen Eagle Embodied target and its VFlash drafter.

A drafter checkpoint records the exact target it was trained against: a hash
of the target's logical state dict, of its text config and of its tokenizer
vocabulary, plus the drafter topology and physical-atom schema. Training
writes this metadata and checks it on resume; loading checks it before the
drafter is attached for decoding.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Mapping, Optional

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .modeling_dflash import (
    VFLASH_NUM_DRAFT_LAYERS,
    DFlashDraftModel,
    build_target_layer_ids,
    module_weight_fingerprint,
    validate_strict_vflash_config,
)
from .physical_atoms import (
    DEFAULT_OOF_LOSS_SCALE,
    validate_oof_loss_scale,
)


# Binding metadata version. A drafter checkpoint records it and refuses to
# run under another.
VFLASH_BINDING_VERSION = 7


def stable_json_sha256(value: Mapping) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_config_fingerprint(text_config) -> str:
    """Fingerprint architecture-critical target text configuration."""
    config = copy.deepcopy(text_config.to_dict())
    for key in (
        "_name_or_path",
        "transformers_version",
        "torch_dtype",
        "_attn_implementation_autoset",
        "_attn_implementation_internal",
    ):
        config.pop(key, None)
    return stable_json_sha256(config)


def tokenizer_fingerprint(tokenizer) -> str:
    vocabulary = sorted(
        ((str(token), int(token_id)) for token, token_id in tokenizer.get_vocab().items()),
        key=lambda item: (item[1], item[0]),
    )
    return stable_json_sha256({"vocabulary": vocabulary})


def build_vflash_draft_config(
    target_text_config,
    *,
    mask_token_id: int,
    physical_atom_config: Mapping,
    target_model_id: str,
    target_weights_sha256: str,
    target_tokenizer_sha256: str,
    target_revision: Optional[str] = None,
    target_layer_ids: Optional[list[int]] = None,
    oof_loss_scale: float = DEFAULT_OOF_LOSS_SCALE,
) -> Qwen3Config:
    """Clone target Qwen3 dimensions into the canonical five-layer drafter."""
    if not isinstance(target_text_config, Qwen3Config):
        raise TypeError("Strict VFlash currently requires a Qwen3 target text config")
    if not target_weights_sha256:
        raise ValueError("Strict VFlash requires an exact target weight fingerprint")
    oof_loss_scale = validate_oof_loss_scale(oof_loss_scale)
    if int(physical_atom_config.get("mask_token_id", -1)) != int(mask_token_id):
        raise ValueError(
            "VFlash mask_token_id must match the frozen target physical schema"
        )

    uniform_layers = build_target_layer_ids(
        int(target_text_config.num_hidden_layers), VFLASH_NUM_DRAFT_LAYERS
    )
    if target_layer_ids is not None and [int(x) for x in target_layer_ids] != uniform_layers:
        raise ValueError(
            f"Strict VFlash target layers must be {uniform_layers}, got {target_layer_ids}"
        )

    config_dict = copy.deepcopy(target_text_config.to_dict())
    config_dict.update(
        architectures=["DFlashDraftModel"],
        num_hidden_layers=VFLASH_NUM_DRAFT_LAYERS,
        layer_types=["full_attention"] * VFLASH_NUM_DRAFT_LAYERS,
        use_cache=True,
    )
    # Training supplies a compiled FlexAttention BlockMask; focused CPU tests
    # retain an explicit 4D eager reference mask.
    config_dict.pop("_attn_implementation_internal", None)
    draft_config = Qwen3Config(**config_dict)
    draft_config.num_target_layers = int(target_text_config.num_hidden_layers)
    draft_config.block_size = 16
    draft_config.physical_atom_config = copy.deepcopy(dict(physical_atom_config))
    draft_config.dflash_config = {
        "version": VFLASH_BINDING_VERSION,
        "architecture": "strict_qwen3_dflash",
        "mask_token_id": int(mask_token_id),
        "target_layer_ids": uniform_layers,
        "num_target_layers": int(target_text_config.num_hidden_layers),
        "target_model_id": str(target_model_id),
        "target_revision": str(target_revision) if target_revision else None,
        "target_weights_fingerprint_algorithm": "eagle-state-dict-sha256-v1",
        "target_weights_sha256": str(target_weights_sha256),
        "target_text_config_sha256": text_config_fingerprint(target_text_config),
        "target_tokenizer_sha256": str(target_tokenizer_sha256),
        "text_block_size": 16,
        # Legacy fields, kept at the values of the reference drafter configs.
        "trajectory_block_size": 54,
        "trajectory_layout": None,
        "trajectory_loss_weights": None,
        "atomic_decoding": None,
        "atomic_split": None,
        "num_draft_layers": VFLASH_NUM_DRAFT_LAYERS,
        "physical_atom_loss_weights": "uniform",
        "oof_loss_scale": oof_loss_scale,
    }
    draft_config.auto_map = {"AutoModel": "modeling_dflash.DFlashDraftModel"}
    return draft_config


def validate_vflash_draft_binding(
    draft_config,
    *,
    target_text_config,
    physical_atom_config: Mapping,
    target_weights_sha256: Optional[str] = None,
    target_tokenizer_sha256: Optional[str] = None,
    oof_loss_scale: Optional[float] = None,
) -> None:
    """Fail closed unless a draft checkpoint matches the exact frozen target.

    A target hash or ``oof_loss_scale`` passed as None is not compared, but
    the checkpoint must still record every target hash.
    """
    if not isinstance(physical_atom_config, Mapping):
        raise ValueError("Strict VFlash requires a target with a physical-atom schema")
    if oof_loss_scale is not None:
        oof_loss_scale = validate_oof_loss_scale(oof_loss_scale)
    binding = getattr(draft_config, "dflash_config", {}) or {}
    if int(binding.get("version", -1)) != VFLASH_BINDING_VERSION:
        raise ValueError(
            "Strict VFlash checkpoint requires binding metadata "
            f"version {VFLASH_BINDING_VERSION}"
        )
    if binding.get("target_weights_fingerprint_algorithm") != (
        "eagle-state-dict-sha256-v1"
    ):
        raise ValueError("Unsupported VFlash target weight fingerprint algorithm")
    validate_strict_vflash_config(draft_config, target_text_config)
    expected_layers = build_target_layer_ids(
        int(target_text_config.num_hidden_layers), VFLASH_NUM_DRAFT_LAYERS
    )
    expected_structure = {
        "architecture": "strict_qwen3_dflash",
        "mask_token_id": int(physical_atom_config["mask_token_id"]),
        "target_layer_ids": expected_layers,
        "num_target_layers": int(target_text_config.num_hidden_layers),
        "text_block_size": 16,
        "num_draft_layers": VFLASH_NUM_DRAFT_LAYERS,
        "trajectory_block_size": 54,
    }
    structural_mismatches = [
        key for key, expected in expected_structure.items()
        if binding.get(key) != expected
    ]
    if structural_mismatches:
        raise ValueError(
            "VFlash checkpoint topology metadata does not match strict VFlash: "
            + ", ".join(structural_mismatches)
        )
    if binding.get("physical_atom_loss_weights") != "uniform" or (
        oof_loss_scale is not None and binding.get("oof_loss_scale") != oof_loss_scale
    ):
        expected_objective = "uniform physical-atom weights"
        if oof_loss_scale is not None:
            expected_objective = f"oof_loss_scale={oof_loss_scale} and {expected_objective}"
        raise ValueError(
            "VFlash checkpoint objective metadata does not match " + expected_objective
        )
    required = {
        "target_weights_sha256": target_weights_sha256,
        "target_text_config_sha256": text_config_fingerprint(target_text_config),
        "target_tokenizer_sha256": target_tokenizer_sha256,
    }
    missing = [key for key in required if not binding.get(key)]
    if missing:
        raise ValueError(
            "Strict VFlash checkpoint is missing target binding metadata: "
            + ", ".join(missing)
        )
    mismatched = [
        key for key, expected in required.items()
        if expected is not None and str(binding.get(key)) != str(expected)
    ]
    if mismatched:
        raise ValueError(
            "VFlash checkpoint is bound to a different frozen Eagle target: "
            + ", ".join(mismatched)
        )
    if getattr(draft_config, "physical_atom_config", None) != dict(
        physical_atom_config
    ):
        raise ValueError(
            "VFlash checkpoint physical-atom token/schema mapping does not match "
            "the frozen Eagle target"
        )


def validate_vflash_target_weights(target, draft_config) -> None:
    """Check a drafter's target-weight binding against ``target``.

    The logical fingerprint is cached on ``target`` as
    ``_vflash_target_weights_sha256``. The target must reset that attribute to
    None whenever its parameters are moved or cast.
    """
    binding = getattr(draft_config, "dflash_config", {}) or {}
    expected = binding.get("target_weights_sha256")
    if not expected:
        raise ValueError("Strict VFlash drafter is missing its target weight binding")
    actual = getattr(target, "_vflash_target_weights_sha256", None)
    if actual is None:
        actual = module_weight_fingerprint(target)
        object.__setattr__(target, "_vflash_target_weights_sha256", actual)
    if actual != expected:
        raise ValueError(
            "VFlash drafter is bound to different frozen Eagle target weights"
        )


_HF_QWEN3_MODULE = "transformers.models.qwen3.modeling_qwen3"


def assert_hf_native_drafter(draft_model: DFlashDraftModel) -> None:
    """Fail unless every drafter MLP and RMSNorm is the Hugging Face Qwen3 class.

    ``replace_liger_fused_ops()`` installs Liger kernels into the Hugging Face
    Qwen3 module for the target. The drafter binds its MLP/RMSNorm classes when
    its own module is imported, which must happen first. Drafter training and
    ``load_vflash_draft`` both run this check.
    """
    found = set()
    for name, module in draft_model.named_modules():
        cls = type(module)
        for kind in ("MLP", "RMSNorm"):
            if not cls.__name__.endswith(kind):
                continue
            found.add(kind)
            if cls.__module__ != _HF_QWEN3_MODULE:
                raise RuntimeError(
                    f"VFlash drafter module {name!r} is {cls.__module__}.{cls.__qualname__}; "
                    "import eaglevl.model.vflash.modeling_dflash before "
                    "replace_liger_fused_ops() so the drafter keeps the Hugging Face "
                    "Qwen3 MLP and RMSNorm."
                )
    if found != {"MLP", "RMSNorm"}:
        raise RuntimeError(f"VFlash drafter has no MLP/RMSNorm modules to check: {sorted(found)}")


def load_vflash_draft(target, draft_model_path, torch_dtype=None, tokenizer=None):
    """Load and validate the VFlash drafter bound to ``target``.

    ``target`` is an Eagle Embodied model. The drafter is returned on the
    target language model's device and dtype, with the attention implementation
    transformers selects by default. The caller keeps it outside the registered
    module tree, so the target's state dict and weight fingerprint do not
    change. A drafter already attached from the same path is revalidated and
    returned. With ``tokenizer``, a newly loaded drafter must also be bound to
    that tokenizer's vocabulary. A newly loaded drafter must keep the Hugging
    Face Qwen3 MLP and RMSNorm classes.
    """
    if not getattr(target, "supports_vflash_generation", False):
        raise ValueError("Strict VFlash integration currently supports a Qwen3 Eagle target")
    target_devices = {parameter.device for parameter in target.language_model.parameters()}
    if len(target_devices) != 1:
        raise ValueError(
            "VFlash generation requires the Eagle language target on one device; "
            "sharded device_map targets are not supported"
        )
    target_parameter = next(target.language_model.parameters())
    if torch_dtype is not None and torch_dtype != target_parameter.dtype:
        raise ValueError(
            "VFlash drafter dtype must match the Eagle language target dtype"
        )
    dtype = target_parameter.dtype
    current_draft = getattr(target, "vflash_draft", None)
    if (
        current_draft is not None
        and getattr(target, "_vflash_draft_path", None) == str(draft_model_path)
    ):
        validate_vflash_target_weights(target, current_draft.config)
        current_draft.to(device=target_parameter.device, dtype=dtype).eval()
        return current_draft

    # Validate the lightweight config and exact target binding before
    # fingerprinting the target and loading the full-width five-layer drafter.
    draft_config = DFlashDraftModel.config_class.from_pretrained(draft_model_path)
    validate_vflash_draft_binding(
        draft_config,
        target_text_config=target.config.text_config,
        physical_atom_config=getattr(target.config, "physical_atom_config", None),
        target_tokenizer_sha256=(
            None if tokenizer is None else tokenizer_fingerprint(tokenizer)
        ),
    )
    if int(draft_config.vocab_size) != int(target.language_model.get_input_embeddings().num_embeddings):
        raise ValueError("VFlash drafter vocabulary does not match the Eagle target")
    binding = getattr(draft_config, "dflash_config", {}) or {}
    if int(binding.get("mask_token_id", -1)) != int(
        draft_config.physical_atom_config.get("mask_token_id", -2)
    ):
        raise ValueError(
            "VFlash drafter mask token does not match its physical schema"
        )
    validate_vflash_target_weights(target, draft_config)
    draft_model = DFlashDraftModel.from_pretrained(
        draft_model_path, config=draft_config, torch_dtype=dtype
    ).to(target_parameter.device).eval()
    assert_hf_native_drafter(draft_model)
    return draft_model
