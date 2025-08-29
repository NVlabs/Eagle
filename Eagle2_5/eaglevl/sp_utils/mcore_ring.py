"""Attention."""
import collections
from contextlib import nullcontext
from importlib.metadata import version as get_pkg_version
from importlib.metadata import PackageNotFoundError
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import logging

from dataclasses import dataclass, fields
import numpy as np
from packaging.version import Version as PkgVersion

import torch
import torch.nn.functional as F


@jit_fuser
def get_cu_seqlens_on_cp_rank(
    cu_seqlens, cu_seqlens_padded_on_cp_rank, cp_size, cp_rank, first_half, second_half
):
    """Compute cu_seqlens of a context parallelism rank"""
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    seqlens_padded = (cu_seqlens_padded_on_cp_rank[1:] - cu_seqlens_padded_on_cp_rank[:-1]) // 2
    zeros = torch.zeros_like(seqlens)
    cu_seqlens_on_cp_rank = torch.zeros_like(cu_seqlens)
    if first_half:
        seqlens_1 = seqlens - cp_rank * seqlens_padded
        seqlens_1 = seqlens_1.clamp(zeros, seqlens_padded)
        cu_seqlens_on_cp_rank[1:].add_(seqlens_1)
    if second_half:
        seqlens_2 = seqlens - (2 * cp_size - cp_rank - 1) * seqlens_padded
        seqlens_2 = seqlens_2.clamp(zeros, seqlens_padded)
        cu_seqlens_on_cp_rank[1:].add_(seqlens_2)
    cu_seqlens_on_cp_rank.cumsum_(dim=0)
    return cu_seqlens_on_cp_rank

class AttnFuncWithCPAndKVP2P(torch.autograd.Function):
    """
    Attention implementation with context parallelism. Exchange KV between CP ranks
    with P2P in ring topology. Split attention compute into multiple steps, and overlap
    current-step compute with next-step communication.
    """

    @staticmethod
    def forward(
        ctx,
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        softmax_scale,
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
        fp8,
        fp8_meta,
        cp_group,
        cp_global_ranks,
        cp_stream,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)
        send_dst = cp_global_ranks[(rank + 1) % cp_size]
        recv_src = cp_global_ranks[(rank - 1) % cp_size]
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (cp_size == 2)

        causal = "causal" in attn_mask_type
        padding = "padding" in attn_mask_type

        if qkv_format in ["bshd", "sbhd"]:
            seq_dim = qkv_format.index("s")
            qkv_layout = qkv_format + "_" + qkv_format[:-2] + "2" + qkv_format[-2:]
        else:
            qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        pad_between_seqs_q = not torch.equal(cu_seqlens_q_padded, cu_seqlens_q)
        pad_between_seqs_kv = not torch.equal(cu_seqlens_kv_padded, cu_seqlens_kv)
        max_seqlen_q = max_seqlen_q // cp_size
        max_seqlen_kv = max_seqlen_kv // cp_size
        cu_seqlens_q_padded = cu_seqlens_q_padded // cp_size
        cu_seqlens_kv_padded = cu_seqlens_kv_padded // cp_size
        cu_seqlens_q_per_step = [None for _ in range(cp_size)]
        cu_seqlens_kv_per_step = [None for _ in range(cp_size)]

        assert qkv_format == "thd" or (
            q.shape[seq_dim] % 2 == 0 and k.shape[seq_dim] % 2 == 0
        ), "Sequence length per GPU needs to be divisible by 2!"
        if causal:
            if qkv_format == "bshd":
                # [b, s, np, hn] -> [b, 2, s//2, np, hn]
                q, k, v = [x.view(x.shape[0], 2, x.shape[1] // 2, *x.shape[2:]) for x in [q, k, v]]
            elif qkv_format == "sbhd":
                # [s, b, np, hn] -> [2, s//2, b, np, hn]
                q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]
        total_tokens_kv = None if qkv_format != "thd" else k.shape[0]
        # remove padded tokens at the end
        k, v = [x if qkv_format != "thd" else x[: cu_seqlens_kv_padded[-1]] for x in [k, v]]
        if attn_bias is not None:
            assert len(attn_bias.shape) == 4, (
                "Only support bias shape of [b, h, sq, sk] for forward, "
                "and [1, h, sq, sk] for backward!"
            )
            assert (
                attn_bias.shape[-2] % 2 == 0 and attn_bias.shape[-1] % (2 * cp_size) == 0
            ), "Sequence length does not meet divisible requirements!"
            # [b, np, sq, sk] -> [b, np, 2, sq//2, 2*cp, sk//(2*cp)]
            attn_bias_ = attn_bias.view(
                *attn_bias.shape[:-2],
                2,
                attn_bias.shape[-2] // 2,
                2 * cp_size,
                attn_bias.shape[-1] // (2 * cp_size),
            )
            # [b, np, sq, sk] -> [b, np, sq, 2*cp, sk//(2*cp)]
            attn_bias = attn_bias.view(
                *attn_bias.shape[:-1], 2 * cp_size, attn_bias.shape[-1] // (2 * cp_size)
            )
        assert q.shape[-1] % 8 == 0, "hidden size per attention head should be multiple of 8"
        fa_optional_forward_kwargs = {}
        if _flash_attn_2_3_plus:
            fa_optional_forward_kwargs["window_size"] = (-1, 0) if causal else (-1, -1)
        if _flash_attn_2_4_plus:
            fa_optional_forward_kwargs["alibi_slopes"] = None
        if _flash_attn_2_5_7_plus:
            fa_optional_forward_kwargs["block_table"] = None

        # Flash Attn inputs
        q_inputs = [None, None]
        kv_inputs = [None, None]
        attn_bias_inputs = [None, None]
        # Flash Attn outputs
        out_per_step = [None for _ in range(cp_size)]
        softmax_lse_per_step = [None for _ in range(cp_size)]
        rng_states = [None for _ in range(cp_size)]
        attn_biases = [None for _ in range(cp_size)]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        # synchronize fwd results correction across steps
        fwd_results_correction_done = torch.cuda.Event()

        if fp8:
            if use_fused_attention:
                fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
                fused_attn_qkv_dtype = fp8_dtype_forward
                fused_attn_backend = FusedAttnBackend["FP8"]
                if fp8_meta["recipe"].fp8_mha:
                    assert (
                        isinstance(q, Float8Tensor)
                        and isinstance(k, Float8Tensor)
                        and isinstance(v, Float8Tensor)
                    ), "q/k/v must be Float8Tensors for FP8 MHA!"
                    fp8_meta["scaling_fwd"].scale_inv[META_QKV] = q._scale_inv
                    q_fp8, k_fp8, v_fp8 = q, k, v
                    q, k, v = q_fp8._data, k_fp8._data, v_fp8._data
                else:
                    q_f16, k_f16, v_f16 = q, k, v
                    q = cast_to_fp8(q_f16, fp8_meta["scaling_fwd"], META_QKV, fp8_dtype_forward)
                    if int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
                        k, v = [
                            cast_to_fp8(x, fp8_meta["scaling_fwd"], META_QKV, fp8_dtype_forward)
                            for x in [k_f16, v_f16]
                        ]
                fp8_meta_kwargs = {}
                fp8_meta_kwargs["d_scale_qkv"] = fp8_meta["scaling_fwd"].scale_inv
                fp8_meta_kwargs["d_scale_qkv_offset"] = META_QKV
                fp8_meta_kwargs["d_scale_s"] = fp8_meta["scaling_fwd"].scale_inv
                fp8_meta_kwargs["d_scale_s_offset"] = META_S
                fp8_meta_kwargs["q_scale_s"] = fp8_meta["scaling_fwd"].scale
                fp8_meta_kwargs["q_scale_s_offset"] = META_S
                fp8_meta_kwargs["q_scale_o"] = fp8_meta["scaling_fwd"].scale
                fp8_meta_kwargs["q_scale_o_offset"] = META_O_CP
                amax_per_step = torch.zeros((2, cp_size), dtype=torch.float32, device=q.device)
            else:
                assert False, "FP8 is only supported with Fused Attention!"
        else:
            q_f16 = q
            if use_fused_attention:
                fp8_meta_kwargs = {}
                fused_attn_qkv_dtype = TE_DType[q.dtype]
                fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        p2p_comm_buffers = [None for _ in range(cp_size)]
        if use_fused_attention and qkv_format in ["bshd", "sbhd"]:
            p2p_comm_buffers[0] = torch.cat((k.unsqueeze(-3), v.unsqueeze(-3)), dim=-3)
        else:
            p2p_comm_buffers[0] = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        send_recv_reqs = [[], []]

        for i in range(cp_size + 1):
            if i < cp_size:
                with torch.cuda.stream(flash_attn_streams[i % 2]):
                    # wait until KV is received
                    for req in send_recv_reqs[(i + 1) % 2]:
                        req.wait()

                    if i < (cp_size - 1):
                        p2p_comm_buffers[i + 1] = torch.empty_like(p2p_comm_buffers[i])
                        send_recv_reqs[i % 2] = flash_attn_p2p_communicate(
                            rank,
                            p2p_comm_buffers[i],
                            send_dst,
                            p2p_comm_buffers[i + 1],
                            recv_src,
                            cp_group,
                            batch_p2p_comm,
                        )

                    if (
                        not fp8
                        or fp8_meta["recipe"].fp8_mha
                        or int(os.getenv("NVTE_FP8_DPA_BWD", "1"))
                    ):
                        kv_inputs[i % 2] = p2p_comm_buffers[i]
                    else:
                        # KV exchange is in BF16/FP16, cast received KV in each step
                        kv_inputs[i % 2] = cast_to_fp8(
                            p2p_comm_buffers[i],
                            fp8_meta["scaling_fwd"],
                            META_QKV,
                            fp8_dtype_forward,
                        )
                    if fp8 and use_fused_attention:
                        fp8_meta_kwargs["amax_s"] = amax_per_step
                        fp8_meta_kwargs["amax_s_offset"] = i
                        fp8_meta_kwargs["amax_o"] = amax_per_step
                        fp8_meta_kwargs["amax_o_offset"] = cp_size + i
                    if causal:
                        if i == 0:
                            if pad_between_seqs_q:
                                cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_q, cu_seqlens_q_padded, cp_size, rank, True, True
                                )
                            else:
                                cu_seqlens_q_per_step[i] = cu_seqlens_q // cp_size
                            if pad_between_seqs_kv:
                                cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_kv, cu_seqlens_kv_padded, cp_size, rank, True, True
                                )
                            else:
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv // cp_size
                            if use_fused_attention:
                                if qkv_format == "bshd":
                                    # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                                    q_inputs[i % 2] = q.view(q.shape[0], -1, *q.shape[-2:])
                                    # [b, 2, sk//2, 2, np, hn] -> [b, sk, 2, np, hn]
                                    kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                        k.shape[0], -1, 2, *k.shape[-2:]
                                    )
                                elif qkv_format == "sbhd":
                                    # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                                    q_inputs[i % 2] = q.view(-1, *q.shape[-3:])
                                    # [2, sk//2, b, 2, np, hn] -> [sk, b, 2, np, hn]
                                    kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                        -1, k.shape[2], 2, *k.shape[-2:]
                                    )
                                elif qkv_format == "thd":
                                    q_inputs[i % 2] = q
                                if attn_bias is not None:
                                    idx = (rank - i) % cp_size
                                    attn_bias_inputs[i % 2] = torch.cat(
                                        (
                                            attn_bias[..., idx, :],
                                            attn_bias[..., (2 * cp_size - idx - 1), :],
                                        ),
                                        dim=-1,
                                    ).contiguous()
                                out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                    is_training,
                                    max_seqlen_q,
                                    max_seqlen_kv,
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    q_inputs[i % 2],
                                    (
                                        kv_inputs[i % 2][..., 0, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][0]
                                    ),
                                    (
                                        kv_inputs[i % 2][..., 1, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][1]
                                    ),
                                    fused_attn_qkv_dtype,
                                    fused_attn_backend,
                                    attn_scale=softmax_scale,
                                    dropout=dropout_p,
                                    qkv_layout=qkv_layout,
                                    attn_mask_type=attn_mask_type,
                                    attn_bias_type=attn_bias_type,
                                    attn_bias=attn_bias_inputs[i % 2],
                                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                                    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                                    **fp8_meta_kwargs,
                                )
                                if fp8:
                                    softmax_lse_per_step[i], _, rng_states[i] = aux_ctx_tensors
                                else:
                                    softmax_lse_per_step[i], rng_states[i], *rest = aux_ctx_tensors
                                    attn_biases[i] = rest[0] if len(rest) > 0 else None
                            else:
                                # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                                q_inputs[i % 2] = q.view(-1, *q.shape[-2:])
                                # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                                kv_inputs[i % 2] = kv_inputs[i % 2].view(2, -1, *k.shape[-2:])
                                (
                                    _,
                                    _,
                                    _,
                                    _,
                                    out_per_step[i],
                                    softmax_lse_per_step[i],
                                    _,
                                    rng_states[i],
                                ) = _flash_attn_forward(
                                    q_inputs[i % 2],
                                    kv_inputs[i % 2][0],
                                    kv_inputs[i % 2][1],
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    max_seqlen_q,
                                    max_seqlen_kv,
                                    dropout_p,
                                    softmax_scale,
                                    causal=True,
                                    return_softmax=False,
                                    **fa_optional_forward_kwargs,
                                )
                        elif i <= rank:
                            if pad_between_seqs_q:
                                cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_q, cu_seqlens_q_padded, cp_size, rank, True, True
                                )
                            else:
                                cu_seqlens_q_per_step[i] = cu_seqlens_q // cp_size
                            if pad_between_seqs_kv:
                                cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_kv,
                                    cu_seqlens_kv_padded,
                                    cp_size,
                                    (rank - i) % cp_size,
                                    True,
                                    False,
                                )
                            else:
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv // (cp_size * 2)
                            if use_fused_attention:
                                if qkv_format == "bshd":
                                    # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                                    q_inputs[i % 2] = q.view(q.shape[0], -1, *q.shape[-2:])
                                    # [b, 2, sk//2, 2, np, hn] -> [b, sk//2, 2, np, hn]
                                    kv_inputs[i % 2] = kv_inputs[i % 2][:, 0, ...].contiguous()
                                elif qkv_format == "sbhd":
                                    # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                                    q_inputs[i % 2] = q.view(-1, *q.shape[-3:])
                                    # [2, sk//2, b, 2, np, hn] -> [sk//2, b, 2, np, hn]
                                    kv_inputs[i % 2] = kv_inputs[i % 2][0].contiguous()
                                elif qkv_format == "thd":
                                    q_inputs[i % 2] = q
                                    # [2, t, np, hn] -> [2, t/2, np, hn]
                                    kv_inputs[i % 2] = tex.thd_read_half_tensor(
                                        kv_inputs[i % 2], cu_seqlens_kv_padded, 0
                                    )
                                if attn_bias is not None:
                                    idx = (rank - i) % cp_size
                                    attn_bias_inputs[i % 2] = attn_bias[..., idx, :].contiguous()
                                out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                    is_training,
                                    max_seqlen_q,
                                    max_seqlen_kv // 2,
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    q_inputs[i % 2],
                                    (
                                        kv_inputs[i % 2][..., 0, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][0]
                                    ),
                                    (
                                        kv_inputs[i % 2][..., 1, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][1]
                                    ),
                                    fused_attn_qkv_dtype,
                                    fused_attn_backend,
                                    attn_scale=softmax_scale,
                                    dropout=dropout_p,
                                    qkv_layout=qkv_layout,
                                    attn_mask_type="padding" if padding else "no_mask",
                                    attn_bias_type=attn_bias_type,
                                    attn_bias=attn_bias_inputs[i % 2],
                                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                                    cu_seqlens_kv_padded=(
                                        None
                                        if cu_seqlens_kv_padded is None
                                        else cu_seqlens_kv_padded // 2
                                    ),
                                    **fp8_meta_kwargs,
                                )
                                if fp8:
                                    softmax_lse_per_step[i], _, rng_states[i] = aux_ctx_tensors
                                else:
                                    softmax_lse_per_step[i], rng_states[i], *rest = aux_ctx_tensors
                                    attn_biases[i] = rest[0] if len(rest) > 0 else None
                            else:
                                # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                                q_inputs[i % 2] = q.view(-1, *q.shape[-2:])
                                if qkv_format == "thd":
                                    # [2, t, np, hn] -> [2, t/2, np, hn]
                                    kv_inputs[i % 2] = tex.thd_read_half_tensor(
                                        kv_inputs[i % 2], cu_seqlens_kv_padded, 0
                                    )
                                else:
                                    # [2, b, 2, sk//2, np, hn] -> [2, b, sk//2, np, hn]
                                    kv_inputs[i % 2] = kv_inputs[i % 2][:, :, 0, ...].contiguous()
                                # [2, b, sk//2, np, hn] -> [2, b*sk//2, np, hn]
                                kv_inputs[i % 2] = kv_inputs[i % 2].view(2, -1, *k.shape[-2:])
                                if _flash_attn_2_3_plus:
                                    fa_optional_forward_kwargs["window_size"] = (-1, -1)
                                (
                                    _,
                                    _,
                                    _,
                                    _,
                                    out_per_step[i],
                                    softmax_lse_per_step[i],
                                    _,
                                    rng_states[i],
                                ) = _flash_attn_forward(
                                    q_inputs[i % 2],
                                    kv_inputs[i % 2][0],
                                    kv_inputs[i % 2][1],
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    max_seqlen_q,
                                    max_seqlen_kv // 2,
                                    dropout_p,
                                    softmax_scale,
                                    causal=False,
                                    return_softmax=False,
                                    **fa_optional_forward_kwargs,
                                )
                        else:
                            if pad_between_seqs_q:
                                cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_q, cu_seqlens_q_padded, cp_size, rank, False, True
                                )
                            else:
                                cu_seqlens_q_per_step[i] = cu_seqlens_q // (cp_size * 2)
                            if pad_between_seqs_kv:
                                cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_kv,
                                    cu_seqlens_kv_padded,
                                    cp_size,
                                    (rank - i) % cp_size,
                                    True,
                                    True,
                                )
                            else:
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv // cp_size
                            if use_fused_attention:
                                if qkv_format == "bshd":
                                    # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
                                    q_inputs[i % 2] = q[:, 1, ...].contiguous()
                                    # [b, 2, sk//2, 2, np, hn] -> [b, sk, 2, np, hn]
                                    kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                        k.shape[0], -1, 2, *k.shape[-2:]
                                    )
                                elif qkv_format == "sbhd":
                                    # [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
                                    q_inputs[i % 2] = q[1].contiguous()
                                    # [2, sk//2, b, 2, np, hn] -> [sk, b, 2, np, hn]
                                    kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                        -1, k.shape[2], 2, *k.shape[-2:]
                                    )
                                elif qkv_format == "thd":
                                    # [t, np, hn] -> [t/2, np, hn]
                                    q_inputs[i % 2] = tex.thd_read_half_tensor(
                                        q, cu_seqlens_q_padded, 1
                                    )
                                if attn_bias is not None:
                                    idx = (rank - i) % cp_size
                                    attn_bias_inputs[i % 2] = torch.cat(
                                        (
                                            attn_bias_[..., 1, :, idx, :],
                                            attn_bias_[..., 1, :, (2 * cp_size - idx - 1), :],
                                        ),
                                        dim=-1,
                                    ).contiguous()
                                out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                    is_training,
                                    max_seqlen_q // 2,
                                    max_seqlen_kv,
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    q_inputs[i % 2],
                                    (
                                        kv_inputs[i % 2][..., 0, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][0]
                                    ),
                                    (
                                        kv_inputs[i % 2][..., 1, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][1]
                                    ),
                                    fused_attn_qkv_dtype,
                                    fused_attn_backend,
                                    attn_scale=softmax_scale,
                                    dropout=dropout_p,
                                    qkv_layout=qkv_layout,
                                    attn_mask_type="padding" if padding else "no_mask",
                                    attn_bias_type=attn_bias_type,
                                    attn_bias=attn_bias_inputs[i % 2],
                                    cu_seqlens_q_padded=(
                                        None
                                        if cu_seqlens_q_padded is None
                                        else cu_seqlens_q_padded // 2
                                    ),
                                    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                                    **fp8_meta_kwargs,
                                )
                                if fp8:
                                    softmax_lse_per_step[i], _, rng_states[i] = aux_ctx_tensors
                                else:
                                    softmax_lse_per_step[i], rng_states[i], *rest = aux_ctx_tensors
                                    attn_biases[i] = rest[0] if len(rest) > 0 else None
                            else:
                                if qkv_format == "thd":
                                    # [t, np, hn] -> [t/2, np, hn]
                                    q_inputs[i % 2] = tex.thd_read_half_tensor(
                                        q, cu_seqlens_q_padded, 1
                                    )
                                else:
                                    # [b, 2, sq//2, np, hn]->[b, sq//2, np, hn]->[b*sq//2, np, hn]
                                    q_inputs[i % 2] = (
                                        q[:, 1, ...].contiguous().view(-1, *q.shape[-2:])
                                    )
                                # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                                kv_inputs[i % 2] = kv_inputs[i % 2].view(2, -1, *k.shape[-2:])
                                if _flash_attn_2_3_plus:
                                    fa_optional_forward_kwargs["window_size"] = (-1, -1)
                                (
                                    _,
                                    _,
                                    _,
                                    _,
                                    out_per_step[i],
                                    softmax_lse_per_step[i],
                                    _,
                                    rng_states[i],
                                ) = _flash_attn_forward(
                                    q_inputs[i % 2],
                                    kv_inputs[i % 2][0],
                                    kv_inputs[i % 2][1],
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    max_seqlen_q // 2,
                                    max_seqlen_kv,
                                    dropout_p,
                                    softmax_scale,
                                    causal=False,
                                    return_softmax=False,
                                    **fa_optional_forward_kwargs,
                                )
                    else:
                        if pad_between_seqs_q:
                            cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                cu_seqlens_q, cu_seqlens_q_padded, cp_size, rank, True, True
                            )
                        else:
                            cu_seqlens_q_per_step[i] = cu_seqlens_q // cp_size
                        if pad_between_seqs_kv:
                            cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                cu_seqlens_kv,
                                cu_seqlens_kv_padded,
                                cp_size,
                                (rank - i) % cp_size,
                                True,
                                True,
                            )
                        else:
                            cu_seqlens_kv_per_step[i] = cu_seqlens_kv // cp_size
                        if use_fused_attention:
                            if attn_bias is not None:
                                idx = (rank - i) % cp_size
                                attn_bias_inputs[i % 2] = torch.cat(
                                    (
                                        attn_bias[..., idx, :],
                                        attn_bias[..., (2 * cp_size - idx - 1), :],
                                    ),
                                    dim=-1,
                                ).contiguous()
                            out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                is_training,
                                max_seqlen_q,
                                max_seqlen_kv,
                                cu_seqlens_q_per_step[i],
                                cu_seqlens_kv_per_step[i],
                                q,
                                (
                                    kv_inputs[i % 2][..., 0, :, :]
                                    if qkv_format in ["bshd", "sbhd"]
                                    else kv_inputs[i % 2][0]
                                ),
                                (
                                    kv_inputs[i % 2][..., 1, :, :]
                                    if qkv_format in ["bshd", "sbhd"]
                                    else kv_inputs[i % 2][1]
                                ),
                                fused_attn_qkv_dtype,
                                fused_attn_backend,
                                attn_scale=softmax_scale,
                                dropout=dropout_p,
                                qkv_layout=qkv_layout,
                                attn_mask_type=attn_mask_type,
                                attn_bias_type=attn_bias_type,
                                attn_bias=attn_bias_inputs[i % 2],
                                cu_seqlens_q_padded=cu_seqlens_q_padded,
                                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                                **fp8_meta_kwargs,
                            )
                            if fp8:
                                softmax_lse_per_step[i], _, rng_states[i] = aux_ctx_tensors
                            else:
                                softmax_lse_per_step[i], rng_states[i], *rest = aux_ctx_tensors
                                attn_biases[i] = rest[0] if len(rest) > 0 else None
                        else:
                            # [b, sq, np, hn] -> [b*sq, np, hn]
                            q_inputs[i % 2] = q.view(-1, *q.shape[-2:])
                            # [2, b, sk, np, hn] -> [2, b*sk, np, hn]
                            kv_inputs[i % 2] = kv_inputs[i % 2].view(2, -1, *k.shape[-2:])
                            (
                                _,
                                _,
                                _,
                                _,
                                out_per_step[i],
                                softmax_lse_per_step[i],
                                _,
                                rng_states[i],
                            ) = _flash_attn_forward(
                                q_inputs[i % 2],
                                kv_inputs[i % 2][0],
                                kv_inputs[i % 2][1],
                                cu_seqlens_q_per_step[i],
                                cu_seqlens_kv_per_step[i],
                                max_seqlen_q,
                                max_seqlen_kv,
                                dropout_p,
                                softmax_scale,
                                causal=False,
                                return_softmax=False,
                                **fa_optional_forward_kwargs,
                            )

            if i > 0:
                # wait until fwd restuls correction of last step is done
                if i > 1:
                    flash_attn_streams[(i - 1) % 2].wait_event(fwd_results_correction_done)

                if use_fused_attention:
                    # [b, np, sq, 1] -> [b, np, sq]
                    softmax_lse_per_step[i - 1].squeeze_(-1)

                with torch.cuda.stream(flash_attn_streams[(i - 1) % 2]):
                    if fp8:
                        out_per_step[i - 1] = cast_from_fp8(
                            out_per_step[i - 1],
                            fp8_meta["scaling_fwd"],
                            META_O_CP,
                            fp8_dtype_forward,
                            TE_DType[torch.float32],
                        )
                    if i == 1:
                        out = torch.zeros_like(q if not fp8 else out_per_step[0]).view(q.shape)
                        softmax_lse = torch.clone(softmax_lse_per_step[0]).to(torch.double)
                        if causal and qkv_format != "thd":
                            # [b, np, sq] -> [b, np, 2, sq//2]
                            softmax_lse_ = softmax_lse.view(
                                *softmax_lse.shape[:-1], 2, softmax_lse.shape[-1] // 2
                            )
                    elif (i - 1) <= rank or not causal:
                        flash_attn_fwd_softmax_lse_correction(
                            softmax_lse, softmax_lse_per_step[i - 1]
                        )
                    else:
                        if qkv_format == "thd":
                            tex.thd_second_half_lse_correction(
                                softmax_lse,
                                softmax_lse_per_step[i - 1],
                                cu_seqlens_q_padded,
                                max_seqlen_q,
                            )
                        else:
                            flash_attn_fwd_softmax_lse_correction(
                                softmax_lse_[..., 1, :], softmax_lse_per_step[i - 1]
                            )

                if i < cp_size:
                    flash_attn_streams[(i - 1) % 2].record_event(fwd_results_correction_done)

        torch.cuda.current_stream().wait_stream(flash_attn_streams[1])

        softmax_lse = softmax_lse.to(torch.float)
        for i in range(cp_size):
            if qkv_format == "bshd":
                out_per_step[i] = out_per_step[i].view(out.shape[0], -1, *out.shape[-2:])
                out_ = out[:, 1, ...]
            elif qkv_format == "sbhd":
                out_per_step[i] = out_per_step[i].view(-1, *out.shape[-3:])
                out_ = out[1]

            if i <= rank or not causal:
                if qkv_format in ["bshd", "sbhd"]:
                    flash_attn_fwd_out_correction(
                        out.view(*out_per_step[i].shape),
                        out_per_step[i],
                        seq_dim,
                        softmax_lse,
                        softmax_lse_per_step[i],
                    )
                elif qkv_format == "thd":
                    tex.thd_out_correction(
                        out,
                        out_per_step[i],
                        softmax_lse,
                        softmax_lse_per_step[i],
                        cu_seqlens_q_padded,
                        False,
                    )
            else:
                if qkv_format in ["bshd", "sbhd"]:
                    flash_attn_fwd_out_correction(
                        out_,
                        out_per_step[i],
                        seq_dim,
                        softmax_lse_[..., 1, :],
                        softmax_lse_per_step[i],
                    )
                elif qkv_format == "thd":
                    tex.thd_out_correction(
                        out,
                        out_per_step[i],
                        softmax_lse,
                        softmax_lse_per_step[i],
                        cu_seqlens_q_padded,
                        True,
                    )

        kv = p2p_comm_buffers[-1]
        if use_fused_attention:
            if qkv_format == "bshd":
                out = out.view(out.shape[0], -1, *out.shape[-2:])
            elif qkv_format == "sbhd":
                out = out.view(-1, *out.shape[-3:])
        else:
            out = out.view(-1, *out.shape[-2:])

        if fp8 and use_fused_attention:
            amax_cp_fwd = amax_per_step.amax(dim=1)
            fp8_meta["scaling_fwd"].amax_history[0][META_S] = amax_cp_fwd[0]
            fp8_meta["scaling_fwd"].amax_history[0][META_O_CP] = amax_cp_fwd[1]

        out_f16 = out.to(q_fp8.dtype if fp8 and fp8_meta["recipe"].fp8_mha else q_f16.dtype)
        if fp8 and (fp8_meta["recipe"].fp8_mha or int(os.getenv("NVTE_FP8_DPA_BWD", "1"))):
            out_fp8 = cast_to_fp8(out_f16, fp8_meta["scaling_fwd"], META_O, fp8_dtype_forward)

        if fp8 and fp8_meta["recipe"].fp8_mha:
            out_ret = Float8Tensor(
                data=out_fp8,
                fp8_meta=fp8_meta,
                fp8_meta_forward=True,
                fp8_meta_index=META_O,
                fp8_dtype=fp8_dtype_forward,
                dtype=q_fp8.dtype,
            )
        else:
            out_ret = out_f16

        if fp8 and int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
            q_save, kv_save, out_save = q, kv, out_fp8
            fp8_fwd_scales = fp8_meta["scaling_fwd"].scale.clone()
            fp8_fwd_scale_invs = fp8_meta["scaling_fwd"].scale_inv.clone()
        elif fp8 and fp8_meta["recipe"].fp8_mha:
            kv_fp8 = Float8Tensor(
                data=kv,
                fp8_meta=fp8_meta,
                fp8_meta_forward=True,
                fp8_meta_index=META_QKV,
                fp8_dtype=fp8_dtype_forward,
                dtype=k_fp8.dtype,
            )
            q_save, kv_save, out_save = q_fp8, kv_fp8, out_f16
            fp8_fwd_scales, fp8_fwd_scale_invs = None, None
        else:
            q_save, kv_save, out_save = q_f16, kv, out_f16
            fp8_fwd_scales, fp8_fwd_scale_invs = None, None

        ctx.save_for_backward(
            q_save,
            kv_save,
            out_save,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            fp8_fwd_scales,
            fp8_fwd_scale_invs,
            *cu_seqlens_q_per_step,
            *cu_seqlens_kv_per_step,
            *rng_states,
            *attn_biases,
        )
        ctx.cp_group = cp_group
        ctx.cp_global_ranks = cp_global_ranks
        ctx.dropout_p = dropout_p
        ctx.total_tokens_kv = total_tokens_kv
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_mask_type = attn_mask_type
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_bias_shape = None if attn_bias is None else attn_bias.shape
        ctx.deterministic = deterministic
        ctx.use_fused_attention = use_fused_attention
        ctx.fp8 = fp8 and int(os.getenv("NVTE_FP8_DPA_BWD", "1"))
        ctx.fp8_meta = fp8_meta
        return out_ret

    @staticmethod
    def backward(ctx, dout):
        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)
        send_dst = ctx.cp_global_ranks[(rank - 1) % cp_size]
        recv_src = ctx.cp_global_ranks[(rank + 1) % cp_size]
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (cp_size == 2)

        (q, kv, out, softmax_lse, cu_seqlens_q_padded, cu_seqlens_kv_padded) = ctx.saved_tensors[:6]
        (fp8_fwd_scales, fp8_fwd_scale_invs) = ctx.saved_tensors[6:8]
        cu_seqlens_q_per_step = ctx.saved_tensors[8 : 8 + cp_size]
        cu_seqlens_kv_per_step = ctx.saved_tensors[8 + cp_size : 8 + cp_size * 2]
        rng_states = ctx.saved_tensors[8 + cp_size * 2 : 8 + cp_size * 3]
        attn_biases = ctx.saved_tensors[8 + cp_size * 3 : 8 + cp_size * 4]

        causal = "causal" in ctx.attn_mask_type
        padding = "padding" in ctx.attn_mask_type
        if ctx.qkv_format in ["bshd", "sbhd"]:
            qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format[:-2] + "2" + ctx.qkv_format[-2:]
        else:
            qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format

        if attn_biases[0] is not None:
            # [b, np, sq, 2*cp, sk//(2*cp)]
            attn_dbias = torch.zeros(
                *ctx.attn_bias_shape, dtype=attn_biases[0].dtype, device=attn_biases[0].device
            )
            # [b, np, sq, 2*cp, sk//(2*cp)] -> [b, np, 2, sq//2, 2*cp, sk//(2*cp)]
            attn_dbias_ = attn_dbias.view(
                *attn_dbias.shape[:-3], 2, attn_dbias.shape[-3] // 2, *attn_dbias.shape[-2:]
            )
        else:
            attn_dbias = None

        if causal:
            if ctx.qkv_format == "thd":
                softmax_lse_ = tex.thd_read_second_half_lse(
                    softmax_lse, cu_seqlens_q_padded, ctx.max_seqlen_q
                )
            else:
                # [b, np, sq] -> [b, np, 2, sq//2]
                softmax_lse_ = softmax_lse.view(
                    *softmax_lse.shape[:-1], 2, softmax_lse.shape[-1] // 2
                )
                softmax_lse_ = softmax_lse_[..., 1, :].contiguous()
                if ctx.use_fused_attention:
                    # [b, np, sq//2] -> [b, np, sq//2, 1]
                    softmax_lse_.unsqueeze_(-1)
        if ctx.use_fused_attention:
            # [b, np, sq] -> [b, np, sq, 1]
            softmax_lse.unsqueeze_(-1)

        if ctx.fp8:
            if ctx.use_fused_attention:
                fp8_dtype_forward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=True)
                fp8_dtype_backward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=False)
                fused_attn_qkv_dtype = fp8_dtype_forward
                fused_attn_dqkv_dtype = fp8_dtype_backward
                fused_attn_backend = FusedAttnBackend["FP8"]
                dq_fp8 = torch.empty((cp_size, *q.shape), dtype=q.dtype, device=q.device)
                dkv_fp8 = torch.empty((cp_size, *kv.shape), dtype=kv.dtype, device=kv.device)
                dkv_fp8_ = torch.empty_like(dkv_fp8)
                dout_dtype = dout.dtype
                if ctx.fp8_meta["recipe"].fp8_mha:
                    assert isinstance(dout, Float8Tensor), "dout must be Float8Tensors for FP8 MHA!"
                    ctx.fp8_meta["scaling_bwd"].scale_inv[META_DO] = dout._scale_inv
                    dout = dout._data
                else:
                    dout = cast_to_fp8(
                        dout, ctx.fp8_meta["scaling_bwd"], META_DO, fp8_dtype_backward
                    )
                p2p_comm_buffers = [[kv, dkv_fp8], [torch.empty_like(kv), dkv_fp8_]]
                fp8_meta_kwargs = {}
                fp8_meta_kwargs["d_scale_qkv"] = fp8_fwd_scale_invs[META_QKV]
                fp8_meta_kwargs["d_scale_s"] = fp8_fwd_scale_invs[META_S]
                fp8_meta_kwargs["d_scale_o"] = fp8_fwd_scale_invs[META_O]
                fp8_meta_kwargs["d_scale_do"] = ctx.fp8_meta["scaling_bwd"].scale_inv[META_DO]
                fp8_meta_kwargs["d_scale_dp"] = ctx.fp8_meta["scaling_bwd"].scale_inv[META_DP]
                fp8_meta_kwargs["q_scale_s"] = fp8_fwd_scales[META_S]
                fp8_meta_kwargs["q_scale_dp"] = ctx.fp8_meta["scaling_bwd"].scale[META_DP]
                fp8_meta_kwargs["q_scale_dqkv"] = ctx.fp8_meta["scaling_bwd"].scale[META_DQKV_CP]
                amax_per_step = torch.zeros((2, cp_size), dtype=torch.float32, device=q.device)
            else:
                assert False, "FP8 is only supported with Fused Attention!"
        else:
            if ctx.fp8_meta is not None and ctx.fp8_meta["recipe"].fp8_mha:
                q, kv, dout = [x.from_float8(x.dtype) for x in [q, kv, dout]]
            dq = torch.empty_like(q)
            if ctx.qkv_format == "thd" and causal:
                dq[cu_seqlens_q_padded[-1] :].fill_(0)
            p2p_comm_buffers = [
                torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
                torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
            ]
            p2p_comm_buffers[0][0].copy_(kv)
            if ctx.use_fused_attention:
                fp8_meta_kwargs = {}
                fused_attn_qkv_dtype = TE_DType[q.dtype]
                fused_attn_dqkv_dtype = TE_DType[dout.dtype]
                fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        out = out.view(*q.shape)
        dout = dout.view(*q.shape)
        send_recv_reqs = []

        fa_optional_backward_kwargs = {}
        if _flash_attn_2_4_plus:
            fa_optional_backward_kwargs["alibi_slopes"] = None
        if _flash_attn_2_4_1_plus:
            fa_optional_backward_kwargs["deterministic"] = ctx.deterministic

        for i in range(cp_size):
            # wait until KV is received
            for req in send_recv_reqs:
                req.wait()

            send_tensor = p2p_comm_buffers[i % 2]
            recv_tensor = p2p_comm_buffers[(i + 1) % 2]
            if ctx.fp8:
                if i < cp_size - 1:
                    send_recv_reqs = flash_attn_p2p_communicate(
                        rank,
                        send_tensor[0],
                        send_dst,
                        recv_tensor[0],
                        recv_src,
                        ctx.cp_group,
                        batch_p2p_comm,
                    )
                else:
                    dkv_a2a_req = torch.distributed.all_to_all_single(
                        dkv_fp8,
                        dkv_fp8_,
                        group=ctx.cp_group,
                        async_op=True,
                    )
                    send_recv_reqs = [dkv_a2a_req]
            else:
                if i == 0:
                    send_tensor = send_tensor[0]
                    recv_tensor = recv_tensor[0]
                if i == (cp_size - 1):
                    send_tensor = send_tensor[1]
                    recv_tensor = recv_tensor[1]
                send_recv_reqs = flash_attn_p2p_communicate(
                    rank, send_tensor, send_dst, recv_tensor, recv_src, ctx.cp_group, batch_p2p_comm
                )

            kv = p2p_comm_buffers[i % 2][0]
            if ctx.fp8 and ctx.use_fused_attention:
                fp8_meta_kwargs["amax_dp"] = amax_per_step[0][i]
                fp8_meta_kwargs["amax_dqkv"] = amax_per_step[0][i]
            # In reversed order of fwd
            if causal:
                if i == (cp_size - 1):
                    if ctx.use_fused_attention:
                        if ctx.qkv_format == "bshd":
                            # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                            q_ = q.view(q.shape[0], -1, *q.shape[-2:])
                            # [b, 2, sk//2, 2, np, hn] -> [b, sk, 2, np, hn]
                            kv_ = kv.view(kv.shape[0], -1, *kv.shape[-3:])
                            # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                            out_ = out.view(out.shape[0], -1, *out.shape[-2:])
                            dout_ = dout.view(dout.shape[0], -1, *dout.shape[-2:])
                        elif ctx.qkv_format == "sbhd":
                            # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                            q_ = q.view(-1, *q.shape[-3:])
                            # [2, sk//2, b, 2, np, hn] -> [sk, b, 2, np, hn]
                            kv_ = kv.view(-1, *kv.shape[-4:])
                            # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                            out_ = out.view(-1, *out.shape[-3:])
                            dout_ = dout.view(-1, *dout.shape[-3:])
                        elif ctx.qkv_format == "thd":
                            q_, kv_, out_, dout_ = q, kv, out, dout
                        if ctx.fp8:
                            aux_ctx_tensors = [
                                softmax_lse,
                                softmax_lse,
                                rng_states[cp_size - i - 1],
                            ]
                        else:
                            aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                        if attn_dbias is not None:
                            aux_ctx_tensors += [attn_biases[cp_size - i - 1]]
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_kv,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            q_,
                            kv_[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[0],
                            kv_[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[1],
                            out_,
                            dout_,
                            fused_attn_qkv_dtype,
                            fused_attn_dqkv_dtype,
                            aux_ctx_tensors,
                            fused_attn_backend,
                            cu_seqlens_q_padded=cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type=ctx.attn_mask_type,
                            attn_bias_type=ctx.attn_bias_type,
                            deterministic=ctx.deterministic,
                            **fp8_meta_kwargs,
                        )
                    else:
                        # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                        q_ = q.view(-1, *q.shape[-2:])
                        dq_ = torch.zeros_like(q_)
                        # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                        kv_ = kv.view(2, -1, *kv.shape[-2:])
                        dkv_ = torch.empty_like(kv_)
                        # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                        out_ = out.view(-1, *out.shape[-2:])
                        dout_ = dout.view(-1, *dout.shape[-2:])
                        if _flash_attn_2_3_plus:
                            fa_optional_backward_kwargs["window_size"] = (-1, 0)
                        _flash_attn_backward(
                            dout_,
                            q_,
                            kv_[0],
                            kv_[1],
                            out_,
                            softmax_lse,
                            dq_,
                            dkv_[0],
                            dkv_[1],
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_kv,
                            ctx.dropout_p,
                            ctx.softmax_scale,
                            True,
                            rng_state=rng_states[cp_size - i - 1],
                            **fa_optional_backward_kwargs,
                        )
                elif i >= (cp_size - rank - 1):
                    if ctx.use_fused_attention:
                        if ctx.qkv_format == "bshd":
                            # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                            q_ = q.view(q.shape[0], -1, *q.shape[-2:])
                            # [b, 2, sk//2, 2, np, hn] -> [b, sk//2, 2, np, hn]
                            kv_ = kv[:, 0, ...].contiguous()
                            # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                            out_ = out.view(out.shape[0], -1, *out.shape[-2:])
                            dout_ = dout.view(dout.shape[0], -1, *dout.shape[-2:])
                        elif ctx.qkv_format == "sbhd":
                            # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                            q_ = q.view(-1, *q.shape[-3:])
                            # [2, sk//2, b, 2, np, hn] -> [sk//2, b, 2, np, hn]
                            kv_ = kv[0].contiguous()
                            # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                            out_ = out.view(-1, *out.shape[-3:])
                            dout_ = dout.view(-1, *dout.shape[-3:])
                        elif ctx.qkv_format == "thd":
                            q_, out_, dout_ = q, out, dout
                            # [2, t, np, hn] -> [2, t/2, np, hn]
                            kv_ = tex.thd_read_half_tensor(kv, cu_seqlens_kv_padded, 0)
                        if ctx.fp8:
                            aux_ctx_tensors = [
                                softmax_lse,
                                softmax_lse,
                                rng_states[cp_size - i - 1],
                            ]
                        else:
                            aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                        if attn_dbias is not None:
                            aux_ctx_tensors += [attn_biases[cp_size - i - 1]]
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_kv // 2,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            q_,
                            kv_[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[0],
                            kv_[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[1],
                            out_,
                            dout_,
                            fused_attn_qkv_dtype,
                            fused_attn_dqkv_dtype,
                            aux_ctx_tensors,
                            fused_attn_backend,
                            cu_seqlens_q_padded=cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=(
                                None if cu_seqlens_kv_padded is None else cu_seqlens_kv_padded // 2
                            ),
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type="padding" if padding else "no_mask",
                            attn_bias_type=ctx.attn_bias_type,
                            deterministic=ctx.deterministic,
                            **fp8_meta_kwargs,
                        )
                    else:
                        # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                        q_ = q.view(-1, *q.shape[-2:])
                        dq_ = torch.zeros_like(q_)
                        if ctx.qkv_format == "thd":
                            # [2, t, np, hn] -> [2, t/2, np, hn]
                            kv_ = tex.thd_read_half_tensor(kv, cu_seqlens_kv_padded, 0)
                        else:
                            # [2, b, 2, sk//2, np, hn]->[2, b, sk//2, np, hn]->[2, b*sk//2, np, hn]
                            kv_ = kv[:, :, 0, ...].contiguous().view(2, -1, *kv.shape[-2:])
                        dkv_ = torch.empty_like(kv_)
                        # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                        out_ = out.view(-1, *out.shape[-2:])
                        dout_ = dout.view(-1, *dout.shape[-2:])
                        if _flash_attn_2_3_plus:
                            fa_optional_backward_kwargs["window_size"] = (-1, -1)
                        _flash_attn_backward(
                            dout_,
                            q_,
                            kv_[0],
                            kv_[1],
                            out_,
                            softmax_lse,
                            dq_,
                            dkv_[0],
                            dkv_[1],
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_kv // 2,
                            ctx.dropout_p,
                            ctx.softmax_scale,
                            False,
                            rng_state=rng_states[cp_size - i - 1],
                            **fa_optional_backward_kwargs,
                        )
                else:
                    if ctx.use_fused_attention:
                        if ctx.qkv_format == "bshd":
                            # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
                            q_ = q[:, 1, ...].contiguous()
                            # [b, 2, sk//2, 2, np, hn] -> [b, sk, 2, np, hn]
                            kv_ = kv.view(kv.shape[0], -1, *kv.shape[-3:])
                            # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
                            out_ = out[:, 1, ...].contiguous()
                            dout_ = dout[:, 1, ...].contiguous()
                        elif ctx.qkv_format == "sbhd":
                            # [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
                            q_ = q[1].contiguous()
                            # [2, sk//2, b, 2, np, hn] -> [sk, b, 2, np, hn]
                            kv_ = kv.view(-1, *kv.shape[-4:])
                            # [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
                            out_ = out[1].contiguous()
                            dout_ = dout[1].contiguous()
                        elif ctx.qkv_format == "thd":
                            # [t, np, hn] -> [t/2, np, hn]
                            q_ = tex.thd_read_half_tensor(q, cu_seqlens_q_padded, 1)
                            out_ = tex.thd_read_half_tensor(out, cu_seqlens_q_padded, 1)
                            dout_ = tex.thd_read_half_tensor(dout, cu_seqlens_q_padded, 1)
                            kv_ = kv
                        if ctx.fp8:
                            aux_ctx_tensors = [
                                softmax_lse_,
                                softmax_lse_,
                                rng_states[cp_size - i - 1],
                            ]
                        else:
                            aux_ctx_tensors = [softmax_lse_, rng_states[cp_size - i - 1]]
                        if attn_dbias is not None:
                            aux_ctx_tensors += [attn_biases[cp_size - i - 1]]
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q // 2,
                            ctx.max_seqlen_kv,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            q_,
                            kv_[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[0],
                            kv_[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[1],
                            out_,
                            dout_,
                            fused_attn_qkv_dtype,
                            fused_attn_dqkv_dtype,
                            aux_ctx_tensors,
                            fused_attn_backend,
                            cu_seqlens_q_padded=(
                                None if cu_seqlens_q_padded is None else cu_seqlens_q_padded // 2
                            ),
                            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type="padding" if padding else "no_mask",
                            attn_bias_type=ctx.attn_bias_type,
                            deterministic=ctx.deterministic,
                            **fp8_meta_kwargs,
                        )
                    else:
                        if ctx.qkv_format == "thd":
                            # [t, np, hn] -> [t/2, np, hn]
                            q_ = tex.thd_read_half_tensor(q, cu_seqlens_q_padded, 1)
                        else:
                            # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn] -> [b*sq//2, np, hn]
                            q_ = q[:, 1, ...].contiguous().view(-1, *q.shape[-2:])
                        dq_ = torch.zeros_like(q_)
                        # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                        kv_ = kv.view(2, -1, *kv.shape[-2:])
                        dkv_ = torch.empty_like(kv_)
                        if ctx.qkv_format == "thd":
                            out_ = tex.thd_read_half_tensor(out, cu_seqlens_q_padded, 1)
                            dout_ = tex.thd_read_half_tensor(dout, cu_seqlens_q_padded, 1)
                        else:
                            # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn] -> [b*sq//2, np, hn]
                            out_ = out[:, 1, ...].contiguous().view(-1, *out.shape[-2:])
                            dout_ = dout[:, 1, ...].contiguous().view(-1, *dout.shape[-2:])
                        if _flash_attn_2_3_plus:
                            fa_optional_backward_kwargs["window_size"] = (-1, -1)
                        _flash_attn_backward(
                            dout_,
                            q_,
                            kv_[0],
                            kv_[1],
                            out_,
                            softmax_lse_,
                            dq_,
                            dkv_[0],
                            dkv_[1],
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            ctx.max_seqlen_q // 2,
                            ctx.max_seqlen_kv,
                            ctx.dropout_p,
                            ctx.softmax_scale,
                            False,
                            rng_state=rng_states[cp_size - i - 1],
                            **fa_optional_backward_kwargs,
                        )
            else:
                if ctx.use_fused_attention:
                    if ctx.fp8:
                        aux_ctx_tensors = [softmax_lse, softmax_lse, rng_states[cp_size - i - 1]]
                    else:
                        aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                    if attn_dbias is not None:
                        aux_ctx_tensors += [attn_biases[cp_size - i - 1]]
                    dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                        ctx.max_seqlen_q,
                        ctx.max_seqlen_kv,
                        cu_seqlens_q_per_step[cp_size - i - 1],
                        cu_seqlens_kv_per_step[cp_size - i - 1],
                        q,
                        kv[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv[0],
                        kv[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv[1],
                        out,
                        dout,
                        fused_attn_qkv_dtype,
                        fused_attn_dqkv_dtype,
                        aux_ctx_tensors,
                        fused_attn_backend,
                        cu_seqlens_q_padded=cu_seqlens_q_padded,
                        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                        attn_scale=ctx.softmax_scale,
                        dropout=ctx.dropout_p,
                        qkv_layout=qkv_layout,
                        attn_mask_type=ctx.attn_mask_type,
                        attn_bias_type=ctx.attn_bias_type,
                        deterministic=ctx.deterministic,
                        **fp8_meta_kwargs,
                    )
                else:
                    # [b, sq, np, hn] -> [b*sq, np, hn]
                    q_ = q.view(-1, *q.shape[-2:])
                    dq_ = torch.zeros_like(q_)
                    # [2, b, sk, np, hn] -> [2, b*sk, np, hn]
                    kv_ = kv.view(2, -1, *kv.shape[-2:])
                    dkv_ = torch.empty_like(kv_)
                    # [b, sq, np, hn] -> [b*sq, np, hn]
                    out_ = out.view(-1, *out.shape[-2:])
                    dout_ = dout.view(-1, *dout.shape[-2:])
                    if _flash_attn_2_3_plus:
                        fa_optional_backward_kwargs["window_size"] = (-1, -1)
                    _flash_attn_backward(
                        dout_,
                        q_,
                        kv_[0],
                        kv_[1],
                        out_,
                        softmax_lse,
                        dq_,
                        dkv_[0],
                        dkv_[1],
                        cu_seqlens_q_per_step[cp_size - i - 1],
                        cu_seqlens_kv_per_step[cp_size - i - 1],
                        ctx.max_seqlen_q,
                        ctx.max_seqlen_kv,
                        ctx.dropout_p,
                        ctx.softmax_scale,
                        False,
                        rng_state=rng_states[cp_size - i - 1],
                        **fa_optional_backward_kwargs,
                    )

            if ctx.fp8:
                dq = dq_fp8[(rank + i + 1) % cp_size]
            if i >= (cp_size - rank - 1) or not causal:
                # [b*sq, np, hn] -> [b, 2, sq//2, np, hn] if causal
                # [b*sq, np, hn] -> [b, sq, np, hn] if not causal
                dq_ = dq_.view(*dq.shape)
            else:
                if ctx.qkv_format == "bshd":
                    # [b*sq//2, np, hn] -> [b, sq//2, np, hn]
                    dq_ = dq_.view(dq.shape[0], *dq.shape[2:])
                elif ctx.qkv_format == "sbhd":
                    # [b*sq//2, np, hn] -> [sq//2, b, np, hn]
                    dq_ = dq_.view(-1, *dq.shape[-3:])

            if ctx.fp8:
                if i >= (cp_size - rank - 1) or not causal:
                    dq.copy_(dq_)
                else:
                    if ctx.qkv_format == "bshd":
                        dq[:, 0, ...].fill_(0)
                        dq[:, 1, ...].copy_(dq_)
                    elif ctx.qkv_format == "sbhd":
                        dq[0].fill_(0)
                        dq[1].copy_(dq_)
            elif causal:
                if i > (cp_size - rank - 1):
                    dq.add_(dq_)
                elif i == (cp_size - rank - 1):
                    if rank == (cp_size - 1):
                        dq.copy_(dq_)
                    else:
                        if ctx.qkv_format == "bshd":
                            dq[:, 0, ...].copy_(dq_[:, 0, ...])
                            dq[:, 1, ...].add_(dq_[:, 1, ...])
                        elif ctx.qkv_format == "sbhd":
                            dq[0].copy_(dq_[0])
                            dq[1].add_(dq_[1])
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dq, dq_, cu_seqlens_q_padded, "copy", "add")
                elif i > 0:
                    if ctx.qkv_format == "bshd":
                        dq[:, 1, ...].add_(dq_)
                    elif ctx.qkv_format == "sbhd":
                        dq[1].add_(dq_)
                    elif ctx.qkv_format == "thd":
                        tex.thd_grad_correction(dq, dq_, cu_seqlens_q_padded, "none", "add")
                else:
                    if ctx.qkv_format == "bshd":
                        dq[:, 1, ...].copy_(dq_)
                    elif ctx.qkv_format == "sbhd":
                        dq[1].copy_(dq_)
                    elif ctx.qkv_format == "thd":
                        tex.thd_grad_correction(dq, dq_, cu_seqlens_q_padded, "none", "copy")
            else:
                if i == 0:
                    dq.copy_(dq_)
                else:
                    dq.add_(dq_)

            if attn_dbias is not None:
                idx = (rank + i + 1) % cp_size
                if i == (cp_size - 1) or not causal:
                    # [b, np, sq, sk//cp] -> [b, np, sq, 2, sk//(2*cp)]
                    dbias_ = dbias_.view(*dbias_.shape[:-1], 2, dbias_.shape[-1] // 2)
                    attn_dbias[..., idx, :].copy_(dbias_[..., 0, :])
                    attn_dbias[..., (2 * cp_size - idx - 1), :].copy_(dbias_[..., 1, :])
                elif i >= (cp_size - rank - 1):
                    # [b, np, sq, sk//(2*cp)]
                    attn_dbias[..., idx, :].copy_(dbias_)
                else:
                    # [b, np, sq//2, sk//cp] -> [b, np, sq//2, 2, sk//(2*cp)]
                    dbias_ = dbias_.view(*dbias_.shape[:-1], 2, dbias_.shape[-1] // 2)
                    attn_dbias_[..., 1, :, idx, :].copy_(dbias_[..., 0, :])
                    attn_dbias_[..., 1, :, (2 * cp_size - idx - 1), :].copy_(dbias_[..., 1, :])

            # wait until dKV is received
            for req in send_recv_reqs:
                req.wait()

            if ctx.fp8:
                if i < cp_size - 1:
                    dkv = dkv_fp8_[(rank + i + 1) % cp_size]
                else:
                    dkv = dkv_fp8[(rank + i + 1) % cp_size]
            else:
                dkv = p2p_comm_buffers[(i + 1) % 2][1]
            if ctx.use_fused_attention:
                dkv_ = torch.cat((dk_.unsqueeze(0), dv_.unsqueeze(0)), dim=0)
                if ctx.qkv_format in ["bshd", "sbhd"]:
                    # [b, 2, sk//2, 2, np, hn] -> [2, b, 2, sk//2, np, hn] or
                    # [2, sk//2, b, 2, np, hn] -> [2, 2, sk//2, b, np, hn]
                    dkv = dkv.view(2, *dkv.shape[0:-3], *dkv.shape[-2:])
            if causal and i >= (cp_size - rank - 1) and i != (cp_size - 1):
                if ctx.qkv_format == "bshd":
                    # [2, b*sk//2, np, hn] -> [2, b, sk//2, np, hn]
                    dkv_ = dkv_.view(*dkv.shape[0:2], *dkv.shape[3:])
                elif ctx.qkv_format == "sbhd":
                    # [2, b*sk//2, np, hn] -> [2, sk//2, b, np, hn]
                    dkv_ = dkv_.view(dkv.shape[0], -1, *dkv.shape[-3:])
            else:
                # [2, b*sk, np, hn] -> [2, b, 2, sk//2, np, hn] if causal
                # [2, b*sk, np, hn] -> [2, b, sk, np, hn] if not causal
                dkv_ = dkv_.view(*dkv.shape)

            if ctx.fp8:
                if causal and i >= (cp_size - rank - 1) and i != (cp_size - 1):
                    if ctx.qkv_format == "bshd":
                        dkv[:, :, 0, ...].copy_(dkv_)
                        dkv[:, :, 1, ...].fill_(0)
                    elif ctx.qkv_format == "sbhd":
                        dkv[:, 0, ...].copy_(dkv_)
                        dkv[:, 1, ...].fill_(0)
                else:
                    dkv.copy_(dkv_)
            elif causal:
                if i == (cp_size - 1):
                    if rank == 0:
                        if ctx.qkv_format == "bshd":
                            dkv[:, :, 0, ...].add_(dkv_[:, :, 0, ...])
                            dkv[:, :, 1, ...].copy_(dkv_[:, :, 1, ...])
                        elif ctx.qkv_format == "sbhd":
                            dkv[:, 0, ...].add_(dkv_[:, 0, ...])
                            dkv[:, 1, ...].copy_(dkv_[:, 1, ...])
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dkv, dkv_, cu_seqlens_kv_padded, "add", "copy")
                    else:
                        dkv.add_(dkv_)
                elif i >= (cp_size - rank - 1):
                    if i == 0 and rank == (cp_size - 1):
                        if ctx.qkv_format == "bshd":
                            dkv[:, :, 0, ...].copy_(dkv_)
                        elif ctx.qkv_format == "sbhd":
                            dkv[:, 0, ...].copy_(dkv_)
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dkv, dkv_, cu_seqlens_kv_padded, "copy", "none")
                    else:
                        if ctx.qkv_format == "bshd":
                            dkv[:, :, 0, ...].add_(dkv_)
                        elif ctx.qkv_format == "sbhd":
                            dkv[:, 0, ...].add_(dkv_)
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dkv, dkv_, cu_seqlens_kv_padded, "add", "none")
                elif i > 0:
                    dkv.add_(dkv_)
                else:
                    dkv.copy_(dkv_)
            else:
                if i == 0:
                    dkv.copy_(dkv_)
                else:
                    dkv.add_(dkv_)

        if ctx.fp8 and ctx.use_fused_attention:
            amax_cp_bwd = amax_per_step.amax(dim=1)
            ctx.fp8_meta["scaling_bwd"].amax_history[0][META_DP] = amax_cp_bwd[0]
            ctx.fp8_meta["scaling_bwd"].amax_history[0][META_DQKV_CP] = amax_cp_bwd[1]
            if ctx.qkv_format in ["bshd", "sbhd"]:
                # [cp, b, 2, sk//2, 2, np, hn] -> [cp, 2, b, 2, sk//2, np, hn] or
                # [cp, 2, sk//2, b, 2, np, hn] -> [cp, 2, 2, sk//2, b, np, hn]
                dkv_fp8 = dkv_fp8.view(cp_size, 2, *dkv_fp8.shape[1:-3], *dkv_fp8.shape[-2:])
            dq, dkv = [
                cast_from_fp8(
                    x,
                    ctx.fp8_meta["scaling_bwd"],
                    META_DQKV_CP,
                    fp8_dtype_backward,
                    TE_DType[torch.float32],
                )
                for x in [dq_fp8, dkv_fp8]
            ]
            dq, dkv = [x.sum(dim=0).to(dout_dtype) for x in [dq, dkv]]

        if causal:
            if ctx.qkv_format == "bshd":
                # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                dq = dq.view(dq.shape[0], -1, *dq.shape[-2:])
                # [2, b, 2, sk//2, np, hn] -> [2, b, sk, np, hn]
                dkv = dkv.view(*dkv.shape[0:2], -1, *dkv.shape[-2:])
            elif ctx.qkv_format == "sbhd":
                # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                dq = dq.view(-1, *dq.shape[-3:])
                # [2, 2, sk//2, b, np, hn] -> [2, sk, b, np, hn]
                dkv = dkv.view(dkv.shape[0], -1, *dkv.shape[-3:])

        if ctx.qkv_format == "thd":
            dkv_ = torch.empty(
                2, ctx.total_tokens_kv, *dkv.shape[-2:], dtype=dkv.dtype, device=dkv.device
            )
            dkv_[:, : cu_seqlens_kv_padded[-1]].copy_(dkv)
            dkv_[:, cu_seqlens_kv_padded[-1] :].fill_(0)
            dkv = dkv_

        if ctx.fp8 and ctx.fp8_meta["recipe"].fp8_mha:
            dq, dkv = [
                cast_to_fp8(x, ctx.fp8_meta["scaling_bwd"], META_DQKV, fp8_dtype_backward)
                for x in [dq, dkv]
            ]
            dq, dk, dv = [
                Float8Tensor(
                    data=x,
                    fp8_meta=ctx.fp8_meta,
                    fp8_meta_forward=False,
                    fp8_meta_index=META_DQKV,
                    fp8_dtype=fp8_dtype_backward,
                    dtype=dout_dtype,
                )
                for x in [dq, dkv[0], dkv[1]]
            ]
        else:
            dk, dv = dkv[0], dkv[1]

        if attn_dbias is not None:
            # [b, np, sq, 2*cp, sk//(2*cp)] -> [b, np, sq, sk]
            attn_dbias = attn_dbias.view(*attn_dbias.shape[:-2], -1)

        return (
            None,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            attn_dbias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )



def attn_forward_func_with_cp(
    is_training,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    dropout_p,
    cp_group,
    cp_global_ranks,
    cp_stream,
    cp_comm_type="p2p",
    softmax_scale=None,
    qkv_format="bshd",
    attn_mask_type="causal",
    attn_bias_type="no_bias",
    attn_bias=None,
    deterministic=False,
    use_fused_attention=False,
    window_size=None,
    fp8=False,
    fp8_meta=None,
) -> torch.Tensor:
    """
    Attention implementation with context parallelism.
    """

    assert qkv_format in [
        "bshd",
        "sbhd",
        "thd",
    ], f"QKV format of {qkv_format} is not supported with context parallelism!"
    assert (
        qkv_format != "sbhd" or use_fused_attention
    ), "FlashAttention does not support sbhd format!"
    assert (
        qkv_format != "thd"
        or not use_fused_attention
        or attn_mask_type in ["padding", "padding_causal"]
    ), (
        f"Context parallelism is not supported for {attn_mask_type} mask type and "
        f"{qkv_format} format with {'FusedAttention' if use_fused_attention else 'FlashAttention'}!"
    )
    assert attn_bias is None or (use_fused_attention and "padding" not in attn_mask_type), (
        """Attention bias is only supported with FusedAttention and "causal" """
        """or "no_mask" mask types!"""
    )
    assert (
        cu_seqlens_q_padded is not None and cu_seqlens_kv_padded is not None
    ), "cu_seqlens_q_padded and cu_seqlens_kv_padded cannot be None with context parallelism!"

    sliding_window_attn = (
        window_size is not None and window_size != (-1, 0) and window_size != (-1, -1)
    )
    assert (
        not sliding_window_attn
        or cp_comm_type == "a2a"
        or (cp_comm_type == "all_gather" and not use_fused_attention)
    ), "The context parallel running configs cannot support sliding window attetnion!"

    args = [
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        softmax_scale,
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
    ]
    assert cp_comm_type == "p2p"
    args += [fp8, fp8_meta, cp_group, cp_global_ranks, cp_stream]
    out = AttnFuncWithCPAndKVP2P.apply(*args)

    return out