# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import copy
from typing import Any, Tuple

import deepspeed.comm as dist
import torch
import torch.distributed as torch_dist
from flash_attn import flash_attn_func
from torch import Tensor
from torch.nn import Module

from llava.train.sequence_parallel.globals import get_ulysess_seq_len, get_ulysess_sp_rank, get_ulysess_sp_size

from .all_to_all import SeqAllGather, SeqAllToAll4D, SeqAllToAll5D

# def single_all_to_all(input, scatter_idx, gather_idx, group):
#     seq_world_size = dist.get_world_size(group)
#     inp_shape = list(input.shape)
#     inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
#     if scatter_idx < 2:
#         input_t = input.reshape([seq_world_size, inp_shape[scatter_idx]] + inp_shape[scatter_idx + 1 :]).contiguous()
#     else:
#         # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
#         input_t = (
#             input.reshape([-1, seq_world_size, inp_shape[scatter_idx]] + inp_shape[scatter_idx + 1 :])
#             .transpose(0, 1)
#             .contiguous()
#         )

#     output = torch.empty_like(input_t)
#     dist.all_to_all_single(output, input_t, group=group)

#     # if scattering the seq-dim, transpose the heads back to the original dimension
#     if scatter_idx < 2:
#         output = output.transpose(0, 1).contiguous()

#     return output.reshape(
#         inp_shape[:gather_idx]
#         + [
#             inp_shape[gather_idx] * seq_world_size,
#         ]
#         + inp_shape[gather_idx + 1 :]
#     ).contiguous()


# class _SeqAllToAll(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:

#         ctx.group = group
#         ctx.scatter_idx = scatter_idx
#         ctx.gather_idx = gather_idx

#         return single_all_to_all(input, scatter_idx, gather_idx, group)

#     @staticmethod
#     def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
#         return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


class UlyssesAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
    ) -> None:

        super().__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *args: Any,
        attention_mask=None,
        dropout_p=0.0,
        softmax_scale=None,
        seqlens_in_batch=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]
        # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)

        # scatter 2, gather 1
        q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx)
        if attention_mask is not None:
            local_attention_mask = copy.deepcopy(attention_mask)
            shard_seqlen = local_attention_mask.size(1)
            ulysess_seq_len = get_ulysess_seq_len()
            max_global_length = max(ulysess_seq_len)
            global_attention_mask_list = []
            for i in range(get_ulysess_sp_size()):
                if i == get_ulysess_sp_rank():
                    global_attention_mask_list.append(
                        torch.cat(
                            [
                                local_attention_mask,
                                torch.zeros(
                                    (local_attention_mask.size(0), max_global_length - shard_seqlen),
                                    dtype=local_attention_mask.dtype,
                                    device=local_attention_mask.device,
                                ),
                            ],
                            dim=1,
                        )
                    )
                else:
                    global_attention_mask_list.append(
                        torch.zeros(
                            (local_attention_mask.size(0), max_global_length),
                            dtype=local_attention_mask.dtype,
                            device=local_attention_mask.device,
                        )
                    )

            global_attention_mask = torch.stack(global_attention_mask_list, dim=0)
            # print("Before all reduce Rank {} global_attention_mask: {}".format(get_ulysess_sp_rank(), global_attention_mask.size()))
            torch_dist.all_reduce(global_attention_mask, group=self.spg)
            # print("After all reduce Rank {} global_attention_mask: {}".format(get_ulysess_sp_rank(), global_attention_mask.size()))
            torch_dist.barrier(group=self.spg)
            new_global_attention_mask_list = list(torch.unbind(global_attention_mask, dim=0))
            # Unpad the global attention mask list and concatenate them
            for i in range(len(new_global_attention_mask_list)):
                new_global_attention_mask_list[i] = new_global_attention_mask_list[i][:, : ulysess_seq_len[i]]
            global_attention_mask = torch.cat(new_global_attention_mask_list, dim=1)
            context_layer = self.local_attn(
                q,
                k,
                v,
                *args,
                attention_mask=global_attention_mask,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                seqlens_in_batch=seqlens_in_batch,
                causal=causal,
            )
        else:
            context_layer = self.local_attn(
                q,
                k,
                v,
                *args,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                # window_size=window_size,
                # alibi_slopes=alibi_slopes,
                # deterministic=deterministic,
                # return_attn_probs=return_attn_probs,
            )
        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)

        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)

        # out e.g., [s/p::h]
        return output
