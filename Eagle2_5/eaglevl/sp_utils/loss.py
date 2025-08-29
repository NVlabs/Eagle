"""Low memory cross entropy without materilizing the logits

This module enables long-context training of large vocab models, e.g., Gemma has 250K vocab and Llama 3 has 150K

Yao Fu, University of Edinburgh
yao.fu@ed.ac.uk
"""

import torch
import torch.nn.functional as F


def cross_ent_normal(x, weight, labels):
    logits = torch.einsum("bsh, vh -> bsv", x, weight)
    vocab = weight.size(0)
    loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1))
    return loss

class LowMemLogitProjCrossEnt(torch.autograd.Function):
    """Low memory implementation of logits projection plus cross entropy loss. 
    Useful for reducing the peak memory when dealing with vocabulary larger than 100000

    TODO: integrate this function into easy context

    Two tricks used here 
    1. Shard the data to reduce peak memory 
    2. Do not save the logits 
    """

    @staticmethod
    # @torch.compile() # Currently we do not use torch.compile because it uses additional memory
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, sp: int=4):
        """
        Args:
            x: size = [batch, seqlen, hidden]
            weight: size = [vocab, hidden]
            labels: size = [batch, seqlen]
        """
        bsz, seqlen, hidden = x.size()
        vocab = weight.size(0)
        micro_seqlen = seqlen // sp
        
        loss = 0
        for i in range(sp): # shard data along the sequence dimension
            logits_i_slice = torch.einsum("bsh, vh -> bsv", x[:, micro_seqlen * i: micro_seqlen * (i + 1)], weight)
            loss_i = F.cross_entropy(logits_i_slice.view(-1, vocab), labels[:, micro_seqlen * i: micro_seqlen * (i + 1)].view(-1))
            loss = loss + loss_i

        loss = loss / sp
        ctx.save_for_backward(x, weight, labels) # because we do no save logits, we save memory
        ctx.sp = sp
        return loss

    # @torch.compile()
    @staticmethod
    def backward(ctx, grad_output):
        """Manually calculate the gradient in a memory-efficient way
        Ref: https://indii.org/blog/gradients-of-softmax-and-logsumexp/
        """
        x, weight, labels = ctx.saved_tensors
        sp = ctx.sp
        device = x.device
        dtype = x.dtype
        bsz, seqlen, hidden = x.size()
        vocab, hidden = weight.size()
        micro_seqlen = seqlen // sp

        d_weight = torch.zeros_like(weight, device=weight.device)
        d_x = []
        for i in range(sp): # shard data along sequence dimension, reduce peak memory 
            x_ = x[:, micro_seqlen * i: micro_seqlen * (i + 1)]
            p = F.softmax(
                    torch.einsum("blh, vh -> blv", x_, weight), 
                    dim=-1
                    )

            # memory efficient in-place backprop
            # loss -> d_logits
            d_logits = -p.view(-1) # [b * l * v] 
            labels_ = labels[:, micro_seqlen * i: micro_seqlen * (i + 1)].view(-1) # [b * l]
            index = torch.arange(bsz * micro_seqlen, device=device) * vocab + labels_
            source = torch.tensor([1] * bsz * micro_seqlen, dtype=dtype, device=device)
            d_logits.index_add_(0, index, source)
            d_logits = -d_logits.view(bsz, micro_seqlen, vocab) / (bsz * seqlen)

            # d_logits -> d_x and d_weight
            d_x.append(torch.einsum("blv, vh -> blh", d_logits, weight))
            d_weight += torch.einsum("blv, blh -> vh", d_logits, x_)
        
        d_weight = grad_output * d_weight
        d_x = grad_output * torch.concat(d_x, 1)
        return d_x, d_weight, None, None

low_mem_cross_ent = LowMemLogitProjCrossEnt.apply