# SPDX-License-Identifier: Apache-2.0
"""
feature_extractor_wrapper.py
============================

Phase 1 (Vision) of the LocateAnything-3B C++/TensorRT port.

This module wraps **MoonViT + mlp1** into a single ``nn.Module`` whose
``forward`` is *exportable to ONNX at one pinned geometry* (N == 1, one
resolution bucket). The export target is a single TRT engine

    ENGINE 1:  pixel_values[L, 3, 14, 14]  ->  vit_embeds[L/4, H_llm]   (LLM space)

The wrapper resolves, at trace time, every data-dependent / non-exportable
construct in the eager MoonViT + LocateAnything connector code:

  * ``cu_seqlens`` degenerates to ``[0, L]`` (N==1) -> plain full self-attention,
    **no attention mask** is needed or emitted.
  * The learnable **bicubic absolute positional embedding** (interpolated from a
    64x64 grid) is resolved once for the bucket's (h, w) grid and **baked as a
    constant buffer** (``self.pos_emb_baked``), so the ONNX never contains a
    dynamic-output-size ``F.interpolate(mode="bicubic")``.
  * The **2D-RoPE** ``freqs_cis`` (per token, 36 complex angles -> here stored as
    cos/sin of length 72 over the interleaved real layout) is **baked as cos/sin
    constant buffers** for the bucket, and ``apply_rope`` is **rewritten from
    ``torch.view_as_complex`` / ``torch.polar`` to real interleaved cos/sin
    arithmetic**. The complex ops have no TRT export path; the real form here is
    chosen to be *numerically consistent with the C++ real-arithmetic RoPE* in
    ``cpp_trt/src/vision_rope/`` (see "RoPE convention" below).
  * The **2x2 patch-merger** becomes a static reshape/permute (no Python loop).
  * The patch-embed ``Conv2d(3, 1152, k=14, s=14)`` is left as an ``nn.Conv2d``
    on a ``[L, 3, 14, 14]`` input; with kernel == stride == input spatial size it
    is mathematically a ``Gemm(588 -> 1152)`` and ONNX/TRT will fold it to one.
  * **mlp1** (the LocateAnything connector, *no pixel-shuffle*) is applied so the
    ONNX emits LLM-space ``vit_embeds`` directly:
        LayerNorm(4608) -> Linear(4608 -> H_llm) -> GELU -> Linear(H_llm -> H_llm)

The wrapper does **not** run here (no checkpoint / torch on this box). It is
written against the real checkpoint and is meant to be loaded + exported on a
Blackwell machine via ``export_vision_onnx.py``.

------------------------------------------------------------------------------
RoPE convention (matches eaglevl/model/moon_vit/modeling_vit.py AND
cpp_trt/src/vision_rope/ exactly -- both re-read from source)
------------------------------------------------------------------------------
MoonViT eager code (``apply_rope`` + ``Rope2DPosEmb._precompute_freqs_cis``)
applies 2D-RoPE as a *complex* rotation over ADJACENT channel pairs:

    q_  = view_as_complex(q.float().view(..., -1, 2))   # slot s = (q[2s], q[2s+1])
    q_  = q_ * freqs_cis                                 # freqs_cis = polar(1, theta)
    q   = view_as_real(q_).flatten(-2)

The rotation table ``freqs_cis`` has ``head_dim/2`` complex slots per token and
is built by INTERLEAVING THE TWO SPATIAL AXES per frequency index ``i``:

    x_pos = flat_pos %  grid_w     # COLUMN / width  index of the token
    y_pos = flat_pos // grid_w     # ROW    / height index of the token
    inv_freq[i] = theta^(-4*i / head_dim)          for i in [0, head_dim//4)
    slot 2*i   = cis(x_pos * inv_freq[i])          # COLUMN (width)  axis  -> even
    slot 2*i+1 = cis(y_pos * inv_freq[i])          # ROW    (height) axis  -> odd

(Token order is row-major: token = y*grid_w + x.) So channel pair
``(q[2s], q[2s+1])`` is rotated by ``angle[t, s]`` where the *even* slots carry
the column angle and the *odd* slots carry the row angle. This is precisely the
layout the C++ ``compute_rope_freqs`` / ``apply_rope_real`` in
``cpp_trt/src/vision_rope/`` produce (cos/sin tables of width
``half_dim == head_dim/2 == 36`` for head_dim==72).

For each complex slot ``s`` with ``(a, b) = (q[2s], q[2s+1])`` and angle
``theta[t,s]`` (``c=cos, d=sin``):

    a' = a*c - b*d
    b' = a*d + b*c

We therefore bake cos/sin of width ``half_dim`` (NOT repeat-interleaved to
head_dim) and apply them with ``apply_rope_real`` below, which expands per pair
to the two equations above -- identical to the C++ reference. ``ROPE_THETA`` is
the MoonViT default 10000. The freq schedule, axis order (x=col even, y=row odd),
and token order all match the re-read sources.

------------------------------------------------------------------------------
Attribute names (matched to modeling_vit.py)
------------------------------------------------------------------------------
The binders target the re-read ``MoonVitPretrainedModel`` layout:
    vit.patch_embed.proj             Conv2d(3,1152,k14,s14)
    vit.patch_embed.pos_emb.weight   [64,64,1152] learnable abs pos-emb
    vit.encoder.blocks[i]            MoonVitEncoderLayer
        .norm0 .wqkv(bias) .wo(bias) .norm1 .mlp (MLP2: fc0->GELU-tanh->fc1)
    vit.encoder.final_layernorm      LayerNorm(1152)
They still fall back to candidate names / type matches for robustness across
checkpoint revisions, and ``export_vision_onnx.py`` prints what it bound so a
human can confirm on first run. The ``mlp1`` connector is applied via its own
``forward`` (``model.mlp1``) so its exact ops come from the checkpoint.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MoonViT RoPE base (Rope2DPosEmb.theta_base default).
# ---------------------------------------------------------------------------
ROPE_THETA = 10000.0


# ---------------------------------------------------------------------------
# Baked 2D-RoPE: build cos/sin constants for an (h, w) patch grid.
#
# EXACT port of Rope2DPosEmb._precompute_freqs_cis + get_freqs_cis from
# eaglevl/model/moon_vit/modeling_vit.py, and bit-identical in layout to the
# C++ compute_rope_freqs in cpp_trt/src/vision_rope/src/vision_rope.cpp:
#
#   inv_freq[i] = theta^(-4*i / head_dim)          i in [0, head_dim//4)
#   x_pos = token %  grid_w  (COLUMN / width)      -> even slots 2*i
#   y_pos = token // grid_w  (ROW    / height)     -> odd  slots 2*i+1
#   token order: row-major, token = y*grid_w + x
#
# Output cos/sin have width half_dim == head_dim//2 (== 36 for head_dim 72), one
# value per complex slot (NOT repeat-interleaved to head_dim).
# ---------------------------------------------------------------------------
def build_2d_rope_cos_sin(
    grid_h: int,
    grid_w: int,
    head_dim: int,
    theta: float = ROPE_THETA,
    dtype: torch.dtype = torch.float32,
):
    """Return (cos, sin) of shape [L, head_dim//2] for L = grid_h*grid_w tokens."""
    assert head_dim % 4 == 0, "2D-RoPE needs head_dim divisible by 4"
    n_freq = head_dim // 4              # per-axis frequency count == half_dim/2
    half_dim = head_dim // 2

    # dim_range = arange(0, head_dim, 4)[:head_dim//4] = [0,4,8,...]
    dim_range = torch.arange(0, head_dim, 4, dtype=torch.float64)[:n_freq]
    inv_freq = 1.0 / (theta ** (dim_range / head_dim))     # [n_freq]

    L = grid_h * grid_w
    flat = torch.arange(L, dtype=torch.float64)
    x_pos = flat % grid_w              # column / width
    y_pos = torch.div(flat, grid_w, rounding_mode="floor")  # row / height

    x_ang = torch.outer(x_pos, inv_freq)   # [L, n_freq]
    y_ang = torch.outer(y_pos, inv_freq)   # [L, n_freq]

    # interleave axes per freq index: even slot = x (column), odd slot = y (row)
    ang = torch.empty(L, half_dim, dtype=torch.float64)
    ang[:, 0::2] = x_ang
    ang[:, 1::2] = y_ang

    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


def apply_rope_real(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Real-arithmetic 2D-RoPE over ADJACENT channel pairs (view_as_complex form).

    x:   [L, n_heads, head_dim]
    cos: [L, head_dim//2]   (broadcast over heads; one angle per complex slot)
    sin: [L, head_dim//2]

    For complex slot s: a=x[...,2s], b=x[...,2s+1], c=cos[...,s], d=sin[...,s]
        out[...,2s]   = a*c - b*d
        out[...,2s+1] = a*d + b*c
    (Identical to la::vision_rope::apply_rope_real.)
    """
    a = x[..., 0::2]                       # [L, H, half_dim]
    b = x[..., 1::2]
    c = cos[:, None, :]                    # [L, 1, half_dim]
    d = sin[:, None, :]
    out_even = a * c - b * d
    out_odd = a * d + b * c
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# Structural binders (attribute names not re-readable in this env).
# ---------------------------------------------------------------------------
def _first_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n), n
    return None, None


def _find_module(root: nn.Module, names, types=None):
    """Find a child module by candidate attribute names, else by type match."""
    mod, name = _first_attr(root, names)
    if mod is not None:
        return mod, name
    if types is not None:
        for n, m in root.named_modules():
            if isinstance(m, types):
                return m, n
    return None, None


class FeatureExtractorWrapper(nn.Module):
    """Wrap MoonViT encoder + mlp1 connector for fixed-geometry ONNX export.

    Parameters
    ----------
    vit : nn.Module
        The MoonViT vision tower (``model.vision_model`` / ``model.vit`` of the
        loaded LocateAnything model).
    mlp1 : nn.Module
        The LocateAnything connector (LayerNorm(4608) -> Linear -> GELU -> Linear),
        ``model.mlp1``.
    grid_h, grid_w : int
        Patch grid for the pinned bucket. L == grid_h * grid_w.
        (grid = image_px / 14.) Must be even in each dim for the 2x2 merger.
    n_heads : int
        Attention heads of MoonViT (so head_dim = hidden // n_heads).
    hidden : int
        MoonViT hidden size (1152).
    """

    def __init__(
        self,
        vit: nn.Module,
        mlp1: nn.Module,
        grid_h: int,
        grid_w: int,
        n_heads: int,
        hidden: int = 1152,
    ):
        super().__init__()
        assert grid_h % 2 == 0 and grid_w % 2 == 0, "2x2 patch-merger needs even grid"
        self.vit = vit
        self.mlp1 = mlp1
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.L = self.grid_h * self.grid_w
        self.n_heads = int(n_heads)
        self.hidden = int(hidden)
        self.head_dim = self.hidden // self.n_heads

        # Force exportable attention everywhere we can. MoonViT stores the impl
        # as a plain str attribute `attn_implementation` on each encoder layer.
        for m in self.vit.modules():
            if hasattr(m, "_attn_implementation"):
                m._attn_implementation = "eager"
            if hasattr(m, "attn_implementation"):
                m.attn_implementation = "eager"
        if hasattr(getattr(self.vit, "config", None), "_attn_implementation"):
            self.vit.config._attn_implementation = "eager"

        # ---- bind submodules we drive manually --------------------------------
        # MoonVitPretrainedModel layout (re-read from modeling_vit.py):
        #   vit.patch_embed.proj            : Conv2d(3,1152,k14,s14)
        #   vit.patch_embed.pos_emb.weight  : [64,64,1152] learnable abs pos-emb
        #   vit.encoder.blocks[i]           : MoonVitEncoderLayer
        #       .norm0 .wqkv(bias) .wo(bias) .norm1 .mlp(MLP2: fc0/act/fc1)
        #   vit.encoder.final_layernorm     : LayerNorm(1152)
        pe, _ = _first_attr(self.vit, ["patch_embed", "patch_embedding"])
        self.patch_embed, _ = _find_module(pe or self.vit, ["proj", "conv1"], (nn.Conv2d,))
        # learnable absolute pos-emb source (a 64x64xdim grid parameter)
        pos_emb_mod, _ = _first_attr(pe or self.vit, ["pos_emb", "position_embedding"])
        self.abs_pos_param = None
        if pos_emb_mod is not None:
            self.abs_pos_param, _ = _first_attr(pos_emb_mod, ["weight"])
        if self.abs_pos_param is None:
            self.abs_pos_param, _ = _first_attr(
                self.vit, ["pos_embed", "abs_pos_emb"]
            )

        # encoder + blocks + final norm
        encoder, _ = _first_attr(self.vit, ["encoder"])
        enc = encoder if encoder is not None else self.vit
        blocks, _ = _find_module(enc, ["blocks", "layers", "encoder_layers"])
        if blocks is None:
            for _n, m in self.vit.named_modules():
                if isinstance(m, nn.ModuleList) and len(m) >= 20:
                    blocks = m
                    break
        self.blocks = blocks
        self.final_norm, _ = _find_module(
            enc, ["final_layernorm", "norm", "ln_post", "post_layernorm"],
            types=(nn.LayerNorm,),
        )

        # ---- bake pos-emb + rope ---------------------------------------------
        self._bake_pos_emb()
        cos, sin = build_2d_rope_cos_sin(self.grid_h, self.grid_w, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # ---------------------------------------------------------------------
    def _bake_pos_emb(self):
        """Resolve the bicubic-interpolated absolute pos-emb for this bucket and
        store it as a constant buffer [L, hidden].

        EXACT port of Learnable2DInterpPosEmb.forward (modeling_vit.py):
            weight: [src_h, src_w, dim]  (default 64x64x1152)
            if (grid_h, grid_w) == (src_h, src_w):  weight.flatten(end_dim=1)
            else: F.interpolate(weight.permute(2,0,1)[None], size=(gh,gw),
                                 mode="bicubic")   # NOTE: no align_corners ->
                                 # torch default align_corners=False
                  .squeeze(0).permute(1,2,0).flatten(end_dim=1)
        Tokens come out row-major (gh outer, gw inner), matching patch order.
        """
        if self.abs_pos_param is None:
            # No learnable abs pos-emb -> add zeros (RoPE-only positional info).
            self.register_buffer(
                "pos_emb_baked", torch.zeros(self.L, self.hidden), persistent=False
            )
            return
        w = self.abs_pos_param.detach().float()      # expect [src_h, src_w, dim]
        if w.dim() != 3:
            raise ValueError(f"Unexpected abs pos-emb weight shape {tuple(w.shape)}; "
                             "expected [src_h, src_w, dim]")
        src_h, src_w, dim = w.shape
        assert dim == self.hidden, (dim, self.hidden)
        if (self.grid_h, self.grid_w) == (src_h, src_w):
            baked = w.reshape(self.L, dim).contiguous()
        else:
            t = w.permute(2, 0, 1).unsqueeze(0)      # [1, dim, src_h, src_w]
            t = F.interpolate(t, size=(self.grid_h, self.grid_w), mode="bicubic")
            baked = t.squeeze(0).permute(1, 2, 0).reshape(self.L, dim).contiguous()
        self.register_buffer("pos_emb_baked", baked, persistent=False)

    # ---------------------------------------------------------------------
    def _attention(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """One MoonViT block forward with baked RoPE and no mask (N==1).

        x: [L, hidden]. Mirrors MoonVitEncoderLayer.forward exactly:
            residual = x; x = norm0(x); attn = wo(attention(wqkv(x))); x += attn
            residual = x; x = mlp(norm1(x)); x += residual
        We drive qkv/wo/norms/mlp explicitly to inject the real-arithmetic RoPE.
        Block attrs (re-read from modeling_vit.py): norm0, wqkv(bias), wo(bias),
        norm1, mlp (MLP2: fc0 -> GELU-tanh -> fc1).
        """
        L = x.shape[0]
        H, Dh = self.n_heads, self.head_dim

        norm0, _ = _find_module(block, ["norm0", "norm1", "ln_1", "input_layernorm"], (nn.LayerNorm,))
        wqkv, _ = _find_module(block, ["wqkv", "qkv", "in_proj"], (nn.Linear,))
        wo, _ = _find_module(block, ["wo", "proj", "out_proj"], (nn.Linear,))
        norm1, _ = _find_module(block, ["norm1", "norm2", "ln_2", "post_attention_layernorm"], (nn.LayerNorm,))

        # --- attention block ---
        residual = x
        h = norm0(x)
        xqkv = wqkv(h)                                 # [L, 3*hidden]
        # MoonViT: view(L, 3, num_heads, head_dim) then unbind(dim=-3)
        xqkv = xqkv.view(L, 3, H, Dh)
        q, k, v = torch.unbind(xqkv, dim=-3)          # each [L, H, Dh]

        q = apply_rope_real(q, self.rope_cos, self.rope_sin)
        k = apply_rope_real(k, self.rope_cos, self.rope_sin)

        # full self-attention (cu_seqlens == [0, L] -> no mask, causal=False)
        q = q.transpose(0, 1)                          # [H, L, Dh]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)  # [H, L, L]
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(v.dtype)
        out = torch.matmul(attn, v)                    # [H, L, Dh]
        out = out.transpose(0, 1).reshape(L, H * Dh)   # [L, hidden]
        out = wo(out)
        x = residual + out

        # --- MLP block (MLP2: fc0 -> GELU-tanh -> fc1; 1152 -> 4304 -> 1152) ---
        residual = x
        mlp, _ = _find_module(block, ["mlp", "feed_forward", "ffn"])
        h2 = norm1(x)
        h2 = self._mlp_forward(mlp, h2)
        x = residual + h2
        return x

    def _mlp_forward(self, mlp: nn.Module, h: torch.Tensor) -> torch.Tensor:
        if mlp is None:
            return h
        # MoonViT MLP2: fc0 (in->hidden), activation (PytorchGELUTanh), fc1 (hidden->out)
        fc0, _ = _find_module(mlp, ["fc0", "fc1", "w1", "up_proj", "dense_h_to_4h"], (nn.Linear,))
        fc1, _ = _find_module(mlp, ["fc1", "fc2", "w2", "down_proj", "dense_4h_to_h"], (nn.Linear,))
        # ensure distinct modules when both candidate-lists alias 'fc1'
        if fc0 is fc1:
            lins = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
            if len(lins) >= 2:
                fc0, fc1 = lins[0], lins[1]
        if fc0 is None or fc1 is None:
            return mlp(h)
        h = fc0(h)
        h = F.gelu(h, approximate="tanh")
        h = fc1(h)
        return h

    # ---------------------------------------------------------------------
    def _patch_merge_2x2(self, x: torch.Tensor) -> torch.Tensor:
        """Static 2x2 patch-merger: [L, hidden] -> [L/4, 4*hidden] (==4608).

        Token order is row-major over (grid_h, grid_w). We fold each 2x2 spatial
        block into one merged token by concatenating the 4 members in the fixed
        order (top-left, top-right, bottom-left, bottom-right) to match the eager
        merger. Verify the concat order against the checkpoint's patch_merger
        (CANNOT verify here)."""
        gh, gw, hid = self.grid_h, self.grid_w, self.hidden
        x = x.reshape(gh, gw, hid)
        x = x.reshape(gh // 2, 2, gw // 2, 2, hid)
        x = x.permute(0, 2, 1, 3, 4).contiguous()      # [gh/2, gw/2, 2, 2, hid]
        x = x.reshape((gh // 2) * (gw // 2), 4 * hid)  # [L/4, 4608]
        return x

    # ---------------------------------------------------------------------
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: [L, 3, 14, 14] -> vit_embeds: [L/4, H_llm]."""
        L = self.L

        # patch embed: Conv2d k14 s14 over each 14x14 patch -> [L, hidden, 1, 1]
        x = self.patch_embed(pixel_values)
        x = x.reshape(L, self.hidden)

        # baked absolute pos-emb
        x = x + self.pos_emb_baked.to(x.dtype)

        # transformer blocks
        for block in self.blocks:
            x = self._attention(block, x)

        # final LayerNorm
        if self.final_norm is not None:
            x = self.final_norm(x)

        # 2x2 patch merge -> [L/4, 4608]
        x = self._patch_merge_2x2(x)

        # mlp1 connector (no pixel-shuffle) -> LLM space [L/4, H_llm]
        vit_embeds = self.mlp1(x)
        return vit_embeds
