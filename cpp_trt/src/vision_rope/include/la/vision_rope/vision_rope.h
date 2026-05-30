// SPDX-License-Identifier: Apache-2.0
//
// Real-arithmetic 2D-RoPE for MoonViT.
//
// This header declares a pure-C++ (no Torch / no TensorRT / no CUDA required)
// reference implementation of the 2D rotary position embedding used by the
// MoonViT vision encoder. It replaces the complex-number path
// (torch.view_as_complex / torch.polar / complex multiply / torch.view_as_real)
// used in the Python reference with a mathematically identical real-valued
// interleaved cos/sin formulation that is exportable by TensorRT.
//
// ---------------------------------------------------------------------------
// Reference (Python) being ported  [Spec Section 4.2]
//   eaglevl/model/moon_vit/modeling_vit.py : Rope2DPosEmb._precompute_freqs_cis
//                                            and apply_rope
// ---------------------------------------------------------------------------
// The MoonViT reference builds a per-token complex rotation table
// `freqs_cis` of shape (L, dim/2) where L = grid_h * grid_w is the number of
// patch tokens and `dim` is the per-head channel count (= hidden_size /
// num_heads). With the spec's stated freqs_cis last dim == 36, dim == 72.
//
// Exact reference (Rope2DPosEmb._precompute_freqs_cis):
//   N        = max_h * max_w
//   flat_pos = arange(0, N)
//   x_pos    = flat_pos %  max_w        # COLUMN (width)  index of each token
//   y_pos    = flat_pos // max_w        # ROW    (height) index of each token
//   dim_range= arange(0, dim, 4)[: dim//4]      # [0, 4, 8, ...], length dim/4
//   freqs    = 1 / (theta_base ** (dim_range / dim))   # length dim/4
//   x_freqs  = outer(x_pos, freqs)      # (N, dim/4)
//   y_freqs  = outer(y_pos, freqs)      # (N, dim/4)
//   x_cis    = polar(1, x_freqs)        # cos(x_freqs) + i sin(x_freqs)
//   y_cis    = polar(1, y_freqs)        # cos(y_freqs) + i sin(y_freqs)
//   freqs_cis= cat([x_cis[...,None], y_cis[...,None]], -1).reshape(..., dim/2)
//
// The cat-then-reshape INTERLEAVES the two axes per frequency index i, so the
// dim/2 complex slots of a token are (this is the make-or-break detail):
//   slot 2*i   = x_cis[i]  -> COLUMN / width  axis, angle = x * freqs[i]
//   slot 2*i+1 = y_cis[i]  -> ROW    / height axis, angle = y * freqs[i]
// matching the reference docstring:
//   ret[..., 2*i]   = cis(w * theta^(-4i/dim))
//   ret[..., 2*i+1] = cis(h * theta^(-4i/dim))   for i in [0, dim//4).
// NOTE: freqs[i] = theta^(-4i/dim) because dim_range = 4*i. Both axes reuse the
// SAME freqs vector; only the position multiplier (x vs y) differs.
//
// Token ordering: get_freqs_cis slices the precomputed table freqs_cis[:h, :w]
// and reshape(-1, dim/2), i.e. ROW-MAJOR over the grid -> token = y*grid_w + x,
// with that token's angles taken at grid position (y, x).
//
// Application (apply_rope) — the part that ports wrong if pairing is misread:
//   xq is reshaped (..., dim) -> (..., dim/2, 2) and viewed as complex so that
//   complex slot s is formed from the INTERLEAVED real pair
//       (x[2*s], x[2*s+1])  ==  x[2*s] + i * x[2*s+1].
//   Then  xq_out_complex[s] = xq_complex[s] * freqs_cis[s].
//   view_as_real flattens back so that
//       out[2*s]   = real(xq_complex[s] * freqs_cis[s])
//       out[2*s+1] = imag(xq_complex[s] * freqs_cis[s]).
//
// Writing the complex multiply in real arithmetic with
//   a = x[2*s], b = x[2*s+1], c = cos(angle[s]), d = sin(angle[s]):
//   (a + i b)(c + i d) = (a c - b d) + i (a d + b c)
//   => out[2*s]   = a*c - b*d
//      out[2*s+1] = a*d + b*c
//
// IMPORTANT ASYMMETRY OF THE TWO INTERLEAVINGS:
//   * apply_rope pairs ADJACENT CHANNELS (2*s, 2*s+1) of the head vector to form
//     one complex number (GPT-NeoX-style adjacent pairs, NOT Llama rotate_half).
//   * the table interleaves the two SPATIAL AXES (x at even slot, y at odd slot).
//   So channel pair (2*s, 2*s+1) is rotated by angle[s], where angle[s] comes
//   from the x-axis when s is even and the y-axis when s is odd. The cos/sin
//   tables produced by compute_rope_freqs() already encode this, so
//   apply_rope_real just consumes cos[t,s]/sin[t,s] uniformly per slot s.
// ---------------------------------------------------------------------------

#ifndef LA_VISION_ROPE_VISION_ROPE_H
#define LA_VISION_ROPE_VISION_ROPE_H

#include <cstddef>
#include <vector>

namespace la {
namespace vision_rope {

// Default RoPE theta base used by MoonViT.
inline constexpr double kDefaultRopeTheta = 10000.0;

// Result of building the rotation tables for a given grid.
//
// Layout: both `cos` and `sin` are row-major (L, half_dim) where
//   L        = grid_h * grid_w        (number of tokens, row-major over (y, x))
//   half_dim = head_dim / 2           (number of complex slots == 36 for MoonViT)
// Element (token t, slot s) is at index  t * half_dim + s.
//
// For token t at grid coord (y = t / grid_w, x = t % grid_w), with frequency
// index i = s / 2 and inv_freq[i] = theta^(-4i/head_dim):
//   - EVEN slot s = 2*i   carries angle = x * inv_freq[i]   (column / width)
//   - ODD  slot s = 2*i+1 carries angle = y * inv_freq[i]   (row / height)
// and cos/sin store cos(angle), sin(angle) respectively. This interleaving of
// the x and y axes mirrors the reference cat([x_cis,y_cis],-1).reshape exactly.
struct RopeTables {
  std::size_t grid_h = 0;
  std::size_t grid_w = 0;
  std::size_t head_dim = 0;   // full per-head channel count (e.g. 72)
  std::size_t half_dim = 0;   // head_dim / 2 (== number of complex slots, e.g. 36)
  std::size_t seq_len = 0;    // grid_h * grid_w
  std::vector<float> cos;     // (seq_len, half_dim) row-major
  std::vector<float> sin;     // (seq_len, half_dim) row-major
};

// Build the cos/sin rotation tables for a (grid_h x grid_w) patch grid.
//
// `head_dim` is the FULL per-head channel dimension (must be divisible by 4 so
// that it splits evenly into two axes of paired channels). For MoonViT use 72,
// which produces half_dim == 36 matching the reference `freqs_cis` shape (L, 36).
//
// `theta` is the RoPE frequency base (default 10000.0).
//
// Throws std::invalid_argument if head_dim is not a positive multiple of 4 or
// if the grid dimensions are zero.
RopeTables compute_rope_freqs(std::size_t grid_h, std::size_t grid_w,
                              std::size_t head_dim,
                              double theta = kDefaultRopeTheta);

// Apply 2D-RoPE to a query OR key tensor in place using the real interleaved
// formulation described in the file header.
//
// Tensor layout (row-major):  q has shape (seq_len, n_heads, head_dim).
// Element (t, h, c) is at index  ((t * n_heads) + h) * head_dim + c.
//
// `cos` and `sin` must come from compute_rope_freqs() for the same grid and
// head_dim; they are shaped (seq_len, half_dim) with half_dim == head_dim/2.
// The same rotation table is broadcast across all heads.
//
// For each (t, h) and each complex slot s in [0, half_dim):
//   a = q[..., 2*s];  b = q[..., 2*s+1]
//   c = cos[t, s];    d = sin[t, s]
//   q[..., 2*s]   = a*c - b*d
//   q[..., 2*s+1] = a*d + b*c
//
// Throws std::invalid_argument on size mismatches.
void apply_rope_real(std::vector<float>& q, std::size_t seq_len,
                     std::size_t n_heads, std::size_t head_dim,
                     const std::vector<float>& cos,
                     const std::vector<float>& sin);

// Raw-pointer overload (no allocation, no STL container coupling) so callers in
// kernels / TRT plugins can use it directly. `q` points to seq_len*n_heads*
// head_dim floats; `cos`/`sin` point to seq_len*(head_dim/2) floats.
void apply_rope_real(float* q, std::size_t seq_len, std::size_t n_heads,
                     std::size_t head_dim, const float* cos, const float* sin);

// Convenience: apply the SAME table to both q and k (the common attention case).
void apply_rope_real_qk(std::vector<float>& q, std::vector<float>& k,
                        std::size_t seq_len, std::size_t n_heads,
                        std::size_t head_dim, const std::vector<float>& cos,
                        const std::vector<float>& sin);

}  // namespace vision_rope
}  // namespace la

#endif  // LA_VISION_ROPE_VISION_ROPE_H
