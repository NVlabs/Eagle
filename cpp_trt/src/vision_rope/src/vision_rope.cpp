// SPDX-License-Identifier: Apache-2.0
//
// Pure-C++ reference implementation of MoonViT real-arithmetic 2D-RoPE.
// See include/la/vision_rope/vision_rope.h for the mathematical derivation
// and the exact element pairing that mirrors the Python complex path.

#include "la/vision_rope/vision_rope.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace la {
namespace vision_rope {

namespace {

// Build the inverse-frequency grid shared by both spatial axes.
//
// Mirrors the reference exactly:
//   dim_range = arange(0, head_dim, 4)[: head_dim/4]   # [0, 4, 8, ...]
//   freqs     = 1 / (theta ** (dim_range / head_dim))
// i.e. inv_freq[i] = theta^(-4*i / head_dim) for i in [0, head_dim/4).
//
// There are head_dim/4 frequencies. The SAME vector is used for the column (x)
// and the row (y) axes; the two axes are interleaved per frequency index when
// the cos/sin tables are filled (x at even slot 2*i, y at odd slot 2*i+1),
// giving head_dim/2 == half_dim total angles per token.
std::vector<double> build_inv_freq(std::size_t head_dim, double theta) {
  const std::size_t n_freq = head_dim / 4;  // per-axis frequency count
  std::vector<double> inv_freq(n_freq);
  const double d = static_cast<double>(head_dim);
  for (std::size_t j = 0; j < n_freq; ++j) {
    // exponent uses index 0, 4, 8, ... (== 4*j) over the full head_dim.
    const double exponent = static_cast<double>(4 * j) / d;
    inv_freq[j] = 1.0 / std::pow(theta, exponent);
  }
  return inv_freq;
}

}  // namespace

RopeTables compute_rope_freqs(std::size_t grid_h, std::size_t grid_w,
                              std::size_t head_dim, double theta) {
  if (grid_h == 0 || grid_w == 0) {
    throw std::invalid_argument(
        "compute_rope_freqs: grid_h and grid_w must be non-zero");
  }
  if (head_dim == 0 || (head_dim % 4) != 0) {
    throw std::invalid_argument(
        "compute_rope_freqs: head_dim must be a positive multiple of 4 (got " +
        std::to_string(head_dim) + ")");
  }

  RopeTables t;
  t.grid_h = grid_h;
  t.grid_w = grid_w;
  t.head_dim = head_dim;
  t.half_dim = head_dim / 2;       // number of complex slots / angles per token
  t.seq_len = grid_h * grid_w;

  const std::vector<double> inv_freq = build_inv_freq(head_dim, theta);
  const std::size_t n_freq = inv_freq.size();      // == head_dim/4 == half_dim/2

  t.cos.assign(t.seq_len * t.half_dim, 0.0f);
  t.sin.assign(t.seq_len * t.half_dim, 0.0f);

  // Tokens are ordered row-major over the grid: token index = y * grid_w + x
  // (the reference slices freqs_cis[:h, :w] then reshape(-1, dim//2)).
  //
  // Slot layout mirrors the reference EXACTLY:
  //   cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1).reshape(..., -1)
  // interleaves the two axes per frequency index i:
  //   slot 2*i   = x_cis[i]  (COLUMN / width  axis,  angle = x * inv_freq[i])
  //   slot 2*i+1 = y_cis[i]  (ROW    / height axis,  angle = y * inv_freq[i])
  // This matches the docstring: ret[...,2i]=cis(w*theta^-4i/dim),
  //                             ret[...,2i+1]=cis(h*theta^-4i/dim).
  for (std::size_t y = 0; y < grid_h; ++y) {
    for (std::size_t x = 0; x < grid_w; ++x) {
      const std::size_t tok = y * grid_w + x;
      const std::size_t base = tok * t.half_dim;
      for (std::size_t i = 0; i < n_freq; ++i) {
        // Even slot 2*i: column (width / x) angle.
        const double ax = static_cast<double>(x) * inv_freq[i];
        t.cos[base + 2 * i] = static_cast<float>(std::cos(ax));
        t.sin[base + 2 * i] = static_cast<float>(std::sin(ax));

        // Odd slot 2*i+1: row (height / y) angle.
        const double ay = static_cast<double>(y) * inv_freq[i];
        t.cos[base + 2 * i + 1] = static_cast<float>(std::cos(ay));
        t.sin[base + 2 * i + 1] = static_cast<float>(std::sin(ay));
      }
    }
  }

  return t;
}

void apply_rope_real(float* q, std::size_t seq_len, std::size_t n_heads,
                     std::size_t head_dim, const float* cos, const float* sin) {
  if (q == nullptr || cos == nullptr || sin == nullptr) {
    throw std::invalid_argument("apply_rope_real: null pointer argument");
  }
  if (head_dim == 0 || (head_dim % 2) != 0) {
    throw std::invalid_argument(
        "apply_rope_real: head_dim must be a positive even number");
  }
  const std::size_t half_dim = head_dim / 2;  // complex slots per token

  for (std::size_t t = 0; t < seq_len; ++t) {
    const float* cos_row = cos + t * half_dim;
    const float* sin_row = sin + t * half_dim;
    for (std::size_t h = 0; h < n_heads; ++h) {
      float* vec = q + ((t * n_heads) + h) * head_dim;
      for (std::size_t s = 0; s < half_dim; ++s) {
        // Interleaved pair: complex slot s is (vec[2s] + i*vec[2s+1]).
        const float a = vec[2 * s];
        const float b = vec[2 * s + 1];
        const float c = cos_row[s];
        const float d = sin_row[s];
        // (a + i b)(c + i d) = (a c - b d) + i (a d + b c)
        vec[2 * s] = a * c - b * d;
        vec[2 * s + 1] = a * d + b * c;
      }
    }
  }
}

void apply_rope_real(std::vector<float>& q, std::size_t seq_len,
                     std::size_t n_heads, std::size_t head_dim,
                     const std::vector<float>& cos,
                     const std::vector<float>& sin) {
  const std::size_t half_dim = head_dim / 2;
  const std::size_t expected_q = seq_len * n_heads * head_dim;
  const std::size_t expected_tab = seq_len * half_dim;
  if (q.size() != expected_q) {
    throw std::invalid_argument(
        "apply_rope_real: q size mismatch (expected " +
        std::to_string(expected_q) + ", got " + std::to_string(q.size()) + ")");
  }
  if (cos.size() != expected_tab || sin.size() != expected_tab) {
    throw std::invalid_argument(
        "apply_rope_real: cos/sin table size mismatch (expected " +
        std::to_string(expected_tab) + ")");
  }
  apply_rope_real(q.data(), seq_len, n_heads, head_dim, cos.data(), sin.data());
}

void apply_rope_real_qk(std::vector<float>& q, std::vector<float>& k,
                        std::size_t seq_len, std::size_t n_heads,
                        std::size_t head_dim, const std::vector<float>& cos,
                        const std::vector<float>& sin) {
  apply_rope_real(q, seq_len, n_heads, head_dim, cos, sin);
  apply_rope_real(k, seq_len, n_heads, head_dim, cos, sin);
}

}  // namespace vision_rope
}  // namespace la
