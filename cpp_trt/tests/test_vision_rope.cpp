// Unit tests for la_vision_rope (2D-RoPE real-arithmetic implementation).
//
// The expected values are computed INDEPENDENTLY with a std::complex-based
// reference (the polar/complex-multiply path the real-arithmetic code replaces),
// so the channel pairing and x/y axis interleaving are verified end-to-end.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <vector>

#include "la/vision_rope/vision_rope.h"

namespace {

using la::vision_rope::RopeTables;
using la::vision_rope::compute_rope_freqs;
using la::vision_rope::apply_rope_real;
using la::vision_rope::kDefaultRopeTheta;

// Independent reference table builder.
// freqs[i] = theta^(-4i/head_dim), i in [0, head_dim/4).
// per token (y=tok/grid_w, x=tok%grid_w):
//   slot 2*i   -> x_cis[i] = polar(1, x*freqs[i])  (column/width)
//   slot 2*i+1 -> y_cis[i] = polar(1, y*freqs[i])  (row/height)
struct RefTable {
  size_t seq_len, half_dim;
  std::vector<double> cos, sin;  // (seq_len, half_dim)
};

RefTable BuildRefTable(size_t grid_h, size_t grid_w, size_t head_dim,
                       double theta) {
  size_t half_dim = head_dim / 2;
  size_t quarter = head_dim / 4;
  size_t seq_len = grid_h * grid_w;
  RefTable t;
  t.seq_len = seq_len;
  t.half_dim = half_dim;
  t.cos.assign(seq_len * half_dim, 0.0);
  t.sin.assign(seq_len * half_dim, 0.0);
  std::vector<double> freqs(quarter);
  for (size_t i = 0; i < quarter; ++i) {
    freqs[i] = std::pow(theta, -4.0 * static_cast<double>(i) /
                                   static_cast<double>(head_dim));
  }
  for (size_t tok = 0; tok < seq_len; ++tok) {
    size_t x = tok % grid_w;
    size_t y = tok / grid_w;
    for (size_t i = 0; i < quarter; ++i) {
      double ax = static_cast<double>(x) * freqs[i];  // even slot
      double ay = static_cast<double>(y) * freqs[i];  // odd slot
      size_t s0 = 2 * i;
      size_t s1 = 2 * i + 1;
      t.cos[tok * half_dim + s0] = std::cos(ax);
      t.sin[tok * half_dim + s0] = std::sin(ax);
      t.cos[tok * half_dim + s1] = std::cos(ay);
      t.sin[tok * half_dim + s1] = std::sin(ay);
    }
  }
  return t;
}

}  // namespace

TEST(RopeFreqs, ShapeMatchesGrid) {
  size_t grid_h = 3, grid_w = 5, head_dim = 72;
  RopeTables tab = compute_rope_freqs(grid_h, grid_w, head_dim, kDefaultRopeTheta);
  EXPECT_EQ(tab.grid_h, grid_h);
  EXPECT_EQ(tab.grid_w, grid_w);
  EXPECT_EQ(tab.head_dim, head_dim);
  EXPECT_EQ(tab.half_dim, head_dim / 2);
  EXPECT_EQ(tab.seq_len, grid_h * grid_w);
  EXPECT_EQ(tab.cos.size(), tab.seq_len * tab.half_dim);
  EXPECT_EQ(tab.sin.size(), tab.seq_len * tab.half_dim);
}

TEST(RopeFreqs, ValuesMatchIndependentComplexReference) {
  struct Case { size_t h, w, dim; };
  for (Case c : {Case{1, 1, 8}, Case{2, 3, 8}, Case{4, 5, 72}, Case{3, 2, 16}}) {
    RopeTables tab = compute_rope_freqs(c.h, c.w, c.dim, kDefaultRopeTheta);
    RefTable ref = BuildRefTable(c.h, c.w, c.dim, kDefaultRopeTheta);
    ASSERT_EQ(tab.cos.size(), ref.cos.size());
    for (size_t k = 0; k < ref.cos.size(); ++k) {
      EXPECT_NEAR(tab.cos[k], static_cast<float>(ref.cos[k]), 1e-5)
          << "cos mismatch h=" << c.h << " w=" << c.w << " dim=" << c.dim
          << " k=" << k;
      EXPECT_NEAR(tab.sin[k], static_cast<float>(ref.sin[k]), 1e-5)
          << "sin mismatch k=" << k;
    }
  }
}

// First token (x=0,y=0): all angles zero -> cos=1, sin=0 everywhere.
TEST(RopeFreqs, FirstTokenIsIdentity) {
  RopeTables tab = compute_rope_freqs(2, 2, 72, kDefaultRopeTheta);
  for (size_t s = 0; s < tab.half_dim; ++s) {
    EXPECT_NEAR(tab.cos[s], 1.0f, 1e-6) << "s=" << s;
    EXPECT_NEAR(tab.sin[s], 0.0f, 1e-6) << "s=" << s;
  }
}

// apply_rope_real must equal the independent complex rotation:
//   complex slot s = vec[2s] + i*vec[2s+1] ; out = slot * (cos[s] + i*sin[s]).
TEST(ApplyRopeReal, EqualsComplexRotation) {
  size_t grid_h = 4, grid_w = 5, head_dim = 72, n_heads = 3;
  size_t seq_len = grid_h * grid_w;
  RopeTables tab = compute_rope_freqs(grid_h, grid_w, head_dim, kDefaultRopeTheta);

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<float> q(seq_len * n_heads * head_dim);
  for (auto& v : q) v = dist(rng);
  std::vector<float> q_copy = q;  // independent reference operates on the copy

  apply_rope_real(q, seq_len, n_heads, head_dim, tab.cos, tab.sin);

  // Independent reference using std::complex<double>.
  size_t half_dim = head_dim / 2;
  for (size_t t = 0; t < seq_len; ++t) {
    for (size_t hd = 0; hd < n_heads; ++hd) {
      size_t base = (t * n_heads + hd) * head_dim;
      for (size_t s = 0; s < half_dim; ++s) {
        double a = q_copy[base + 2 * s];
        double b = q_copy[base + 2 * s + 1];
        double c = tab.cos[t * half_dim + s];
        double d = tab.sin[t * half_dim + s];
        std::complex<double> z(a, b);
        std::complex<double> rot(c, d);
        std::complex<double> out = z * rot;
        EXPECT_NEAR(q[base + 2 * s], static_cast<float>(out.real()), 1e-4)
            << "t=" << t << " head=" << hd << " s=" << s << " real";
        EXPECT_NEAR(q[base + 2 * s + 1], static_cast<float>(out.imag()), 1e-4)
            << "t=" << t << " head=" << hd << " s=" << s << " imag";
      }
    }
  }
}

// Table is broadcast across heads: applying RoPE to identical per-head vectors
// must yield identical per-head results.
TEST(ApplyRopeReal, BroadcastsAcrossHeads) {
  size_t grid_h = 2, grid_w = 3, head_dim = 16, n_heads = 4;
  size_t seq_len = grid_h * grid_w;
  RopeTables tab = compute_rope_freqs(grid_h, grid_w, head_dim, kDefaultRopeTheta);

  std::mt19937 rng(777);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Per-token vector replicated across all heads.
  std::vector<float> q(seq_len * n_heads * head_dim);
  for (size_t t = 0; t < seq_len; ++t) {
    std::vector<float> token_vec(head_dim);
    for (auto& v : token_vec) v = dist(rng);
    for (size_t hd = 0; hd < n_heads; ++hd) {
      size_t base = (t * n_heads + hd) * head_dim;
      for (size_t k = 0; k < head_dim; ++k) q[base + k] = token_vec[k];
    }
  }

  apply_rope_real(q, seq_len, n_heads, head_dim, tab.cos, tab.sin);

  for (size_t t = 0; t < seq_len; ++t) {
    size_t base0 = (t * n_heads + 0) * head_dim;
    for (size_t hd = 1; hd < n_heads; ++hd) {
      size_t base = (t * n_heads + hd) * head_dim;
      for (size_t k = 0; k < head_dim; ++k) {
        EXPECT_FLOAT_EQ(q[base + k], q[base0 + k])
            << "t=" << t << " head=" << hd << " k=" << k;
      }
    }
  }
}
