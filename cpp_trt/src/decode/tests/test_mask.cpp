#include "la/decode/mask.h"

#include <cmath>

#include <gtest/gtest.h>

using la::decode::build_ar_mask;
using la::decode::build_mtp_window_mask;
using la::decode::Mask;
using la::decode::MASK_NEG_INF;

namespace {

bool is_blocked(float v) { return std::isinf(v) && v < 0.0f; }
bool is_kept(float v) { return v == 0.0f; }

// ---------------- build_ar_mask ----------------

TEST(ArMask, PrefillSquareLowerTriangular) {
  const int n = 5;
  Mask m = build_ar_mask(n);  // q_len == kv_len
  ASSERT_EQ(m.q, n);
  ASSERT_EQ(m.kv, n);
  for (int qi = 0; qi < n; ++qi) {
    for (int kj = 0; kj < n; ++kj) {
      if (kj <= qi) {
        EXPECT_TRUE(is_kept(m.at(qi, kj))) << "qi=" << qi << " kj=" << kj;
      } else {
        EXPECT_TRUE(is_blocked(m.at(qi, kj))) << "qi=" << qi << " kj=" << kj;
      }
    }
  }
}

TEST(ArMask, SingleArStepSeesAll) {
  const int kv = 7;
  Mask m = build_ar_mask(kv, /*q_len=*/1);
  ASSERT_EQ(m.q, 1);
  ASSERT_EQ(m.kv, kv);
  for (int kj = 0; kj < kv; ++kj) {
    EXPECT_TRUE(is_kept(m.at(0, kj))) << "kj=" << kj;
  }
}

TEST(ArMask, GeneralRightAlignedQueries) {
  // kv_len=6, q_len=3 -> queries are absolute positions 3,4,5.
  const int kv = 6, q = 3;
  Mask m = build_ar_mask(kv, q);
  ASSERT_EQ(m.q, q);
  ASSERT_EQ(m.kv, kv);
  const int offset = kv - q;  // 3
  for (int qi = 0; qi < q; ++qi) {
    const int abs_pos = offset + qi;
    for (int kj = 0; kj < kv; ++kj) {
      if (kj <= abs_pos) {
        EXPECT_TRUE(is_kept(m.at(qi, kj))) << "qi=" << qi << " kj=" << kj;
      } else {
        EXPECT_TRUE(is_blocked(m.at(qi, kj))) << "qi=" << qi << " kj=" << kj;
      }
    }
  }
}

TEST(ArMask, NegInfIsActuallyNegativeInfinity) {
  Mask m = build_ar_mask(2);
  EXPECT_TRUE(std::isinf(m.at(0, 1)));
  EXPECT_LT(m.at(0, 1), 0.0f);
  EXPECT_EQ(MASK_NEG_INF, m.at(0, 1));
}

// ---------------- build_mtp_window_mask ----------------

TEST(MtpMask, ShapeAndRegions) {
  const int kv = 10;
  const int block = 6;
  Mask m = build_mtp_window_mask(kv, block);
  ASSERT_EQ(m.q, kv);
  ASSERT_EQ(m.kv, kv);

  const int prefix_len = kv - block;       // 4
  const int excluded_col = prefix_len - 1; // 3

  // (a) Causal prefix region: rows [0, prefix_len)
  for (int qi = 0; qi < prefix_len; ++qi) {
    for (int kj = 0; kj < kv; ++kj) {
      if (kj < prefix_len && kj <= qi) {
        EXPECT_TRUE(is_kept(m.at(qi, kj))) << "prefix qi=" << qi << " kj=" << kj;
      } else {
        EXPECT_TRUE(is_blocked(m.at(qi, kj)))
            << "prefix qi=" << qi << " kj=" << kj;
      }
    }
  }

  // window rows [prefix_len, kv)
  for (int qi = prefix_len; qi < kv; ++qi) {
    // (c) prefix keys: kept except excluded_col
    for (int kj = 0; kj < prefix_len; ++kj) {
      if (kj == excluded_col) {
        EXPECT_TRUE(is_blocked(m.at(qi, kj)))
            << "excluded col qi=" << qi << " kj=" << kj;
      } else {
        EXPECT_TRUE(is_kept(m.at(qi, kj)))
            << "win->prefix qi=" << qi << " kj=" << kj;
      }
    }
    // (b) window keys: all kept (bidirectional)
    for (int kj = prefix_len; kj < kv; ++kj) {
      EXPECT_TRUE(is_kept(m.at(qi, kj)))
          << "window bidir qi=" << qi << " kj=" << kj;
    }
  }
}

TEST(MtpMask, ExcludedColumnIndexExact) {
  // Load-bearing: excluded_col == kv_len - block_size - 1.
  const int kv = 9, block = 6;
  Mask m = build_mtp_window_mask(kv, block);
  const int excluded_col = kv - block - 1;  // 2
  // First window row.
  const int qi = kv - block;  // 3
  EXPECT_TRUE(is_blocked(m.at(qi, excluded_col)));
  // Neighbors of excluded col are kept.
  if (excluded_col - 1 >= 0)
    EXPECT_TRUE(is_kept(m.at(qi, excluded_col - 1)));
  // excluded_col + 1 is still within prefix (prefix_len-1 == excluded_col), so
  // excluded_col is the LAST prefix col; excluded_col+1 == prefix_len -> window.
  EXPECT_TRUE(is_kept(m.at(qi, excluded_col + 1)));
}

TEST(MtpMask, WindowIsFullyBidirectional) {
  const int kv = 8, block = 6;
  Mask m = build_mtp_window_mask(kv, block);
  const int prefix_len = kv - block;  // 2
  // Every window query attends to every window key, including "future" ones.
  for (int qi = prefix_len; qi < kv; ++qi) {
    for (int kj = prefix_len; kj < kv; ++kj) {
      EXPECT_TRUE(is_kept(m.at(qi, kj)));
    }
  }
  // Specifically, an earlier window row sees a later window col (non-causal).
  EXPECT_TRUE(is_kept(m.at(prefix_len, kv - 1)));
}

TEST(MtpMask, BlockSizeZeroReducesToCausal) {
  const int kv = 5;
  Mask mtp = build_mtp_window_mask(kv, 0);
  Mask ar = build_ar_mask(kv);
  ASSERT_EQ(mtp.data.size(), ar.data.size());
  for (size_t i = 0; i < mtp.data.size(); ++i) {
    // Both blocked or both kept.
    EXPECT_EQ(is_blocked(mtp.data[i]), is_blocked(ar.data[i])) << "i=" << i;
    EXPECT_EQ(is_kept(mtp.data[i]), is_kept(ar.data[i])) << "i=" << i;
  }
}

TEST(MtpMask, NoPrefixWhenKvEqualsBlock) {
  const int kv = 6, block = 6;
  Mask m = build_mtp_window_mask(kv, block);
  // prefix_len == 0 -> entire mask is the bidirectional window, fully kept.
  for (int qi = 0; qi < kv; ++qi)
    for (int kj = 0; kj < kv; ++kj)
      EXPECT_TRUE(is_kept(m.at(qi, kj))) << "qi=" << qi << " kj=" << kj;
}

}  // namespace
