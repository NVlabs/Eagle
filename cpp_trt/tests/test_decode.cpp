// Unit tests for la_decode (mask + position-id builders).
//
// Each test computes the EXPECTED value with an independent, naive
// reimplementation inside the test itself (NOT by calling the code under test),
// so off-by-one / region-boundary bugs are caught. This is internal-consistency
// and spec-conformance testing, NOT PyTorch parity (impossible in this env).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "la/decode/mask.h"
#include "la/decode/position_ids.h"

namespace {

using la::decode::Mask;

// Independent predicate: a mask entry is "blocked" iff it is negative infinity
// (or, defensively, any very-large-negative value the impl might use).
bool IsBlocked(float v) {
  if (std::isinf(v) && v < 0.0f) return true;
  // Tolerate finfo-min style fills just in case.
  return v < -1.0e30f;
}

bool IsKeep(float v) { return v == 0.0f; }

}  // namespace

// ---------------------------------------------------------------------------
// build_ar_mask
// ---------------------------------------------------------------------------

// Prefill case: q_len == kv_len => pure lower-triangular causal mask.
TEST(ArMask, PrefillIsLowerTriangular) {
  for (int n : {1, 2, 3, 5, 8}) {
    Mask m = la::decode::build_ar_mask(n);  // q_len == kv_len overload
    ASSERT_EQ(m.q, n);
    ASSERT_EQ(m.kv, n);
    ASSERT_EQ(static_cast<int>(m.data.size()), n * n);
    for (int qi = 0; qi < n; ++qi) {
      for (int kj = 0; kj < n; ++kj) {
        float v = m.data[qi * n + kj];
        // Independent expectation: keep iff kj <= qi (lower triangle incl. diag).
        if (kj <= qi) {
          EXPECT_TRUE(IsKeep(v)) << "n=" << n << " qi=" << qi << " kj=" << kj;
        } else {
          EXPECT_TRUE(IsBlocked(v)) << "n=" << n << " qi=" << qi << " kj=" << kj;
        }
      }
    }
  }
}

// Right-aligned queries: row qi maps to absolute position (kv_len-q_len)+qi,
// keep kj <= abs.
TEST(ArMask, RightAlignedGeneral) {
  struct Case { int kv; int q; };
  for (Case c : {Case{6, 3}, Case{10, 4}, Case{5, 5}, Case{7, 1}}) {
    Mask m = la::decode::build_ar_mask(c.kv, c.q);
    ASSERT_EQ(m.q, c.q);
    ASSERT_EQ(m.kv, c.kv);
    ASSERT_EQ(static_cast<int>(m.data.size()), c.q * c.kv);
    for (int qi = 0; qi < c.q; ++qi) {
      int abs = (c.kv - c.q) + qi;  // independent absolute position
      for (int kj = 0; kj < c.kv; ++kj) {
        float v = m.data[qi * c.kv + kj];
        if (kj <= abs) {
          EXPECT_TRUE(IsKeep(v)) << "kv=" << c.kv << " q=" << c.q
                                 << " qi=" << qi << " kj=" << kj;
        } else {
          EXPECT_TRUE(IsBlocked(v)) << "kv=" << c.kv << " q=" << c.q
                                    << " qi=" << qi << " kj=" << kj;
        }
      }
    }
  }
}

// Single AR step (q_len==1): the one row sees all kv (full row of keeps).
TEST(ArMask, SingleStepSeesAllKv) {
  for (int kv : {1, 4, 9}) {
    Mask m = la::decode::build_ar_mask(kv, 1);
    ASSERT_EQ(m.q, 1);
    ASSERT_EQ(m.kv, kv);
    for (int kj = 0; kj < kv; ++kj) {
      EXPECT_TRUE(IsKeep(m.data[kj])) << "kv=" << kv << " kj=" << kj;
    }
  }
}

// ---------------------------------------------------------------------------
// build_mtp_window_mask
// ---------------------------------------------------------------------------

// Independent reference builder of the MTP window mask per the spec:
//   square [kv,kv]; prefix_len = kv - block_size.
//   prefix rows [0,prefix_len): causal over prefix, window keys blocked.
//   window rows [prefix_len,kv) x window cols: ALL kept (bidirectional).
//   window rows x prefix cols: all kept EXCEPT excluded_col = prefix_len-1 (-inf).
//   prefix_len==0 -> whole mask bidirectional, no exclusion.
std::vector<bool> RefMtpKeep(int kv, int block_size) {
  std::vector<bool> keep(static_cast<size_t>(kv) * kv, false);
  int prefix_len = kv - block_size;
  if (prefix_len < 0) prefix_len = 0;
  int excluded_col = prefix_len - 1;  // kv - block_size - 1
  for (int qi = 0; qi < kv; ++qi) {
    bool q_in_window = (qi >= prefix_len);
    for (int kj = 0; kj < kv; ++kj) {
      bool kj_in_prefix = (kj < prefix_len);
      bool k;
      if (prefix_len == 0) {
        k = true;  // fully bidirectional
      } else if (!q_in_window) {
        // prefix row: causal over prefix only, window keys blocked
        k = kj_in_prefix && (kj <= qi);
      } else {
        // window row
        if (kj_in_prefix) {
          k = (kj != excluded_col);
        } else {
          k = true;  // window col, bidirectional
        }
      }
      keep[static_cast<size_t>(qi) * kv + kj] = k;
    }
  }
  return keep;
}

TEST(MtpWindowMask, MatchesIndependentReference) {
  const int block_size = 6;
  for (int kv : {7, 8, 10, 12, 6 /*prefix_len==0*/, 9}) {
    Mask m = la::decode::build_mtp_window_mask(kv, block_size);
    ASSERT_EQ(m.q, kv);
    ASSERT_EQ(m.kv, kv);
    ASSERT_EQ(static_cast<int>(m.data.size()), kv * kv);
    std::vector<bool> ref = RefMtpKeep(kv, block_size);
    for (int qi = 0; qi < kv; ++qi) {
      for (int kj = 0; kj < kv; ++kj) {
        float v = m.data[qi * kv + kj];
        bool want_keep = ref[static_cast<size_t>(qi) * kv + kj];
        if (want_keep) {
          EXPECT_TRUE(IsKeep(v))
              << "kv=" << kv << " qi=" << qi << " kj=" << kj << " expected keep";
        } else {
          EXPECT_TRUE(IsBlocked(v))
              << "kv=" << kv << " qi=" << qi << " kj=" << kj << " expected block";
        }
      }
    }
  }
}

// Targeted: the single blocked prefix column inside the window region must be
// exactly -inf at (kv-block_size-1), and its neighbors must be kept.
TEST(MtpWindowMask, SingleExcludedColumnIsBlocked) {
  const int block_size = 6;
  for (int kv : {8, 10, 12}) {
    Mask m = la::decode::build_mtp_window_mask(kv, block_size);
    int prefix_len = kv - block_size;
    int excluded_col = prefix_len - 1;
    int first_window_row = prefix_len;
    // The excluded column is blocked for every window row.
    for (int qi = first_window_row; qi < kv; ++qi) {
      EXPECT_TRUE(IsBlocked(m.data[qi * kv + excluded_col]))
          << "kv=" << kv << " window row qi=" << qi
          << " excluded_col=" << excluded_col;
      // A different prefix column (excluded_col-1, if it exists) is kept.
      if (excluded_col - 1 >= 0) {
        EXPECT_TRUE(IsKeep(m.data[qi * kv + (excluded_col - 1)]))
            << "kv=" << kv << " qi=" << qi << " col=" << (excluded_col - 1);
      }
    }
  }
}

// Bidirectional bottom-right block: window rows x window cols all kept.
TEST(MtpWindowMask, BottomRightBlockIsBidirectional) {
  const int block_size = 6;
  for (int kv : {8, 11, 14}) {
    Mask m = la::decode::build_mtp_window_mask(kv, block_size);
    int prefix_len = kv - block_size;
    for (int qi = prefix_len; qi < kv; ++qi) {
      for (int kj = prefix_len; kj < kv; ++kj) {
        EXPECT_TRUE(IsKeep(m.data[qi * kv + kj]))
            << "kv=" << kv << " qi=" << qi << " kj=" << kj
            << " bottom-right should be kept";
      }
    }
  }
}

// prefix_len == 0 (kv == block_size): entire mask bidirectional, no exclusion.
TEST(MtpWindowMask, PrefixLenZeroFullyBidirectional) {
  const int block_size = 6;
  int kv = block_size;
  Mask m = la::decode::build_mtp_window_mask(kv, block_size);
  for (int i = 0; i < kv * kv; ++i) {
    EXPECT_TRUE(IsKeep(m.data[i])) << "idx=" << i << " expected fully bidirectional";
  }
}

// ---------------------------------------------------------------------------
// build_position_ids_mtp
// ---------------------------------------------------------------------------

TEST(PositionIdsMtp, ArangeThenMinusOneOnLastSix) {
  struct Case { int64_t start; int64_t total; int64_t n_future; };
  for (Case c : {Case{0, 10, 6}, Case{3, 20, 6}, Case{5, 11, 6}, Case{0, 6, 6},
                 Case{2, 9, 4}}) {
    std::vector<int64_t> got =
        la::decode::build_position_ids_mtp(c.start, c.total, c.n_future);
    // Independent reference: arange(start,total) then subtract 1 from last
    // n_future entries.
    int64_t len = c.total - c.start;
    ASSERT_GE(len, 0);
    std::vector<int64_t> ref(static_cast<size_t>(len));
    for (int64_t i = 0; i < len; ++i) ref[i] = c.start + i;
    for (int64_t i = std::max<int64_t>(0, len - c.n_future); i < len; ++i) {
      ref[i] -= 1;
    }
    ASSERT_EQ(got.size(), ref.size())
        << "start=" << c.start << " total=" << c.total;
    for (size_t i = 0; i < ref.size(); ++i) {
      EXPECT_EQ(got[i], ref[i]) << "i=" << i << " start=" << c.start
                               << " total=" << c.total;
    }
  }
}

// Concrete hand-computed vector from the module spec example.
TEST(PositionIdsMtp, KnownVector) {
  // arange(0,10) = 0..9 ; subtract 1 from last 6 -> [0,1,2,3,3,4,5,6,7,8]
  std::vector<int64_t> got = la::decode::build_position_ids_mtp(0, 10, 6);
  std::vector<int64_t> want = {0, 1, 2, 3, 3, 4, 5, 6, 7, 8};
  EXPECT_EQ(got, want);
}

// ---------------------------------------------------------------------------
// build_position_ids_ar
// ---------------------------------------------------------------------------

TEST(PositionIdsAr, PlainArange) {
  struct Case { int64_t start; int q; };
  for (Case c : {Case{0, 1}, Case{0, 5}, Case{7, 1}, Case{100, 4}}) {
    std::vector<int64_t> got = la::decode::build_position_ids_ar(c.start, c.q);
    ASSERT_EQ(static_cast<int>(got.size()), c.q);
    for (int i = 0; i < c.q; ++i) {
      EXPECT_EQ(got[i], c.start + i) << "start=" << c.start << " i=" << i;
    }
  }
}
