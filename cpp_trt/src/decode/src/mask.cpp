// la/decode/mask.cpp  -- pure C++ reference for the decode masks.
//
// See include/la/decode/mask.h for the full index documentation. The comments
// here re-state the load-bearing indices inline so the implementation is
// auditable on its own.

#include "la/decode/mask.h"

#include <cassert>

namespace la {
namespace decode {

// ---------------------------------------------------------------------------
// (1) build_ar_mask
// ---------------------------------------------------------------------------
Mask build_ar_mask(int kv_len, int q_len) {
  assert(kv_len >= 0);
  assert(q_len >= 0);
  assert(q_len <= kv_len);

  Mask m;
  m.q = q_len;
  m.kv = kv_len;
  m.data.assign(static_cast<std::size_t>(q_len) * kv_len, MASK_NEG_INF);

  // Queries are right-aligned: query row qi corresponds to absolute sequence
  // position  abs = (kv_len - q_len) + qi.  It may attend to key kj iff
  // kj <= abs.  For q_len == kv_len this is the plain lower triangle; for
  // q_len == 1 the one row (abs == kv_len-1) keeps every column.
  const int offset = kv_len - q_len;
  for (int qi = 0; qi < q_len; ++qi) {
    const int abs_pos = offset + qi;
    for (int kj = 0; kj <= abs_pos && kj < kv_len; ++kj) {
      m.at(qi, kj) = 0.0f;
    }
  }
  return m;
}

// ---------------------------------------------------------------------------
// (2) build_mtp_window_mask
// ---------------------------------------------------------------------------
Mask build_mtp_window_mask(int kv_len, int block_size) {
  assert(block_size >= 0);
  assert(kv_len >= block_size);

  Mask m;
  m.q = kv_len;
  m.kv = kv_len;
  m.data.assign(static_cast<std::size_t>(kv_len) * kv_len, MASK_NEG_INF);

  // prefix_len: number of established "x0" context tokens.
  // window: absolute key/query indices [prefix_len, kv_len).
  const int prefix_len = kv_len - block_size;

  // The single prefix column the window must NOT see.
  //   excluded_col = kv_len - block_size - 1 == prefix_len - 1
  // (the last prefix token). Negative when prefix_len == 0 -> no exclusion.
  const int excluded_col = prefix_len - 1;

  // (a) Causal x0 prefix: rows qi in [0, prefix_len) over keys kj in
  //     [0, prefix_len), kept iff kj <= qi. Window keys remain blocked.
  for (int qi = 0; qi < prefix_len; ++qi) {
    for (int kj = 0; kj <= qi; ++kj) {  // kj < prefix_len guaranteed (kj<=qi<prefix_len)
      m.at(qi, kj) = 0.0f;
    }
  }

  // Window rows: qi in [prefix_len, kv_len).
  for (int qi = prefix_len; qi < kv_len; ++qi) {
    // (c) Window -> prefix: keep ALL prefix keys except excluded_col.
    for (int kj = 0; kj < prefix_len; ++kj) {
      if (kj == excluded_col) {
        continue;  // leave as -inf (blocked)
      }
      m.at(qi, kj) = 0.0f;
    }
    // (b) Bidirectional window: keep ALL window keys.
    for (int kj = prefix_len; kj < kv_len; ++kj) {
      m.at(qi, kj) = 0.0f;
    }
  }

  return m;
}

}  // namespace decode
}  // namespace la
