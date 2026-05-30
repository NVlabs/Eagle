// la/decode/mask.h
//
// Decode-mask + position-id builders for Parallel Box Decoding (LocateAnything).
// Phase 0: pure-C++ reference implementation, batch == 1.
//
// This module ports the mask / position-id construction logic from the Python
// reference (see report `ported_from`):
//   - update_causal_mask_for_one_gen_window_2d  (MTP window mask)
//   - the AR / prefill causal mask
//   - position-id building in generate() / generate_utils.py
//
// Spec: Sections 5.3 (masks) and 5.4 (position ids).
//
// ---------------------------------------------------------------------------
// LAYOUT CONVENTIONS
// ---------------------------------------------------------------------------
// All attention masks are ADDITIVE float masks shaped logically as
// [batch=1, heads=1, q, kv] and returned as a flat std::vector<float> in
// row-major order: index(qi, kj) = qi * kv + kj.
//
//   value 0.0f      -> "keep"  (query qi may attend to key kj)
//   value -inf      -> "block" (query qi may NOT attend to key kj)
//
// We use -std::numeric_limits<float>::infinity() for the blocked value, matching
// the PyTorch convention of filling masked positions with the dtype's min/-inf
// before the softmax. (See MASK_NEG_INF below.)
//
// Position ids are returned as std::vector<int64_t> of length q (one entry per
// query/new token), matching torch.arange semantics used in the reference.
// ---------------------------------------------------------------------------

#ifndef LA_DECODE_MASK_H
#define LA_DECODE_MASK_H

#include <cstdint>
#include <limits>
#include <vector>

namespace la {
namespace decode {

// The additive value used for a blocked attention position.
// Defined as -infinity to match the PyTorch additive-mask convention used in
// the reference (masked_fill(..., float("-inf")) prior to softmax).
inline constexpr float MASK_NEG_INF = -std::numeric_limits<float>::infinity();

// Result of a mask build: a flat additive mask plus its logical [q, kv] shape.
struct Mask {
  int q = 0;             // number of query rows
  int kv = 0;            // number of key columns
  std::vector<float> data;  // size == q * kv, row-major: data[qi*kv + kj]

  // Convenience accessor (no bounds checking in release builds).
  float& at(int qi, int kj) { return data[static_cast<std::size_t>(qi) * kv + kj]; }
  float at(int qi, int kj) const {
    return data[static_cast<std::size_t>(qi) * kv + kj];
  }
};

// ---------------------------------------------------------------------------
// (1) build_ar_mask
// ---------------------------------------------------------------------------
// Standard lower-triangular causal additive mask for autoregressive attention.
//
// Two modes, both produced from the SAME function via `q_len`:
//   * Prefill : q_len == kv_len. Square lower-triangular mask. Query row qi
//               attends to key kj iff kj <= qi (0), else -inf.
//   * AR step : q_len == 1.       The single new query (logical position
//               kv_len-1) attends to ALL kv keys (every column kept, since the
//               new token is the most-recent position and may see all of the
//               cached past plus itself).
//
// General rule for an arbitrary q_len <= kv_len (right-aligned queries, i.e. the
// last `q_len` positions are the queries): query row qi maps to absolute
// position  abs = (kv_len - q_len) + qi.  It attends to key kj iff
//   kj <= abs   ->  keep (0)
//   kj >  abs   ->  block (-inf)
// For q_len == kv_len this is the plain lower triangle; for q_len == 1 the
// single row has abs == kv_len-1 so every column is kept.
//
// Returns a Mask with q == q_len, kv == kv_len.
Mask build_ar_mask(int kv_len, int q_len);

// Convenience overload: prefill square mask (q_len == kv_len).
inline Mask build_ar_mask(int kv_len) { return build_ar_mask(kv_len, kv_len); }

// ---------------------------------------------------------------------------
// (2) build_mtp_window_mask
// ---------------------------------------------------------------------------
// Multi-Token-Prediction (MTP) "one generation window" mask used during
// Parallel Box Decoding.
//
// The sequence of length `kv_len` is split into:
//   prefix_len  = kv_len - block_size      (the established "x0" context)
//   window      = the last `block_size` positions, absolute indices
//                 [prefix_len, kv_len)
//
// This builder produces a SQUARE additive mask of shape [kv_len, kv_len]
// (q == kv == kv_len) describing attention for the whole sequence in a single
// forward pass (the prefill-with-window pass). It has THREE regions:
//
//  (a) Causal x0 prefix.
//      For query rows qi in [0, prefix_len): standard causal triangle over the
//      prefix keys. qi attends to kj iff kj <= qi (and kj < prefix_len). All
//      window keys (kj >= prefix_len) are blocked for prefix queries.
//
//  (b) Bidirectional block_size x block_size window (bottom-right).
//      For query rows qi in [prefix_len, kv_len) and key cols kj in
//      [prefix_len, kv_len): ALL kept (0). The window is fully BIDIRECTIONAL:
//      every future-box query attends to every other future-box query/key
//      regardless of order. This is the key departure from causal decoding.
//
//  (c) Window -> prefix attention, with ONE excluded column.
//      For query rows qi in [prefix_len, kv_len) and key cols kj in
//      [0, prefix_len): kept (0) for ALL prefix keys EXCEPT the single column
//
//          excluded_col = kv_len - block_size - 1   (== prefix_len - 1)
//
//      which is set to -inf. i.e. the window attends to the entire x0 prefix
//      except the last prefix token. An off-by-one here shifts every predicted
//      box, so this index is load-bearing:
//          excluded_col = prefix_len - 1 = (kv_len - block_size) - 1.
//      If prefix_len == 0 there is no prefix and no excluded column.
//
// Defaults: block_size == 6 (six future boxes / MTP heads).
//
// Preconditions: kv_len >= block_size >= 0. (If block_size == 0 this reduces to
// a plain causal mask.)
//
// Returns a Mask with q == kv == kv_len.
Mask build_mtp_window_mask(int kv_len, int block_size = 6);

}  // namespace decode
}  // namespace la

#endif  // LA_DECODE_MASK_H
