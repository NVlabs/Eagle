// la/decode/position_ids.h
//
// Position-id builders for Parallel Box Decoding (LocateAnything), Phase 0.
// Pure C++, batch == 1.
//
// Ported from the position-id construction in generate() / generate_utils.py.
// Spec: Section 5.4.
//
// Position ids are returned as std::vector<int64_t> (matching torch.long /
// torch.arange semantics in the reference).

#ifndef LA_DECODE_POSITION_IDS_H
#define LA_DECODE_POSITION_IDS_H

#include <cstdint>
#include <vector>

namespace la {
namespace decode {

// ---------------------------------------------------------------------------
// (3) build_position_ids_mtp
// ---------------------------------------------------------------------------
// MTP / Parallel Box Decoding position ids.
//
// Base sequence is  arange(start_idx, total_len), i.e. the half-open range
//   [start_idx, total_len)  ->  length == (total_len - start_idx)
// matching torch.arange(start_idx, total_len).
//
// Then the LAST `n_future` entries each have 1 SUBTRACTED from them. These are
// the `n_future` future-box (MTP) tokens, which share the position of the token
// immediately preceding the window so that all parallel boxes are decoded as if
// they occupy the same "next" position rather than consecutive positions.
//
// Example: start_idx=0, total_len=10, n_future=6
//   base    = [0,1,2,3,4,5,6,7,8,9]
//   result  = [0,1,2,3,4,5,6,7,8,9]  with last 6 minus 1
//           = [0,1,2,3,3,4,5,6,7,8]
//   (indices 4..9 had 1 subtracted)
//
// Preconditions: total_len >= start_idx; 0 <= n_future <= (total_len-start_idx).
// If n_future == 0 this is a plain arange.
//
// Returns a vector of length (total_len - start_idx).
std::vector<std::int64_t> build_position_ids_mtp(std::int64_t start_idx,
                                                 std::int64_t total_len,
                                                 std::int64_t n_future = 6);

// ---------------------------------------------------------------------------
// (4) build_position_ids_ar
// ---------------------------------------------------------------------------
// Plain autoregressive position ids: arange(start_idx, start_idx + q_len), i.e.
//   [start_idx, start_idx + q_len)  ->  length == q_len.
// For an AR decode step q_len == 1, giving the single id [start_idx].
//
// Preconditions: q_len >= 0.
//
// Returns a vector of length q_len.
std::vector<std::int64_t> build_position_ids_ar(std::int64_t start_idx,
                                                int q_len);

}  // namespace decode
}  // namespace la

#endif  // LA_DECODE_POSITION_IDS_H
