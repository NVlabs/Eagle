// la/decode/position_ids.cpp -- pure C++ reference for MTP / AR position ids.
//
// See include/la/decode/position_ids.h for full documentation.

#include "la/decode/position_ids.h"

#include <cassert>

namespace la {
namespace decode {

// (3) MTP position ids: arange(start_idx, total_len), then subtract 1 from the
//     last n_future entries.
std::vector<std::int64_t> build_position_ids_mtp(std::int64_t start_idx,
                                                 std::int64_t total_len,
                                                 std::int64_t n_future) {
  assert(total_len >= start_idx);
  const std::int64_t len = total_len - start_idx;
  assert(n_future >= 0);
  assert(n_future <= len);

  std::vector<std::int64_t> ids(static_cast<std::size_t>(len));
  for (std::int64_t i = 0; i < len; ++i) {
    ids[static_cast<std::size_t>(i)] = start_idx + i;
  }
  // Subtract 1 from the last n_future entries (the future-box / MTP tokens).
  for (std::int64_t i = len - n_future; i < len; ++i) {
    ids[static_cast<std::size_t>(i)] -= 1;
  }
  return ids;
}

// (4) AR position ids: arange(start_idx, start_idx + q_len).
std::vector<std::int64_t> build_position_ids_ar(std::int64_t start_idx,
                                                int q_len) {
  assert(q_len >= 0);
  std::vector<std::int64_t> ids(static_cast<std::size_t>(q_len));
  for (int i = 0; i < q_len; ++i) {
    ids[static_cast<std::size_t>(i)] = start_idx + i;
  }
  return ids;
}

}  // namespace decode
}  // namespace la
