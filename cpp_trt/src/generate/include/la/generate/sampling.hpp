// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef LA_GENERATE_SAMPLING_HPP
#define LA_GENERATE_SAMPLING_HPP

#include <cstdint>
#include <vector>

// Greedy + repetition-penalty sampling, ported from
// Embodied/eaglevl/utils/locany/generate_utils.py.
//
// Per the locked decision (spec §2, §5.4): greedy argmax (T=0) + repetition
// penalty only. top_p / top_k / temperature sampling are intentionally NOT
// implemented here.
//
// All functions operate on a single distribution (one position) given as a
// flat std::vector<float> of length vocab_size. Pure functions, no engine.

namespace la::generate {

// In-place repetition penalty over one logit row.
//
// EXACTLY matches `apply_repetition_penalty` in generate_utils.py for a single
// row: for every token id that appears in `seen_token_ids`, a positive logit is
// divided by `penalty` and a negative logit is multiplied by `penalty`. Tokens
// at logit value 0 are unchanged (matches the python `logits > 0` / `~positive`
// split, where 0 falls in the "negative" branch but 0*penalty == 0). Ids
// outside [0, vocab_size) are ignored, matching the python valid-token clamp.
// `penalty == 1.0` is a no-op (matches the python early return).
//
// The python applies the penalty over the *unique* set of generated ids; we
// take the running id set directly. Duplicates in `seen_token_ids` are harmless
// because the operation is applied per-vocab-id, not per-occurrence (we mark a
// boolean seen-mask first, exactly like the python `token_mask`).
void apply_repetition_penalty(std::vector<float>& logits,
                              const std::vector<std::int64_t>& seen_token_ids,
                              float penalty = 1.1f);

// Greedy argmax over one logit row. Returns the index of the maximum element.
// Ties resolve to the lowest index, matching torch.max / argmax semantics used
// by the greedy branch (`probs.max(dim=-1)`). Softmax is monotonic, so argmax
// over logits == argmax over probs (the python takes max over probs; we take it
// over logits to avoid a redundant softmax — identical result for greedy).
// Returns -1 for an empty row.
[[nodiscard]] std::int64_t greedy_argmax(const std::vector<float>& logits);

// Numerically-stable softmax of one logit row into `out` (resized to match).
// Provided because the box-decode FSM (decode_bbox_avg / is_valid_box_frame)
// thresholds on probabilities, not logits. lm_head + softmax are kept FP32 to
// match the spec's FP32-softmax parity requirement (spec §8, Risk R4).
void softmax(const std::vector<float>& logits, std::vector<float>& out);

}  // namespace la::generate

#endif  // LA_GENERATE_SAMPLING_HPP
