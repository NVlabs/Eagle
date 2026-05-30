// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "la/generate/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace la::generate {

void apply_repetition_penalty(std::vector<float>& logits,
                              const std::vector<std::int64_t>& seen_token_ids,
                              float penalty) {
    // python: `if repetition_penalty == 1.0: return logits`
    if (penalty == 1.0f) {
        return;
    }
    const std::int64_t vocab = static_cast<std::int64_t>(logits.size());
    if (vocab == 0) {
        return;
    }

    // Build the [V] seen mask (python `token_mask`), clamping to valid ids.
    std::vector<char> seen(static_cast<std::size_t>(vocab), 0);
    for (std::int64_t id : seen_token_ids) {
        if (id >= 0 && id < vocab) {
            seen[static_cast<std::size_t>(id)] = 1;
        }
    }

    // python:
    //   positive = logits > 0
    //   logits = where(seen & positive, logits / penalty, logits)
    //   logits = where(seen & ~positive, logits * penalty, logits)
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (!seen[i]) {
            continue;
        }
        float& v = logits[i];
        if (v > 0.0f) {
            v /= penalty;
        } else {
            // Includes v == 0 (0 * penalty == 0, a no-op) and v < 0, matching
            // the python `~positive` branch exactly.
            v *= penalty;
        }
    }
}

std::int64_t greedy_argmax(const std::vector<float>& logits) {
    if (logits.empty()) {
        return -1;
    }
    std::size_t best = 0;
    float best_v = logits[0];
    for (std::size_t i = 1; i < logits.size(); ++i) {
        // Strict `>` keeps the lowest index on ties (torch.argmax semantics).
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return static_cast<std::int64_t>(best);
}

void softmax(const std::vector<float>& logits, std::vector<float>& out) {
    out.resize(logits.size());
    if (logits.empty()) {
        return;
    }
    const float m = *std::max_element(logits.begin(), logits.end());
    double denom = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const double e = std::exp(static_cast<double>(logits[i]) -
                                  static_cast<double>(m));
        out[i] = static_cast<float>(e);
        denom += e;
    }
    const float inv = static_cast<float>(1.0 / denom);
    for (float& v : out) {
        v *= inv;
    }
}

}  // namespace la::generate
