// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

#include "la/generate/sampling.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using la::generate::apply_repetition_penalty;
using la::generate::greedy_argmax;
using la::generate::softmax;

TEST(Sampling, RepetitionPenaltyNoOpAtOne) {
    std::vector<float> logits = {1.0f, -2.0f, 3.0f};
    const std::vector<float> before = logits;
    apply_repetition_penalty(logits, {0, 1, 2}, 1.0f);
    EXPECT_EQ(logits, before);
}

TEST(Sampling, RepetitionPenaltyPositiveDividedNegativeMultiplied) {
    // Matches generate_utils.apply_repetition_penalty: positive/penalty,
    // negative*penalty, only for seen ids.
    std::vector<float> logits = {2.0f, -4.0f, 5.0f, 0.0f};
    apply_repetition_penalty(logits, {0, 1, 3}, 2.0f);  // ids 0,1,3 seen; 2 not
    EXPECT_FLOAT_EQ(logits[0], 1.0f);    // 2 / 2
    EXPECT_FLOAT_EQ(logits[1], -8.0f);   // -4 * 2
    EXPECT_FLOAT_EQ(logits[2], 5.0f);    // unseen -> unchanged
    EXPECT_FLOAT_EQ(logits[3], 0.0f);    // 0 * 2 == 0
}

TEST(Sampling, RepetitionPenaltyIgnoresOutOfRangeAndDuplicates) {
    std::vector<float> logits = {3.0f, 3.0f};
    apply_repetition_penalty(logits, {0, 0, 0, 99, -1}, 3.0f);
    EXPECT_FLOAT_EQ(logits[0], 1.0f);  // divided once, duplicates harmless
    EXPECT_FLOAT_EQ(logits[1], 3.0f);  // id 1 not in seen set
}

TEST(Sampling, GreedyArgmaxLowestIndexOnTie) {
    EXPECT_EQ(greedy_argmax({1.0f, 5.0f, 5.0f, 2.0f}), 1);
    EXPECT_EQ(greedy_argmax({-1.0f, -2.0f, -0.5f}), 2);
    EXPECT_EQ(greedy_argmax({}), -1);
}

TEST(Sampling, SoftmaxSumsToOne) {
    std::vector<float> out;
    softmax({1.0f, 2.0f, 3.0f}, out);
    double s = 0.0;
    for (float v : out) s += v;
    EXPECT_NEAR(s, 1.0, 1e-5);
    EXPECT_GT(out[2], out[1]);
    EXPECT_GT(out[1], out[0]);
}

TEST(Sampling, SoftmaxStableLargeValues) {
    std::vector<float> out;
    softmax({1000.0f, 1001.0f}, out);  // would overflow without max-subtraction
    double s = 0.0;
    for (float v : out) s += v;
    EXPECT_NEAR(s, 1.0, 1e-5);
    EXPECT_TRUE(std::isfinite(out[0]));
}

}  // namespace
