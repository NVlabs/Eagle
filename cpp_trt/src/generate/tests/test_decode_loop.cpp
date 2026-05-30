// SPDX-License-Identifier: Apache-2.0

#include "la/generate/decode_loop.hpp"

#include <gtest/gtest.h>

#include <vector>

#include "la/config/token_ids.hpp"
#include "la/llm/LlmEngine.h"

namespace {

using la::config::TokenIds;
using la::generate::DecodeOptions;
using la::generate::DecodeResult;
using la::generate::EmbedLookup;
using la::generate::GenerationMode;
using la::generate::run_ar_decode;

// A scripted mock ILlmEngine: returns a pre-programmed sequence of "next-token"
// logit rows, one per step() call. Each row is a one-hot over the vocab at the
// scripted id (so greedy_argmax picks it). Tracks the AR step contract + KV.
class MockEngine : public ::la::llm::ILlmEngine {
public:
    MockEngine(int vocab, std::vector<std::int64_t> scripted_next_ids)
        : vocab_(vocab), scripted_(std::move(scripted_next_ids)) {}

    ::la::llm::StepOutput step(const ::la::llm::StepInput& in) override {
        // AR contract: q_len 1, attn_mask null (built-in causal), kv_len ==
        // current cache, position id == cache length.
        EXPECT_EQ(in.q_len, 1);
        EXPECT_EQ(in.attn_mask_4d, nullptr);
        EXPECT_EQ(in.kv_len, kv_len_);
        EXPECT_NE(in.inputs_embeds, nullptr);
        EXPECT_NE(in.position_ids, nullptr);
        EXPECT_EQ(in.position_ids[0], static_cast<std::int64_t>(kv_len_));
        kv_len_ += in.q_len;  // engine appended one key

        ::la::llm::StepOutput out;
        out.vocab = vocab_;
        out.logits.assign(static_cast<std::size_t>(vocab_), 0.0f);
        const std::int64_t id =
            (call_ < static_cast<std::int64_t>(scripted_.size()))
                ? scripted_[static_cast<std::size_t>(call_)]
                : 0;
        if (id >= 0 && id < vocab_) {
            out.logits[static_cast<std::size_t>(id)] = 100.0f;  // dominant
        }
        ++call_;
        return out;
    }

    void reset_kv() override { kv_len_ = 0; }
    int kv_len() const override { return kv_len_; }
    void rollback_kv(int to_len) override { kv_len_ = to_len; }
    ::la::llm::Backend backend() const override {
        return ::la::llm::Backend::StandaloneTrt;
    }
    bool supports_dense_4d_mask() const override { return true; }

    // Test harness: simulate the prefill having populated `n` keys.
    void set_prefilled(int n) { kv_len_ = n; }
    std::int64_t calls() const { return call_; }

private:
    int vocab_;
    std::vector<std::int64_t> scripted_;
    std::int64_t call_ = 0;
    int kv_len_ = 0;
};

std::vector<float> OneHot(int vocab, std::int64_t id) {
    std::vector<float> v(static_cast<std::size_t>(vocab), 0.0f);
    if (id >= 0 && id < vocab) v[static_cast<std::size_t>(id)] = 100.0f;
    return v;
}

EmbedLookup ZeroEmbed() {
    return [](std::int64_t, float* out, std::int64_t hidden) {
        for (std::int64_t i = 0; i < hidden; ++i) out[i] = 0.0f;
    };
}

TEST(DecodeLoop, GeneratesUntilImEnd) {
    TokenIds tok = TokenIds::Fallback();
    const int vocab = static_cast<int>(tok.switch_token) + 10;  // cover ids
    MockEngine eng(vocab, {6, 7, tok.im_end});

    std::vector<std::int64_t> prompt = {1, 2, 3};
    eng.set_prefilled(static_cast<int>(prompt.size()));  // prefill populated KV

    DecodeOptions opts;
    opts.mode = GenerationMode::Slow;
    opts.repetition_penalty = 1.1f;
    opts.max_new_tokens = 100;
    opts.hidden = 4;

    DecodeResult r = run_ar_decode(eng, OneHot(vocab, 5), prompt, tok,
                                   ZeroEmbed(), opts);

    // First gen token (5) from prefill, then 6, 7, im_end.
    EXPECT_EQ(r.generated_ids,
              (std::vector<std::int64_t>{5, 6, 7, tok.im_end}));
    EXPECT_TRUE(r.hit_im_end);
    // token 5 -> step, 6 -> step, 7 -> step; the 3rd step yields im_end, emitted
    // and stops the loop before stepping it. So 3 steps.
    EXPECT_EQ(r.steps, 3);
    EXPECT_EQ(eng.calls(), 3);
}

TEST(DecodeLoop, StopsAtMaxNewTokens) {
    TokenIds tok = TokenIds::Fallback();
    const int vocab = static_cast<int>(tok.switch_token) + 10;
    MockEngine eng(vocab, {10, 11, 12, 13, 14, 15});

    std::vector<std::int64_t> prompt = {1, 2};
    eng.set_prefilled(static_cast<int>(prompt.size()));

    DecodeOptions opts;
    opts.mode = GenerationMode::Slow;
    opts.repetition_penalty = 1.0f;  // disable to keep ids deterministic
    opts.max_new_tokens = 3;  // prompt_len(2) + 3 = cap 5
    opts.hidden = 2;

    DecodeResult r = run_ar_decode(eng, OneHot(vocab, 9), prompt, tok,
                                   ZeroEmbed(), opts);

    EXPECT_FALSE(r.hit_im_end);
    // committed=2. Emit 9 (committed+1=3 <5 -> step). Emit 10 (3,+1=4 <5 ->
    // step). Emit 11 (4,+1=5 >=5 -> stop, no step). generated={9,10,11}, 2 steps.
    EXPECT_EQ(r.generated_ids, (std::vector<std::int64_t>{9, 10, 11}));
    EXPECT_EQ(r.steps, 2);
}

TEST(DecodeLoop, RepetitionPenaltyAffectsChoice) {
    TokenIds tok = TokenIds::Fallback();
    const int vocab = 8;

    // Prefill: ids 3 and 4 close; 3 already in prompt so penalty demotes it.
    std::vector<float> prefill(static_cast<std::size_t>(vocab), 0.0f);
    prefill[3] = 2.0f;   // seen in prompt -> /1.1 = 1.818
    prefill[4] = 1.9f;   // unseen -> stays 1.9 (wins after penalty)

    MockEngine eng(vocab, {tok.im_end});
    std::vector<std::int64_t> prompt = {3};  // makes id 3 "seen"
    eng.set_prefilled(static_cast<int>(prompt.size()));

    DecodeOptions opts;
    opts.repetition_penalty = 1.1f;
    opts.max_new_tokens = 10;
    opts.hidden = 1;

    DecodeResult r = run_ar_decode(eng, prefill, prompt, tok, ZeroEmbed(), opts);

    ASSERT_FALSE(r.generated_ids.empty());
    EXPECT_EQ(r.generated_ids[0], 4);  // penalty flipped the argmax to 4
}

}  // namespace
