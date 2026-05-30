// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// Unit tests for the la_generate module (Phase-2 PURE-C++ surface):
//   * sampling.hpp : apply_repetition_penalty, greedy_argmax, softmax
//   * coord.hpp    : coord_token_to_value, value_to_pixel, coord_token_to_pixel
//   * parse.hpp    : parse_token_stream (id-native), parse_text (text fallback)
//
// Each test computes its EXPECTED value with an INDEPENDENT in-test reference
// (internal-consistency / spec-conformance), NOT PyTorch parity. The references
// here are re-derivations of the ported numeric rules straight from the spec /
// header contracts, written separately from the production implementation.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "la/config/token_ids.hpp"
#include "la/generate/coord.hpp"
#include "la/generate/parse.hpp"
#include "la/generate/sampling.hpp"

namespace {

using ::la::generate::Detection;
using ::la::generate::DetectionKind;
using ::la::generate::ParseResult;
using ::la::generate::ResizeTransform;
using TokenIds = ::la::config::TokenIds;

// ---------------------------------------------------------------------------
// Independent reference reimplementations (kept deliberately separate from the
// production code under test).
// ---------------------------------------------------------------------------

// Reference repetition penalty: build a unique seen-mask, then for every seen
// id in range divide positive logits and multiply negative logits. 0 unchanged.
std::vector<float> ref_apply_rep_penalty(std::vector<float> logits,
                                         const std::vector<std::int64_t>& seen,
                                         float penalty) {
    if (penalty == 1.0f) return logits;
    const std::int64_t vocab = static_cast<std::int64_t>(logits.size());
    std::vector<char> mask(logits.size(), 0);
    for (std::int64_t id : seen) {
        if (id >= 0 && id < vocab) mask[static_cast<std::size_t>(id)] = 1;
    }
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (!mask[i]) continue;
        if (logits[i] > 0.0f) {
            logits[i] = logits[i] / penalty;
        } else {
            logits[i] = logits[i] * penalty;
        }
    }
    return logits;
}

// Reference argmax: lowest index wins ties; -1 for empty.
std::int64_t ref_argmax(const std::vector<float>& v) {
    if (v.empty()) return -1;
    std::int64_t best = 0;
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[best]) best = static_cast<std::int64_t>(i);
    }
    return best;
}

// Reference numerically-stable softmax.
std::vector<float> ref_softmax(const std::vector<float>& v) {
    std::vector<float> out(v.size());
    if (v.empty()) return out;
    float m = v[0];
    for (float x : v) m = std::max(m, x);
    double sum = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        double e = std::exp(static_cast<double>(v[i] - m));
        out[i] = static_cast<float>(e);
        sum += e;
    }
    for (float& x : out) x = static_cast<float>(x / sum);
    return out;
}

// ---------------------------------------------------------------------------
// sampling: apply_repetition_penalty
// ---------------------------------------------------------------------------

TEST(Sampling, RepPenaltyMatchesIndependentReimpl) {
    std::vector<float> logits = {2.0f, -3.0f, 0.0f, 5.0f, -1.0f};
    std::vector<std::int64_t> seen = {0, 1, 3};  // not 2, not 4
    const float penalty = 1.1f;

    std::vector<float> expected = ref_apply_rep_penalty(logits, seen, penalty);
    std::vector<float> got = logits;
    la::generate::apply_repetition_penalty(got, seen, penalty);

    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_FLOAT_EQ(got[i], expected[i]) << "index " << i;
    }
    // Spot-check the rule directly: positive id 0 divided, negative id 1
    // multiplied, unseen id 2 (which is 0) unchanged, unseen id 4 unchanged.
    EXPECT_FLOAT_EQ(got[0], 2.0f / penalty);
    EXPECT_FLOAT_EQ(got[1], -3.0f * penalty);
    EXPECT_FLOAT_EQ(got[2], 0.0f);
    EXPECT_FLOAT_EQ(got[3], 5.0f / penalty);
    EXPECT_FLOAT_EQ(got[4], -1.0f);  // unseen
}

TEST(Sampling, RepPenaltyOnlyTouchesSeenIds) {
    std::vector<float> base = {1.0f, -2.0f, 3.0f, -4.0f};
    std::vector<float> got = base;
    la::generate::apply_repetition_penalty(got, /*seen=*/{}, 1.1f);
    // Empty seen set => no change.
    for (std::size_t i = 0; i < got.size(); ++i) EXPECT_FLOAT_EQ(got[i], base[i]);
}

TEST(Sampling, RepPenaltyNoOpAtOne) {
    std::vector<float> base = {1.0f, -2.0f, 3.0f};
    std::vector<float> got = base;
    la::generate::apply_repetition_penalty(got, {0, 1, 2}, 1.0f);
    for (std::size_t i = 0; i < got.size(); ++i) EXPECT_FLOAT_EQ(got[i], base[i]);
}

TEST(Sampling, RepPenaltyIgnoresOutOfRangeAndDuplicateIds) {
    std::vector<float> logits = {2.0f, -2.0f};
    // id 5 is out of range; id 0 duplicated should be harmless (per-vocab op).
    std::vector<std::int64_t> seen = {0, 0, 5, -1};
    std::vector<float> expected = ref_apply_rep_penalty(logits, seen, 1.1f);
    std::vector<float> got = logits;
    la::generate::apply_repetition_penalty(got, seen, 1.1f);
    ASSERT_EQ(got.size(), expected.size());
    EXPECT_FLOAT_EQ(got[0], 2.0f / 1.1f);  // applied once, not twice
    EXPECT_FLOAT_EQ(got[1], -2.0f);        // id 1 not in seen
    for (std::size_t i = 0; i < got.size(); ++i) EXPECT_FLOAT_EQ(got[i], expected[i]);
}

// ---------------------------------------------------------------------------
// sampling: greedy_argmax
// ---------------------------------------------------------------------------

TEST(Sampling, GreedyArgmaxMatchesReference) {
    std::vector<float> v = {-1.0f, 0.5f, 9.0f, 9.0f, 3.0f};
    EXPECT_EQ(la::generate::greedy_argmax(v), ref_argmax(v));
    EXPECT_EQ(la::generate::greedy_argmax(v), 2);  // lowest of the two 9.0 ties
}

TEST(Sampling, GreedyArgmaxTieBreaksToLowestIndex) {
    std::vector<float> v = {5.0f, 5.0f, 5.0f};
    EXPECT_EQ(la::generate::greedy_argmax(v), 0);
}

TEST(Sampling, GreedyArgmaxEmptyReturnsMinusOne) {
    std::vector<float> v;
    EXPECT_EQ(la::generate::greedy_argmax(v), -1);
}

TEST(Sampling, GreedyArgmaxSingleElement) {
    std::vector<float> v = {-42.0f};
    EXPECT_EQ(la::generate::greedy_argmax(v), 0);
}

// ---------------------------------------------------------------------------
// sampling: softmax
// ---------------------------------------------------------------------------

TEST(Sampling, SoftmaxMatchesStableReference) {
    std::vector<float> v = {1.0f, 2.0f, 3.0f, -1.0f};
    std::vector<float> got;
    la::generate::softmax(v, got);
    std::vector<float> expected = ref_softmax(v);
    ASSERT_EQ(got.size(), expected.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], 1e-6) << "index " << i;
        sum += got[i];
    }
    EXPECT_NEAR(sum, 1.0, 1e-5);
    // argmax of softmax == argmax of logits (monotonic).
    EXPECT_EQ(ref_argmax(got), la::generate::greedy_argmax(v));
}

TEST(Sampling, SoftmaxHandlesLargeValuesWithoutOverflow) {
    std::vector<float> v = {1000.0f, 1000.0f, 1001.0f};
    std::vector<float> got;
    la::generate::softmax(v, got);
    ASSERT_EQ(got.size(), 3u);
    for (float p : got) {
        EXPECT_TRUE(std::isfinite(p));
        EXPECT_GE(p, 0.0f);
        EXPECT_LE(p, 1.0f);
    }
    EXPECT_GT(got[2], got[0]);  // larger logit -> larger prob
}

// ---------------------------------------------------------------------------
// coord: coord_token_to_value / value_to_pixel / coord_token_to_pixel
// ---------------------------------------------------------------------------

TEST(Coord, TokenToValueMidpoint) {
    const std::int64_t coord_start = 151000;  // arbitrary band start
    // (coord_start + 500 - coord_start) / 1000 == 0.5
    EXPECT_DOUBLE_EQ(la::generate::coord_token_to_value(coord_start + 500, coord_start), 0.5);
}

TEST(Coord, TokenToValueEndpoints) {
    const std::int64_t coord_start = 151000;
    EXPECT_DOUBLE_EQ(la::generate::coord_token_to_value(coord_start + 0, coord_start), 0.0);
    EXPECT_DOUBLE_EQ(la::generate::coord_token_to_value(coord_start + 1000, coord_start), 1.0);
}

TEST(Coord, TokenToValueIndependentFormula) {
    const std::int64_t coord_start = 7;
    for (std::int64_t bin : {0, 1, 250, 333, 999, 1000}) {
        double expected = static_cast<double>(bin) / 1000.0;
        EXPECT_DOUBLE_EQ(la::generate::coord_token_to_value(coord_start + bin, coord_start),
                         expected)
            << "bin " << bin;
    }
}

TEST(Coord, ValueToPixelInverseRoundTrip) {
    // Known transform: original 640px resized by scale=0.5 -> 320px, padded to a
    // 336px canvas (top-left anchored, offset 0).
    ResizeTransform t;
    t.scale = 0.5;
    t.canvas_dim = 336.0;
    t.offset = 0.0;

    // Independent forward: a known original pixel -> normalized model value.
    //   canvas_px = orig_px * scale + offset
    //   value     = canvas_px / canvas_dim
    auto forward = [&](double orig_px) {
        double canvas_px = orig_px * t.scale + t.offset;
        return canvas_px / t.canvas_dim;
    };

    for (double orig_px : {0.0, 100.0, 320.0, 639.0}) {
        double v = forward(orig_px);
        double back = la::generate::value_to_pixel(v, t);
        EXPECT_NEAR(back, orig_px, 1e-6) << "orig_px " << orig_px;
    }
}

TEST(Coord, ValueToPixelWithOffset) {
    ResizeTransform t;
    t.scale = 2.0;
    t.canvas_dim = 1000.0;
    t.offset = 50.0;
    // value 0.3 -> canvas_px = 300 -> orig = (300 - 50) / 2 = 125
    EXPECT_NEAR(la::generate::value_to_pixel(0.3, t), 125.0, 1e-9);
}

TEST(Coord, TokenToPixelComposesValueAndInverse) {
    const std::int64_t coord_start = 1000;
    ResizeTransform t;
    t.scale = 0.5;
    t.canvas_dim = 800.0;
    t.offset = 0.0;
    const std::int64_t bin = 500;  // value 0.5 -> canvas_px 400 -> orig 800
    double via_compose = la::generate::coord_token_to_pixel(coord_start + bin, coord_start, t);
    double v = la::generate::coord_token_to_value(coord_start + bin, coord_start);
    double via_steps = la::generate::value_to_pixel(v, t);
    EXPECT_DOUBLE_EQ(via_compose, via_steps);
    EXPECT_NEAR(via_compose, 800.0, 1e-6);
}

// ---------------------------------------------------------------------------
// parse: id-native scanner
//
// We build the TokenIds via its Fallback() factory (the documented default
// instance) and then read its members for stream construction so the test does
// not hard-code any particular numeric layout. Helper boxes are assembled from
// the same struct under test, making the assertions self-consistent.
// ---------------------------------------------------------------------------

// Encode a per-mille bin as a coordinate token id.
std::int64_t coord_id(const TokenIds& tok, std::int64_t bin) {
    return tok.coord_start + bin;
}

TEST(Parse, RefLabelledBoxFourGroups) {
    TokenIds tok = TokenIds::Fallback();

    // <ref> car... </ref> <box> <x1=100> <y1=200> <x2=800> <y2=900> </box> <im_end>
    // The ref body ids are opaque to the parser (decoded by the tokenizer
    // later); we use two arbitrary BASE-vocab ids to stand in for "car". They
    // must NOT collide with any structural/coord id, so we use small ids well
    // below the special-token range (mirrors the sibling test's use of 42/43).
    const std::int64_t car_a = 42;
    const std::int64_t car_b = 43;

    std::vector<std::int64_t> ids = {
        tok.ref_start, car_a, car_b, tok.ref_end,
        tok.box_start,
        coord_id(tok, 100), coord_id(tok, 200),
        coord_id(tok, 800), coord_id(tok, 900),
        tok.box_end,
        tok.im_end,
    };

    std::vector<la::generate::RefLabel> labels;
    ParseResult res = la::generate::parse_token_stream(ids, tok, &labels);

    ASSERT_EQ(res.detections.size(), 1u);
    const Detection& d = res.detections[0];
    EXPECT_EQ(d.kind, DetectionKind::Box);
    ASSERT_EQ(d.coords_permille.size(), 4u);
    EXPECT_EQ(d.coords_permille[0], 100);
    EXPECT_EQ(d.coords_permille[1], 200);
    EXPECT_EQ(d.coords_permille[2], 800);
    EXPECT_EQ(d.coords_permille[3], 900);

    // The raw ref ids are captured parallel to the detection.
    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels[0].ids, (std::vector<std::int64_t>{car_a, car_b}));
}

TEST(Parse, TwoGroupFrameIsPoint) {
    TokenIds tok = TokenIds::Fallback();
    std::vector<std::int64_t> ids = {
        tok.box_start,
        coord_id(tok, 333), coord_id(tok, 666),
        tok.box_end,
        tok.im_end,
    };
    ParseResult res = la::generate::parse_token_stream(ids, tok, nullptr);
    ASSERT_EQ(res.detections.size(), 1u);
    EXPECT_EQ(res.detections[0].kind, DetectionKind::Point);
    ASSERT_EQ(res.detections[0].coords_permille.size(), 2u);
    EXPECT_EQ(res.detections[0].coords_permille[0], 333);
    EXPECT_EQ(res.detections[0].coords_permille[1], 666);
}

TEST(Parse, NoneBoxYieldsNoneDetection) {
    TokenIds tok = TokenIds::Fallback();
    // <box> none </box> : zero coordinate groups inside the wrapper.
    std::vector<std::int64_t> ids = {
        tok.box_start, tok.none, tok.box_end, tok.im_end,
    };
    ParseResult res = la::generate::parse_token_stream(ids, tok, nullptr);
    ASSERT_EQ(res.detections.size(), 1u);
    EXPECT_EQ(res.detections[0].kind, DetectionKind::None);
    EXPECT_TRUE(res.detections[0].coords_permille.empty());
}

TEST(Parse, MultipleDetections) {
    TokenIds tok = TokenIds::Fallback();
    std::vector<std::int64_t> ids = {
        // box 1 (4 groups)
        tok.box_start,
        coord_id(tok, 10), coord_id(tok, 20), coord_id(tok, 30), coord_id(tok, 40),
        tok.box_end,
        // box 2 (2 groups -> point)
        tok.box_start,
        coord_id(tok, 500), coord_id(tok, 500),
        tok.box_end,
        tok.im_end,
    };
    ParseResult res = la::generate::parse_token_stream(ids, tok, nullptr);
    ASSERT_EQ(res.detections.size(), 2u);
    EXPECT_EQ(res.detections[0].kind, DetectionKind::Box);
    EXPECT_EQ(res.detections[0].coords_permille,
              (std::vector<std::int64_t>{10, 20, 30, 40}));
    EXPECT_EQ(res.detections[1].kind, DetectionKind::Point);
    EXPECT_EQ(res.detections[1].coords_permille,
              (std::vector<std::int64_t>{500, 500}));
}

TEST(Parse, ScanningStopsAtImEnd) {
    TokenIds tok = TokenIds::Fallback();
    std::vector<std::int64_t> ids = {
        tok.box_start,
        coord_id(tok, 1), coord_id(tok, 2), coord_id(tok, 3), coord_id(tok, 4),
        tok.box_end,
        tok.im_end,
        // everything past im_end must be ignored
        tok.box_start, coord_id(tok, 9), coord_id(tok, 9), tok.box_end,
    };
    ParseResult res = la::generate::parse_token_stream(ids, tok, nullptr);
    ASSERT_EQ(res.detections.size(), 1u);
    EXPECT_EQ(res.detections[0].kind, DetectionKind::Box);
}

// ---------------------------------------------------------------------------
// parse: text fallback (golden worker-string parity)
// ---------------------------------------------------------------------------

TEST(ParseText, WorkerBoxStringFourGroups) {
    // Worker emits: <box><x1><y1><x2><y2></box> with per-mille integers.
    ParseResult res = la::generate::parse_text("<box><100><200><800><900></box>");
    ASSERT_EQ(res.detections.size(), 1u);
    EXPECT_EQ(res.detections[0].kind, DetectionKind::Box);
    EXPECT_EQ(res.detections[0].coords_permille,
              (std::vector<std::int64_t>{100, 200, 800, 900}));
}

TEST(ParseText, WorkerPointStringTwoGroups) {
    ParseResult res = la::generate::parse_text("<box><333><666></box>");
    ASSERT_EQ(res.detections.size(), 1u);
    EXPECT_EQ(res.detections[0].kind, DetectionKind::Point);
    EXPECT_EQ(res.detections[0].coords_permille,
              (std::vector<std::int64_t>{333, 666}));
}

TEST(ParseText, WorkerNoneString) {
    ParseResult res = la::generate::parse_text("<box>none</box>");
    ASSERT_EQ(res.detections.size(), 1u);
    EXPECT_EQ(res.detections[0].kind, DetectionKind::None);
    EXPECT_TRUE(res.detections[0].coords_permille.empty());
}

TEST(ParseText, RefLabelAttachesInTextFallback) {
    ParseResult res = la::generate::parse_text("<ref>car</ref><box><1><2><3><4></box>");
    ASSERT_EQ(res.detections.size(), 1u);
    EXPECT_EQ(res.detections[0].kind, DetectionKind::Box);
    ASSERT_TRUE(res.detections[0].label.has_value());
    EXPECT_EQ(res.detections[0].label.value(), "car");
    EXPECT_EQ(res.detections[0].coords_permille,
              (std::vector<std::int64_t>{1, 2, 3, 4}));
}

TEST(ParseText, IdNativeAndTextAgreeOnGroupCount) {
    // Internal-consistency: the same logical box decoded two ways must agree on
    // kind and on the per-mille coordinates.
    TokenIds tok = TokenIds::Fallback();
    std::vector<std::int64_t> ids = {
        tok.box_start,
        coord_id(tok, 11), coord_id(tok, 22), coord_id(tok, 33), coord_id(tok, 44),
        tok.box_end, tok.im_end,
    };
    ParseResult a = la::generate::parse_token_stream(ids, tok, nullptr);
    ParseResult b = la::generate::parse_text("<box><11><22><33><44></box>");
    ASSERT_EQ(a.detections.size(), 1u);
    ASSERT_EQ(b.detections.size(), 1u);
    EXPECT_EQ(a.detections[0].kind, b.detections[0].kind);
    EXPECT_EQ(a.detections[0].coords_permille, b.detections[0].coords_permille);
}

}  // namespace
