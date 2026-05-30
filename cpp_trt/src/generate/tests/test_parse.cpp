// SPDX-License-Identifier: Apache-2.0

#include "la/generate/parse.hpp"

#include <gtest/gtest.h>

#include <vector>

#include "la/config/token_ids.hpp"

namespace {

using la::config::TokenIds;
using la::generate::DetectionKind;
using la::generate::parse_text;
using la::generate::parse_token_stream;
using la::generate::ParseResult;
using la::generate::RefLabel;

// Coord token id from a per-mille value, using the canonical TokenIds band.
std::int64_t C(const TokenIds& t, std::int64_t permille) {
    return t.coord_start + permille;
}

TEST(Parse, IdStreamBoxFourCoords) {
    TokenIds t = TokenIds::Fallback();
    std::vector<std::int64_t> ids = {
        t.box_start, C(t, 100), C(t, 200), C(t, 300), C(t, 400), t.box_end};
    ParseResult r = parse_token_stream(ids, t);
    ASSERT_EQ(r.detections.size(), 1u);
    EXPECT_EQ(r.detections[0].kind, DetectionKind::Box);
    EXPECT_EQ(r.detections[0].coords_permille,
              (std::vector<std::int64_t>{100, 200, 300, 400}));
}

TEST(Parse, IdStreamPointTwoCoords) {
    TokenIds t = TokenIds::Fallback();
    std::vector<std::int64_t> ids = {t.box_start, C(t, 600), C(t, 700),
                                     t.box_end};
    ParseResult r = parse_token_stream(ids, t);
    ASSERT_EQ(r.detections.size(), 1u);
    EXPECT_EQ(r.detections[0].kind, DetectionKind::Point);
    EXPECT_EQ(r.detections[0].coords_permille,
              (std::vector<std::int64_t>{600, 700}));
}

TEST(Parse, IdStreamBoxNone) {
    TokenIds t = TokenIds::Fallback();
    std::vector<std::int64_t> ids = {t.box_start, t.none, t.box_end};
    ParseResult r = parse_token_stream(ids, t);
    ASSERT_EQ(r.detections.size(), 1u);
    EXPECT_EQ(r.detections[0].kind, DetectionKind::None);
    EXPECT_TRUE(r.detections[0].coords_permille.empty());
}

TEST(Parse, IdStreamRefAttachesAsLabel) {
    TokenIds t = TokenIds::Fallback();
    // <ref> 42 43 </ref> <box> ... </box>
    std::vector<std::int64_t> ids = {
        t.ref_start, 42, 43, t.ref_end,
        t.box_start, C(t, 10), C(t, 20), C(t, 30), C(t, 40), t.box_end};
    std::vector<RefLabel> labels;
    ParseResult r = parse_token_stream(ids, t, &labels);
    ASSERT_EQ(r.detections.size(), 1u);
    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels[0].ids, (std::vector<std::int64_t>{42, 43}));
    EXPECT_EQ(r.detections[0].kind, DetectionKind::Box);
}

TEST(Parse, IdStreamMultipleDetections) {
    TokenIds t = TokenIds::Fallback();
    std::vector<std::int64_t> ids = {
        t.box_start, C(t, 1), C(t, 2), C(t, 3), C(t, 4), t.box_end,
        t.box_start, C(t, 5), C(t, 6), t.box_end,
        t.im_end};
    ParseResult r = parse_token_stream(ids, t);
    ASSERT_EQ(r.detections.size(), 2u);
    EXPECT_EQ(r.detections[0].kind, DetectionKind::Box);
    EXPECT_EQ(r.detections[1].kind, DetectionKind::Point);
}

TEST(Parse, IdStreamStopsAtImEnd) {
    TokenIds t = TokenIds::Fallback();
    std::vector<std::int64_t> ids = {t.im_end, t.box_start, C(t, 1), C(t, 2),
                                     C(t, 3), C(t, 4), t.box_end};
    ParseResult r = parse_token_stream(ids, t);
    EXPECT_TRUE(r.detections.empty());
}

TEST(Parse, IdStreamUncertainSentinel) {
    TokenIds t = TokenIds::Fallback();
    // A box with one uncertain (id 0) coord slot still counts as a 4-group box.
    std::vector<std::int64_t> ids = {t.box_start, C(t, 10), 0, C(t, 30),
                                     C(t, 40), t.box_end};
    ParseResult r = parse_token_stream(ids, t);
    ASSERT_EQ(r.detections.size(), 1u);
    EXPECT_EQ(r.detections[0].kind, DetectionKind::Box);
    EXPECT_EQ(r.detections[0].coords_permille,
              (std::vector<std::int64_t>{10, 0, 30, 40}));
}

// ---- Text fallback parser (worker regex parity) ----

TEST(ParseText, FourCoordBox) {
    ParseResult r = parse_text("<box><100><200><300><400></box>");
    ASSERT_EQ(r.detections.size(), 1u);
    EXPECT_EQ(r.detections[0].kind, DetectionKind::Box);
    EXPECT_EQ(r.detections[0].coords_permille,
              (std::vector<std::int64_t>{100, 200, 300, 400}));
}

TEST(ParseText, TwoCoordPoint) {
    ParseResult r = parse_text("<box><500><600></box>");
    ASSERT_EQ(r.detections.size(), 1u);
    EXPECT_EQ(r.detections[0].kind, DetectionKind::Point);
}

TEST(ParseText, NoneBox) {
    ParseResult r = parse_text("<box>none</box>");
    ASSERT_EQ(r.detections.size(), 1u);
    EXPECT_EQ(r.detections[0].kind, DetectionKind::None);
}

TEST(ParseText, RefLabelAttached) {
    ParseResult r = parse_text("<ref>cat</ref><box><1><2><3><4></box>");
    ASSERT_EQ(r.detections.size(), 1u);
    ASSERT_TRUE(r.detections[0].label.has_value());
    EXPECT_EQ(*r.detections[0].label, "cat");
}

}  // namespace
