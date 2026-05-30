// SPDX-License-Identifier: Apache-2.0

#include "la/generate/coord.hpp"

#include <gtest/gtest.h>

#include "la/config/token_ids.hpp"

namespace {

using la::generate::coord_token_to_pixel;
using la::generate::coord_token_to_value;
using la::generate::ResizeTransform;
using la::generate::value_to_pixel;

TEST(Coord, TokenToValueIsPerMille) {
    const std::int64_t cs = 151677;
    EXPECT_DOUBLE_EQ(coord_token_to_value(cs, cs), 0.0);
    EXPECT_DOUBLE_EQ(coord_token_to_value(cs + 500, cs), 0.5);
    EXPECT_DOUBLE_EQ(coord_token_to_value(cs + 1000, cs), 1.0);
    EXPECT_DOUBLE_EQ(coord_token_to_value(cs + 123, cs), 0.123);
}

TEST(Coord, CoordBandViaConfigTokenIds) {
    // Cooperate with la::config::TokenIds (the canonical id struct).
    la::config::TokenIds t = la::config::TokenIds::Fallback();
    EXPECT_EQ(t.coord_end - t.coord_start, 1000);
    EXPECT_TRUE(t.IsCoordToken(t.coord_start));
    EXPECT_TRUE(t.IsCoordToken(t.coord_end));
    EXPECT_FALSE(t.IsCoordToken(t.coord_start - 1));
    EXPECT_FALSE(t.IsCoordToken(t.coord_end + 1));
    // CoordBin == (id - coord_start), the per-mille value coord.hpp divides.
    EXPECT_EQ(t.CoordBin(t.coord_start + 250), 250);
    EXPECT_DOUBLE_EQ(coord_token_to_value(t.coord_start + 250, t.coord_start),
                     0.25);
}

TEST(Coord, ValueToPixelUndoesScale) {
    // canvas 280px, image resized 2x (orig 140 -> 280). value 0.5 -> canvas
    // 140px -> orig 70px.
    ResizeTransform t;
    t.scale = 2.0;
    t.canvas_dim = 280.0;
    t.offset = 0.0;
    EXPECT_DOUBLE_EQ(value_to_pixel(0.5, t), 70.0);
    EXPECT_DOUBLE_EQ(value_to_pixel(1.0, t), 140.0);
    EXPECT_DOUBLE_EQ(value_to_pixel(0.0, t), 0.0);
}

TEST(Coord, ValueToPixelWithOffset) {
    ResizeTransform t;
    t.scale = 1.0;
    t.canvas_dim = 200.0;
    t.offset = 20.0;  // 20px top-left pad
    EXPECT_DOUBLE_EQ(value_to_pixel(0.5, t), 80.0);  // 100 - 20
}

TEST(Coord, TokenToPixelComposed) {
    const std::int64_t cs = 151677;
    ResizeTransform t;
    t.scale = 2.0;
    t.canvas_dim = 280.0;
    EXPECT_DOUBLE_EQ(coord_token_to_pixel(cs + 500, cs, t), 70.0);
}

}  // namespace
