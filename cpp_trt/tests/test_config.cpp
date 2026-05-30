// Unit tests for la_config (TokenIds + JSON loader).
//
// Independent expectations: the coord-band invariant (1001 ids inclusive),
// bin<->id arithmetic recomputed in the test, and a JSON fixture whose values
// are asserted field-by-field.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "la/config/token_ids.hpp"

namespace {

using la::config::TokenIds;
using la::config::token_id_t;
using la::config::kUnsetTokenId;
using la::config::kCoordBandSpan;
using la::config::LoadTokenIds;

// Build a minimal, internally-consistent TokenIds with a valid coord band.
TokenIds MakeValid() {
  TokenIds t;
  t.image_token = 100;
  t.box_start = 200;
  t.box_end = 201;
  t.ref_start = 202;
  t.ref_end = 203;
  t.mask = 204;
  t.coord_start = 1000;
  t.coord_end = 1000 + kCoordBandSpan;  // contiguous 1001-id band
  t.none = 4064;
  t.null_token = 9000;
  t.im_end = 9001;
  t.switch_token = 9002;
  return t;
}

}  // namespace

TEST(TokenIds, ValidPassesValidate) {
  TokenIds t = MakeValid();
  EXPECT_NO_THROW(t.Validate(/*require_all=*/false));
  EXPECT_NO_THROW(t.Validate(/*require_all=*/true));
  EXPECT_TRUE(t.IsComplete());
}

// Validate() must throw when the coord band is not exactly 1000 wide.
TEST(TokenIds, ValidateThrowsOnWrongCoordBand) {
  {
    TokenIds t = MakeValid();
    t.coord_end = t.coord_start + 999;  // 1000 ids -> wrong
    EXPECT_THROW(t.Validate(false), std::runtime_error);
  }
  {
    TokenIds t = MakeValid();
    t.coord_end = t.coord_start + 1001;  // 1002 ids -> wrong
    EXPECT_THROW(t.Validate(false), std::runtime_error);
  }
  {
    TokenIds t = MakeValid();
    t.coord_end = t.coord_start + kCoordBandSpan;  // exactly right
    EXPECT_NO_THROW(t.Validate(false));
  }
}

TEST(TokenIds, ValidateThrowsOnNegativeCoordStart) {
  TokenIds t = MakeValid();
  t.coord_start = -1;
  t.coord_end = t.coord_start + kCoordBandSpan;
  EXPECT_THROW(t.Validate(false), std::runtime_error);
}

TEST(TokenIds, ValidateThrowsOnUnsetCoordWhenRequired) {
  TokenIds t;  // all unset
  EXPECT_THROW(t.Validate(false), std::runtime_error);  // coord_start/end unset
}

// IsCoordToken / CoordBin / CoordTokenForBin: independent arithmetic.
TEST(TokenIds, CoordBandArithmetic) {
  TokenIds t = MakeValid();  // coord_start=1000, coord_end=2000
  EXPECT_TRUE(t.IsCoordToken(t.coord_start));
  EXPECT_TRUE(t.IsCoordToken(t.coord_end));
  EXPECT_TRUE(t.IsCoordToken(1500));
  EXPECT_FALSE(t.IsCoordToken(t.coord_start - 1));
  EXPECT_FALSE(t.IsCoordToken(t.coord_end + 1));

  // bin = id - coord_start, in [0,1000]
  EXPECT_EQ(t.CoordBin(t.coord_start), 0);
  EXPECT_EQ(t.CoordBin(t.coord_start + 1), 1);
  EXPECT_EQ(t.CoordBin(t.coord_end), kCoordBandSpan);  // 1000
  EXPECT_EQ(t.CoordBin(t.coord_start - 1), -1);        // out of band

  // id = coord_start + bin
  EXPECT_EQ(t.CoordTokenForBin(0), t.coord_start);
  EXPECT_EQ(t.CoordTokenForBin(kCoordBandSpan), t.coord_end);
  EXPECT_EQ(t.CoordTokenForBin(kCoordBandSpan + 1), kUnsetTokenId);  // out of range
  EXPECT_EQ(t.CoordTokenForBin(-1), kUnsetTokenId);

  // round trip across the whole band
  for (token_id_t bin = 0; bin <= kCoordBandSpan; ++bin) {
    token_id_t id = t.CoordTokenForBin(bin);
    EXPECT_EQ(t.CoordBin(id), bin) << "bin=" << bin;
  }
}

// JSON loader parses a flat fixture; values asserted field-by-field.
TEST(LoadTokenIds, ParsesFlatFixture) {
  const std::string json = R"({
    "image_token_index": 151667,
    "box_start": 151668,
    "box_end": 151669,
    "ref_start": 151672,
    "ref_end": 151673,
    "mask": 151676,
    "coord_start": 151677,
    "coord_end": 152677,
    "none": 4064,
    "null": 152678,
    "im_end": 151645,
    "switch_token": 152679
  })";
  TokenIds t = LoadTokenIds(json, /*require_all=*/false);
  EXPECT_EQ(t.image_token, 151667);
  EXPECT_EQ(t.box_start, 151668);
  EXPECT_EQ(t.box_end, 151669);
  EXPECT_EQ(t.ref_start, 151672);
  EXPECT_EQ(t.ref_end, 151673);
  EXPECT_EQ(t.mask, 151676);
  EXPECT_EQ(t.coord_start, 151677);
  EXPECT_EQ(t.coord_end, 152677);
  EXPECT_EQ(t.none, 4064);
  EXPECT_EQ(t.null_token, 152678);
  EXPECT_EQ(t.im_end, 151645);
  EXPECT_EQ(t.switch_token, 152679);

  // independent coord-band check on the parsed values
  EXPECT_EQ(t.coord_end - t.coord_start, kCoordBandSpan);
  EXPECT_NO_THROW(t.Validate(true));
}

// Unknown keys ignored; missing keys remain kUnsetTokenId.
TEST(LoadTokenIds, UnknownKeysIgnoredMissingUnset) {
  const std::string json = R"({
    "box_start": 5,
    "box_end": 6,
    "totally_unknown_key": 999
  })";
  TokenIds t = LoadTokenIds(json, /*require_all=*/false);
  EXPECT_EQ(t.box_start, 5);
  EXPECT_EQ(t.box_end, 6);
  EXPECT_EQ(t.image_token, kUnsetTokenId);
  EXPECT_EQ(t.coord_start, kUnsetTokenId);
}

// Malformed JSON throws.
TEST(LoadTokenIds, MalformedThrows) {
  EXPECT_THROW(LoadTokenIds("{ not json", false), std::exception);
}

// require_all on an incomplete fixture throws via Validate(true).
TEST(LoadTokenIds, RequireAllThrowsOnIncomplete) {
  const std::string json = R"({ "box_start": 1, "box_end": 2 })";
  EXPECT_THROW(LoadTokenIds(json, /*require_all=*/true), std::runtime_error);
}
