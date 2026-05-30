#include "la/decode/position_ids.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

using la::decode::build_position_ids_ar;
using la::decode::build_position_ids_mtp;

namespace {

TEST(PosIdsMtp, BasicSubtractLastNFuture) {
  // arange(0,10) then last 6 minus 1.
  auto ids = build_position_ids_mtp(0, 10, 6);
  std::vector<std::int64_t> expected = {0, 1, 2, 3, 3, 4, 5, 6, 7, 8};
  EXPECT_EQ(ids, expected);
}

TEST(PosIdsMtp, NonZeroStart) {
  // arange(100,108) = [100..107]; last 6 minus 1.
  auto ids = build_position_ids_mtp(100, 108, 6);
  std::vector<std::int64_t> expected = {100, 101, 101, 102, 103, 104, 105, 106};
  EXPECT_EQ(ids, expected);
}

TEST(PosIdsMtp, NFutureZeroIsPlainArange) {
  auto ids = build_position_ids_mtp(5, 11, 0);
  std::vector<std::int64_t> expected = {5, 6, 7, 8, 9, 10};
  EXPECT_EQ(ids, expected);
}

TEST(PosIdsMtp, NFutureEqualsLength) {
  auto ids = build_position_ids_mtp(0, 6, 6);
  std::vector<std::int64_t> expected = {-1, 0, 1, 2, 3, 4};
  EXPECT_EQ(ids, expected);
}

TEST(PosIdsMtp, LengthIsTotalMinusStart) {
  auto ids = build_position_ids_mtp(3, 20, 6);
  EXPECT_EQ(ids.size(), 17u);
}

TEST(PosIdsAr, PlainArange) {
  auto ids = build_position_ids_ar(0, 5);
  std::vector<std::int64_t> expected = {0, 1, 2, 3, 4};
  EXPECT_EQ(ids, expected);
}

TEST(PosIdsAr, SingleStep) {
  auto ids = build_position_ids_ar(42, 1);
  std::vector<std::int64_t> expected = {42};
  EXPECT_EQ(ids, expected);
}

TEST(PosIdsAr, ZeroLen) {
  auto ids = build_position_ids_ar(7, 0);
  EXPECT_TRUE(ids.empty());
}

}  // namespace
