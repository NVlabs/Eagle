// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
// Tests for the TRT-free surface of la::api::LocateAnythingWorker:
//   * parse_boxes / parse_points (pure C++ ports of the worker regex parsers)
//   * predict() throwing the TRT-free stub
//   * GenerationParams defaults mirroring the worker

#include "la/api/worker.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace {

using la::api::Box;
using la::api::GenerationMode;
using la::api::GenerationParams;
using la::api::ImageRGB;
using la::api::LocateAnythingWorker;
using la::api::Point;

// ---- parse_boxes: r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>" ----

TEST(ParseBoxes, SingleBoxScaledToPixels) {
  // 1000-scale: 500/1000*200=100, etc.
  const std::vector<Box> boxes =
      LocateAnythingWorker::parse_boxes("<box><500><250><750><500></box>",
                                        /*w=*/200, /*h=*/400);
  ASSERT_EQ(boxes.size(), 1u);
  EXPECT_DOUBLE_EQ(boxes[0].x1, 100.0);
  EXPECT_DOUBLE_EQ(boxes[0].y1, 100.0);
  EXPECT_DOUBLE_EQ(boxes[0].x2, 150.0);
  EXPECT_DOUBLE_EQ(boxes[0].y2, 200.0);
}

TEST(ParseBoxes, MultipleBoxesWithSurroundingText) {
  const std::string answer =
      "here: <box><0><0><1000><1000></box> and <box><100><200><300><400></box>!";
  const std::vector<Box> boxes =
      LocateAnythingWorker::parse_boxes(answer, /*w=*/1000, /*h=*/1000);
  ASSERT_EQ(boxes.size(), 2u);
  EXPECT_DOUBLE_EQ(boxes[0].x1, 0.0);
  EXPECT_DOUBLE_EQ(boxes[0].x2, 1000.0);
  EXPECT_DOUBLE_EQ(boxes[1].x1, 100.0);
  EXPECT_DOUBLE_EQ(boxes[1].y1, 200.0);
  EXPECT_DOUBLE_EQ(boxes[1].x2, 300.0);
  EXPECT_DOUBLE_EQ(boxes[1].y2, 400.0);
}

TEST(ParseBoxes, NoMatchReturnsEmpty) {
  EXPECT_TRUE(LocateAnythingWorker::parse_boxes("no boxes here", 100, 100).empty());
  // A 2-int (point) wrapper must NOT match the 4-int box parser.
  EXPECT_TRUE(
      LocateAnythingWorker::parse_boxes("<box><10><20></box>", 100, 100).empty());
}

TEST(ParseBoxes, NoneBoxIsNotParsedAsCoordinates) {
  // <box>none</box> has no integer groups -> no boxes.
  EXPECT_TRUE(
      LocateAnythingWorker::parse_boxes("<box>none</box>", 100, 100).empty());
}

// ---- parse_points: r"<box><(\d+)><(\d+)></box>" ----

TEST(ParsePoints, SinglePointScaledToPixels) {
  const std::vector<Point> pts =
      LocateAnythingWorker::parse_points("<box><500><250></box>", 200, 400);
  ASSERT_EQ(pts.size(), 1u);
  EXPECT_DOUBLE_EQ(pts[0].x, 100.0);
  EXPECT_DOUBLE_EQ(pts[0].y, 100.0);
}

TEST(ParsePoints, MultiplePoints) {
  const std::vector<Point> pts = LocateAnythingWorker::parse_points(
      "<box><0><0></box><box><1000><1000></box>", 50, 80);
  ASSERT_EQ(pts.size(), 2u);
  EXPECT_DOUBLE_EQ(pts[0].x, 0.0);
  EXPECT_DOUBLE_EQ(pts[0].y, 0.0);
  EXPECT_DOUBLE_EQ(pts[1].x, 50.0);
  EXPECT_DOUBLE_EQ(pts[1].y, 80.0);
}

// ---- GenerationParams defaults mirror the worker ----

TEST(GenerationParams, DefaultsMatchWorker) {
  GenerationParams p;
  EXPECT_EQ(p.mode, GenerationMode::kHybrid);
  EXPECT_EQ(p.max_new_tokens, 2048);
  EXPECT_TRUE(p.do_sample);
  EXPECT_FLOAT_EQ(p.temperature, 0.7f);
  EXPECT_FLOAT_EQ(p.top_p, 0.9f);
  EXPECT_FLOAT_EQ(p.repetition_penalty, 1.1f);
  EXPECT_TRUE(p.verbose);
}

// ---- predict() stub in the TRT-free build ----

#if !defined(LA_BUILD_TRT)
TEST(Worker, PredictThrowsWithoutTrt) {
  LocateAnythingWorker w("/nonexistent/model");
  ImageRGB img;  // empty view is fine -- predict throws before touching it.
  EXPECT_THROW(w.predict(img, "Detect all the text in box format."),
               std::runtime_error);
}

TEST(Worker, ConvenienceMethodsThrowWithoutTrt) {
  LocateAnythingWorker w("/nonexistent/model");
  ImageRGB img;
  EXPECT_THROW(w.detect(img, {"person"}), std::runtime_error);
  EXPECT_THROW(w.point(img, "the light"), std::runtime_error);
  EXPECT_THROW(w.detect_text(img), std::runtime_error);
}
#endif  // !LA_BUILD_TRT

}  // namespace
