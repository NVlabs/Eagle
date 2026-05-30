// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// Unit tests for the la_api PURE-C++ surface (Phase 2):
//   * prompts.h : load-bearing prompt strings (build_detect_prompt, ground_*,
//                 point, ground_gui) compared against literal expected strings
//                 copied verbatim from the PyTorch worker
//                 (Embodied/locateanything_worker.py).
//   * worker.h  : parse_boxes / parse_points (pure C++, always available) and
//                 the GenerationParams worker defaults.
//
// The expected prompt strings below are the INDEPENDENT reference: they are the
// exact wording the reference worker emits, written out as literals here (not
// derived from the implementation). The load-bearing distinctions tested are:
//   detect()       uses "matches"  + "</c>" category separator
//   ground_multi() uses "match"    (NOT "matches")
//   point() / ground_gui(point)    use the "Point to:" prefix

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "la/api/prompts.h"
#include "la/api/worker.h"

namespace {

using ::la::api::GenerationMode;
using ::la::api::GenerationParams;
using ::la::api::GuiOutputType;
using ::la::api::LocateAnythingWorker;

// ---------------------------------------------------------------------------
// prompts: exact worker strings
// ---------------------------------------------------------------------------

TEST(Prompts, DetectJoinsCategoriesWithSlashCAndUsesMatches) {
    // Worker: "Locate all the instances that matches the following description: "
    //          + "</c>".join(categories) + "."
    std::string got = la::api::build_detect_prompt({"person", "car"});
    EXPECT_EQ(got,
              "Locate all the instances that matches the following description: "
              "person</c>car.");
}

TEST(Prompts, DetectSingleCategory) {
    std::string got = la::api::build_detect_prompt({"dog"});
    EXPECT_EQ(got,
              "Locate all the instances that matches the following description: dog.");
}

TEST(Prompts, GroundSingleUsesMatches) {
    std::string got = la::api::build_ground_single_prompt("the red apple");
    EXPECT_EQ(got,
              "Locate a single instance that matches the following description: "
              "the red apple.");
}

TEST(Prompts, GroundMultiUsesMatchNotMatches) {
    std::string got = la::api::build_ground_multi_prompt("the cars");
    EXPECT_EQ(got,
              "Locate all the instances that match the following description: "
              "the cars.");
    // Load-bearing: ground_multi uses "match", NOT "matches".
    EXPECT_EQ(got.find("matches"), std::string::npos);
    EXPECT_NE(got.find(" match the following"), std::string::npos);
}

TEST(Prompts, DetectAndGroundMultiDifferOnVerb) {
    std::string detect = la::api::build_detect_prompt({"x"});
    std::string multi = la::api::build_ground_multi_prompt("x");
    EXPECT_NE(detect.find(" matches "), std::string::npos);
    EXPECT_NE(multi.find(" match "), std::string::npos);
    EXPECT_EQ(multi.find(" matches "), std::string::npos);
    EXPECT_NE(detect, multi);
}

TEST(Prompts, GroundText) {
    std::string got = la::api::build_ground_text_prompt("the title");
    EXPECT_EQ(got, "Please locate the text referred as the title.");
}

TEST(Prompts, DetectText) {
    EXPECT_EQ(la::api::build_detect_text_prompt(), "Detect all the text in box format.");
}

TEST(Prompts, PointUsesPointToPrefix) {
    std::string got = la::api::build_point_prompt("the button");
    EXPECT_EQ(got, "Point to: the button.");
}

TEST(Prompts, GroundGuiBoxEnum) {
    std::string got = la::api::build_ground_gui_prompt("the menu", GuiOutputType::kBox);
    EXPECT_EQ(got,
              "Locate the region that matches the following description: the menu.");
}

TEST(Prompts, GroundGuiPointEnumUsesPointTo) {
    std::string got = la::api::build_ground_gui_prompt("the menu", GuiOutputType::kPoint);
    EXPECT_EQ(got, "Point to: the menu.");
}

TEST(Prompts, GroundGuiDefaultsToBox) {
    // Default argument is kBox.
    std::string got = la::api::build_ground_gui_prompt("the menu");
    EXPECT_EQ(got,
              "Locate the region that matches the following description: the menu.");
}

TEST(Prompts, GroundGuiStringOverloadPointSelectsPoint) {
    EXPECT_EQ(la::api::build_ground_gui_prompt("q", std::string("point")),
              "Point to: q.");
}

TEST(Prompts, GroundGuiStringOverloadNonPointSelectsBox) {
    // Worker: anything other than "point" maps to the box prompt.
    EXPECT_EQ(la::api::build_ground_gui_prompt("q", std::string("box")),
              "Locate the region that matches the following description: q.");
    EXPECT_EQ(la::api::build_ground_gui_prompt("q", std::string("anything-else")),
              "Locate the region that matches the following description: q.");
}

// ---------------------------------------------------------------------------
// worker: GenerationParams defaults
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// worker: parse_boxes / parse_points (pure C++, always available)
//
// Independent reference: per-mille integer / 1000 * dim, group order xyxy.
// ---------------------------------------------------------------------------

TEST(ParseBoxes, FourGroupBoxToPixels) {
    const int W = 640;
    const int H = 480;
    auto boxes = LocateAnythingWorker::parse_boxes("<box><100><200><800><900></box>", W, H);
    ASSERT_EQ(boxes.size(), 1u);
    // Reference: x = permille/1000 * W ; y = permille/1000 * H.
    EXPECT_NEAR(boxes[0].x1, 100.0 / 1000.0 * W, 1e-6);
    EXPECT_NEAR(boxes[0].y1, 200.0 / 1000.0 * H, 1e-6);
    EXPECT_NEAR(boxes[0].x2, 800.0 / 1000.0 * W, 1e-6);
    EXPECT_NEAR(boxes[0].y2, 900.0 / 1000.0 * H, 1e-6);
}

TEST(ParseBoxes, MultipleBoxes) {
    auto boxes = LocateAnythingWorker::parse_boxes(
        "<box><0><0><500><500></box><box><500><500><1000><1000></box>", 1000, 1000);
    ASSERT_EQ(boxes.size(), 2u);
    EXPECT_NEAR(boxes[0].x2, 500.0, 1e-6);
    EXPECT_NEAR(boxes[1].x1, 500.0, 1e-6);
    EXPECT_NEAR(boxes[1].x2, 1000.0, 1e-6);
}

TEST(ParseBoxes, NoneStringYieldsNoBoxes) {
    auto boxes = LocateAnythingWorker::parse_boxes("<box>none</box>", 640, 480);
    EXPECT_TRUE(boxes.empty());
}

TEST(ParseBoxes, TwoGroupPointDoesNotMatchBoxParser) {
    // A 2-int point wrapper must not be parsed as a 4-int box.
    auto boxes = LocateAnythingWorker::parse_boxes("<box><100><200></box>", 640, 480);
    EXPECT_TRUE(boxes.empty());
}

TEST(ParsePoints, TwoGroupPointToPixels) {
    const int W = 800;
    const int H = 600;
    auto points = LocateAnythingWorker::parse_points("<box><250><500></box>", W, H);
    ASSERT_EQ(points.size(), 1u);
    EXPECT_NEAR(points[0].x, 250.0 / 1000.0 * W, 1e-6);
    EXPECT_NEAR(points[0].y, 500.0 / 1000.0 * H, 1e-6);
}

TEST(ParsePoints, FourGroupBoxDoesNotMatchPointParser) {
    auto points = LocateAnythingWorker::parse_points("<box><1><2><3><4></box>", 640, 480);
    EXPECT_TRUE(points.empty());
}

TEST(ParsePoints, NoneStringYieldsNoPoints) {
    auto points = LocateAnythingWorker::parse_points("<box>none</box>", 640, 480);
    EXPECT_TRUE(points.empty());
}

}  // namespace
