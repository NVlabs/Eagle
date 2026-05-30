// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
// Verbatim-string tests for la::api prompt builders. The expected strings are
// the EXACT output of Embodied/locateanything_worker.py and are load-bearing.

#include "la/api/prompts.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

using la::api::GuiOutputType;

// ---- detect: joins categories with "</c>" and uses "matches" ----

TEST(Prompts, DetectSingleCategory) {
  EXPECT_EQ(la::api::build_detect_prompt({"person"}),
            "Locate all the instances that matches the following description: "
            "person.");
}

TEST(Prompts, DetectMultipleCategoriesJoinedWithSeparator) {
  // Mirrors the worker example: detect(img, ["person", "car", "bicycle"]).
  EXPECT_EQ(la::api::build_detect_prompt({"person", "car", "bicycle"}),
            "Locate all the instances that matches the following description: "
            "person</c>car</c>bicycle.");
}

TEST(Prompts, DetectEmptyCategories) {
  EXPECT_EQ(la::api::build_detect_prompt({}),
            "Locate all the instances that matches the following description: .");
}

TEST(Prompts, DetectTwoCategoriesOnlyOneSeparator) {
  EXPECT_EQ(la::api::build_detect_prompt({"a", "b"}),
            "Locate all the instances that matches the following description: "
            "a</c>b.");
}

// ---- ground_single ----

TEST(Prompts, GroundSingle) {
  EXPECT_EQ(la::api::build_ground_single_prompt("the red cup"),
            "Locate a single instance that matches the following description: "
            "the red cup.");
}

// ---- ground_multi: uses "match" (contrast with detect's "matches") ----

TEST(Prompts, GroundMultiUsesMatchNotMatches) {
  EXPECT_EQ(la::api::build_ground_multi_prompt("people wearing red shirts"),
            "Locate all the instances that match the following description: "
            "people wearing red shirts.");
}

TEST(Prompts, DetectAndGroundMultiDifferOnVerb) {
  const std::string detect = la::api::build_detect_prompt({"x"});
  const std::string multi = la::api::build_ground_multi_prompt("x");
  // detect -> "matches", ground_multi -> "match": they must NOT be equal and
  // detect must not contain the bare "match the".
  EXPECT_NE(detect, multi);
  EXPECT_NE(detect.find("matches the following"), std::string::npos);
  EXPECT_NE(multi.find("match the following"), std::string::npos);
  EXPECT_EQ(multi.find("matches the following"), std::string::npos);
}

// ---- ground_text ----

TEST(Prompts, GroundText) {
  EXPECT_EQ(la::api::build_ground_text_prompt("the title at the top"),
            "Please locate the text referred as the title at the top.");
}

// ---- detect_text (no argument) ----

TEST(Prompts, DetectText) {
  EXPECT_EQ(la::api::build_detect_text_prompt(),
            "Detect all the text in box format.");
}

// ---- ground_gui (box vs point) ----

TEST(Prompts, GroundGuiBoxEnum) {
  EXPECT_EQ(la::api::build_ground_gui_prompt("the search button",
                                             GuiOutputType::kBox),
            "Locate the region that matches the following description: "
            "the search button.");
}

TEST(Prompts, GroundGuiPointEnumUsesPointTo) {
  EXPECT_EQ(la::api::build_ground_gui_prompt("the search button",
                                             GuiOutputType::kPoint),
            "Point to: the search button.");
}

TEST(Prompts, GroundGuiDefaultsToBox) {
  EXPECT_EQ(la::api::build_ground_gui_prompt("x"),
            "Locate the region that matches the following description: x.");
}

TEST(Prompts, GroundGuiStringPointSelectsPoint) {
  EXPECT_EQ(la::api::build_ground_gui_prompt("the search button", "point"),
            "Point to: the search button.");
}

TEST(Prompts, GroundGuiStringBoxSelectsBox) {
  EXPECT_EQ(la::api::build_ground_gui_prompt("x", "box"),
            "Locate the region that matches the following description: x.");
}

TEST(Prompts, GroundGuiStringUnknownFallsBackToBox) {
  // Worker: `if output_type == "point": ... else: ...` -- anything not "point"
  // is the box prompt.
  EXPECT_EQ(la::api::build_ground_gui_prompt("x", "anything-else"),
            "Locate the region that matches the following description: x.");
}

// ---- point ----

TEST(Prompts, Point) {
  EXPECT_EQ(la::api::build_point_prompt("the traffic light"),
            "Point to: the traffic light.");
}

TEST(Prompts, PointAndGuiPointProduceSameWording) {
  EXPECT_EQ(la::api::build_point_prompt("q"),
            la::api::build_ground_gui_prompt("q", GuiOutputType::kPoint));
}

}  // namespace
