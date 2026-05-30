// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
// la/api/prompts.cpp -- verbatim port of LocateAnythingWorker prompt strings.

#include "la/api/prompts.h"

namespace la {
namespace api {

namespace {

// Reproduces Python's '"</c>".join(categories)'.
std::string join_categories(const std::vector<std::string>& categories) {
  std::string out;
  for (std::size_t i = 0; i < categories.size(); ++i) {
    if (i != 0) {
      out += "</c>";
    }
    out += categories[i];
  }
  return out;
}

}  // namespace

std::string build_detect_prompt(const std::vector<std::string>& categories) {
  // worker.detect():
  //   cats = "</c>".join(categories)
  //   f"Locate all the instances that matches the following description: {cats}."
  return "Locate all the instances that matches the following description: " +
         join_categories(categories) + ".";
}

std::string build_ground_single_prompt(const std::string& phrase) {
  // worker.ground_single():
  //   f"Locate a single instance that matches the following description: {phrase}."
  return "Locate a single instance that matches the following description: " +
         phrase + ".";
}

std::string build_ground_multi_prompt(const std::string& phrase) {
  // worker.ground_multi():
  //   f"Locate all the instances that match the following description: {phrase}."
  return "Locate all the instances that match the following description: " +
         phrase + ".";
}

std::string build_ground_text_prompt(const std::string& phrase) {
  // worker.ground_text():
  //   f"Please locate the text referred as {phrase}."
  return "Please locate the text referred as " + phrase + ".";
}

std::string build_detect_text_prompt() {
  // worker.detect_text():
  //   "Detect all the text in box format."
  return "Detect all the text in box format.";
}

std::string build_ground_gui_prompt(const std::string& query,
                                    GuiOutputType output_type) {
  // worker.ground_gui():
  //   if output_type == "point": f"Point to: {phrase}."
  //   else:                      f"Locate the region that matches the following description: {phrase}."
  if (output_type == GuiOutputType::kPoint) {
    return "Point to: " + query + ".";
  }
  return "Locate the region that matches the following description: " + query +
         ".";
}

std::string build_ground_gui_prompt(const std::string& query,
                                    const std::string& output_type) {
  // Mirror the worker's exact branch: only the literal "point" selects the
  // point prompt; everything else (including the default "box") is the box
  // prompt.
  return build_ground_gui_prompt(
      query,
      output_type == "point" ? GuiOutputType::kPoint : GuiOutputType::kBox);
}

std::string build_point_prompt(const std::string& query) {
  // worker.point():
  //   f"Point to: {phrase}."
  return "Point to: " + query + ".";
}

}  // namespace api
}  // namespace la
