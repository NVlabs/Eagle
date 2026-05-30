// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
// la/api/prompts.h
//
// Pure-C++ user-prompt builders for the LocateAnything task suite.
//
// These reproduce -- VERBATIM -- the prompt wording emitted by the PyTorch
// reference worker (Embodied/locateanything_worker.py). The exact wording is
// LOAD-BEARING: the model was trained on these exact strings and any deviation
// (e.g. "matches" vs "match", the "</c>" category separator, the "Point to:"
// pointing prefix) silently degrades or changes model behaviour.
//
// This header is intentionally TRT-free / dependency-free: it has no TensorRT,
// TRT-LLM, PyTorch, CUDA, or sibling-module dependency, so it always compiles
// and is unit-tested in the default build.
//
// Reference mapping (worker method -> builder):
//   detect(categories)          -> build_detect_prompt(categories)
//   ground_single(phrase)       -> build_ground_single_prompt(phrase)
//   ground_multi(phrase)        -> build_ground_multi_prompt(phrase)
//   ground_text(phrase)         -> build_ground_text_prompt(phrase)
//   detect_text()               -> build_detect_text_prompt()
//   ground_gui(phrase, type)    -> build_ground_gui_prompt(query, output_type)
//   point(phrase)               -> build_point_prompt(query)

#ifndef LA_API_PROMPTS_H
#define LA_API_PROMPTS_H

#include <string>
#include <vector>

namespace la {
namespace api {

// Output-type selector for GUI grounding (ground_gui). Mirrors the worker's
// string `output_type` argument ("box" | "point").
enum class GuiOutputType {
  kBox,    // -> "Locate the region that matches the following description: ..."
  kPoint,  // -> "Point to: ..."
};

// Object detection / document layout analysis.
//
// Verbatim worker logic:
//   cats = "</c>".join(categories)
//   "Locate all the instances that matches the following description: {cats}."
//
// NOTE the singular-looking verb "matches" is INTENTIONAL and load-bearing --
// detect() uses "matches" while ground_multi() uses "match". Do not "fix" it.
std::string build_detect_prompt(const std::vector<std::string>& categories);

// Phrase grounding -- single instance.
//   "Locate a single instance that matches the following description: {phrase}."
std::string build_ground_single_prompt(const std::string& phrase);

// Phrase grounding -- multiple instances.
//   "Locate all the instances that match the following description: {phrase}."
// (uses "match", contrast with build_detect_prompt's "matches").
std::string build_ground_multi_prompt(const std::string& phrase);

// Text grounding.
//   "Please locate the text referred as {phrase}."
std::string build_ground_text_prompt(const std::string& phrase);

// Scene text detection (no argument).
//   "Detect all the text in box format."
std::string build_detect_text_prompt();

// GUI grounding (box or point).
//   point: "Point to: {query}."
//   box:   "Locate the region that matches the following description: {query}."
std::string build_ground_gui_prompt(const std::string& query,
                                    GuiOutputType output_type = GuiOutputType::kBox);

// Convenience overload accepting the raw worker-style string ("box" | "point").
// Any value other than "point" maps to the box prompt (matching the worker's
// `if output_type == "point": ... else: ...`).
std::string build_ground_gui_prompt(const std::string& query,
                                    const std::string& output_type);

// Pointing.
//   "Point to: {query}."
std::string build_point_prompt(const std::string& query);

}  // namespace api
}  // namespace la

#endif  // LA_API_PROMPTS_H
