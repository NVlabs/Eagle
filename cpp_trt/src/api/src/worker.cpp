// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
// la/api/worker.cpp
//
// LocateAnythingWorker implementation.
//
//   * Prompt composition + result parsing are PURE C++ and always compiled.
//   * The engine-touching pipeline (predict()) is compiled only under
//     LA_BUILD_TRT. Without it, predict() throws a clear std::runtime_error so
//     the default build links and the API surface stays stable.

#include "la/api/worker.h"

#include <cctype>
#include <stdexcept>
#include <utility>

#if defined(LA_BUILD_TRT)
// NOTE (assumption, gated behind LA_BUILD_TRT so the default build is
// unaffected): the engine path is expected to reuse the sibling modules
//   la::tokenizer  -- chat template ("py_apply_chat_template" reimpl) + tokenize
//   la::config     -- token-id constants (image_token_index, coord_start, ...)
//   la::preprocess -- image -> pixel_values[L,3,14,14] + grid_hws
//   la::vision     -- MoonViT + mlp1 -> vit_embeds[L/4, H_llm]
//   la::decode     -- MTP/AR 4D masks + position ids
//   la::generate   -- PBD/AR decode loop, sampling, repetition penalty,
//                     coord detokenize, box/point parsing
// The exact headers/signatures of those modules are not pinned here; this TU
// only #includes them inside this guard so the default (TRT-free) build never
// requires them. Wire the concrete calls when the engines land (Phase 2-4).
#endif

namespace la {
namespace api {

// ---------------------------------------------------------------------------
// Engine state (opaque). Empty in the TRT-free build.
// ---------------------------------------------------------------------------
struct LocateAnythingWorker::Impl {
#if defined(LA_BUILD_TRT)
  // Engines / runtime handles live here in the TRT build:
  //   la::vision::VisionEngine vision;
  //   <LLM engine handle>      llm;
  //   la::tokenizer::Tokenizer tokenizer;
  //   la::config::TokenIds     token_ids;
  // Left undeclared until the engine modules' types are pinned.
#endif
};

// ---------------------------------------------------------------------------
// Construction / lifetime
// ---------------------------------------------------------------------------
LocateAnythingWorker::LocateAnythingWorker(std::string model_dir)
    : model_dir_(std::move(model_dir)) {
#if defined(LA_BUILD_TRT)
  impl_ = new Impl();
  // TODO(phase2+): load tokenizer.json, preprocessor_config.json, token-id
  // constants, and the vision + LLM engines from model_dir_ into *impl_.
#else
  // TRT-free build: no engines are loaded. The object is still usable for the
  // pure-C++ prompt builders and static parse_* helpers.
  impl_ = nullptr;
#endif
}

LocateAnythingWorker::~LocateAnythingWorker() { delete impl_; }

LocateAnythingWorker::LocateAnythingWorker(LocateAnythingWorker&& other) noexcept
    : impl_(other.impl_), model_dir_(std::move(other.model_dir_)) {
  other.impl_ = nullptr;
}

LocateAnythingWorker& LocateAnythingWorker::operator=(
    LocateAnythingWorker&& other) noexcept {
  if (this != &other) {
    delete impl_;
    impl_ = other.impl_;
    model_dir_ = std::move(other.model_dir_);
    other.impl_ = nullptr;
  }
  return *this;
}

// ---------------------------------------------------------------------------
// predict() -- engine-touching; stubbed when TRT is unavailable.
// ---------------------------------------------------------------------------
PredictResult LocateAnythingWorker::predict(const ImageRGB& image,
                                            const std::string& question,
                                            const GenerationParams& params) {
#if defined(LA_BUILD_TRT)
  // End-to-end pipeline (mirrors LocateAnythingWorker.predict):
  //   1. messages = [{user: [image, text=question]}]
  //   2. text = chat_template(messages, add_generation_prompt=True)   (la::tokenizer)
  //   3. pixel_values, grid_hws = preprocess(image)                   (la::preprocess)
  //   4. input_ids = tokenize(text), expand <IMG_CONTEXT> x (h*w)//4  (la::tokenizer)
  //   5. vit_embeds = vision(pixel_values, grid_hws)                  (la::vision)
  //   6. inputs_embeds = splice(embed(input_ids), vit_embeds @ image_token_index)
  //   7. answer = generate(inputs_embeds, params)                     (la::generate)
  //          mode/max_new_tokens/repetition_penalty/etc. from `params`
  //          (defaults already mirror the worker: hybrid, 2048, rp=1.1, ...)
  //   8. return {answer}
  (void)image;
  (void)question;
  (void)params;
  throw std::runtime_error(
      "la::api::LocateAnythingWorker::predict: LA_BUILD_TRT is defined but the "
      "engine pipeline is not yet wired (Phase 2-4). model_dir=" + model_dir_);
#else
  (void)image;
  (void)question;
  (void)params;
  throw std::runtime_error(
      "la::api::LocateAnythingWorker::predict requires the TensorRT / TRT-LLM "
      "engines (build with -DLA_BUILD_TRT=ON). The prompt builders "
      "(la::api::prompts) and parse_boxes/parse_points are available in this "
      "TRT-free build.");
#endif
}

// ---------------------------------------------------------------------------
// Convenience task methods -- compose the load-bearing prompt, then predict().
// ---------------------------------------------------------------------------
PredictResult LocateAnythingWorker::detect(
    const ImageRGB& image, const std::vector<std::string>& categories,
    const GenerationParams& params) {
  return predict(image, build_detect_prompt(categories), params);
}

PredictResult LocateAnythingWorker::ground_single(const ImageRGB& image,
                                                  const std::string& phrase,
                                                  const GenerationParams& params) {
  return predict(image, build_ground_single_prompt(phrase), params);
}

PredictResult LocateAnythingWorker::ground_multi(const ImageRGB& image,
                                                 const std::string& phrase,
                                                 const GenerationParams& params) {
  return predict(image, build_ground_multi_prompt(phrase), params);
}

PredictResult LocateAnythingWorker::ground_text(const ImageRGB& image,
                                                const std::string& phrase,
                                                const GenerationParams& params) {
  return predict(image, build_ground_text_prompt(phrase), params);
}

PredictResult LocateAnythingWorker::detect_text(const ImageRGB& image,
                                                const GenerationParams& params) {
  return predict(image, build_detect_text_prompt(), params);
}

PredictResult LocateAnythingWorker::ground_gui(const ImageRGB& image,
                                               const std::string& phrase,
                                               GuiOutputType output_type,
                                               const GenerationParams& params) {
  return predict(image, build_ground_gui_prompt(phrase, output_type), params);
}

PredictResult LocateAnythingWorker::point(const ImageRGB& image,
                                          const std::string& phrase,
                                          const GenerationParams& params) {
  return predict(image, build_point_prompt(phrase), params);
}

// ---------------------------------------------------------------------------
// Output parsing -- PURE C++ verbatim port of parse_boxes / parse_points.
//
// The Python uses regex:
//   boxes : r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
//   points: r"<box><(\d+)><(\d+)></box>"
// We hand-scan to avoid a <regex> dependency and to keep behaviour explicit.
// Coordinate value = group / 1000 * image_dim.
// ---------------------------------------------------------------------------
namespace {

// Try to parse, starting at s[pos], the sequence: literal "<box>" followed by
// exactly `count` "<digits>" groups followed by literal "</box>". On success,
// fills `out` with the `count` parsed integers, sets `next` to one past the
// matched region, and returns true. Otherwise returns false and leaves outputs
// untouched (caller advances by one char).
bool match_box_with_n_ints(const std::string& s, std::size_t pos, int count,
                           std::vector<long>& out, std::size_t& next) {
  const std::string kOpen = "<box>";
  const std::string kClose = "</box>";
  if (s.compare(pos, kOpen.size(), kOpen) != 0) {
    return false;
  }
  std::size_t p = pos + kOpen.size();
  std::vector<long> vals;
  vals.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    if (p >= s.size() || s[p] != '<') {
      return false;
    }
    ++p;  // consume '<'
    std::size_t digit_start = p;
    while (p < s.size() &&
           std::isdigit(static_cast<unsigned char>(s[p])) != 0) {
      ++p;
    }
    if (p == digit_start) {
      return false;  // \d+ requires at least one digit
    }
    if (p >= s.size() || s[p] != '>') {
      return false;
    }
    long v = 0;
    // Manual parse to avoid locale / exception overhead; digits only.
    for (std::size_t d = digit_start; d < p; ++d) {
      v = v * 10 + (s[d] - '0');
    }
    vals.push_back(v);
    ++p;  // consume '>'
  }
  if (s.compare(p, kClose.size(), kClose) != 0) {
    return false;
  }
  out = std::move(vals);
  next = p + kClose.size();
  return true;
}

}  // namespace

std::vector<Box> LocateAnythingWorker::parse_boxes(const std::string& answer,
                                                   int image_width,
                                                   int image_height) {
  std::vector<Box> boxes;
  const double w = static_cast<double>(image_width);
  const double h = static_cast<double>(image_height);
  std::size_t i = 0;
  while (i < answer.size()) {
    std::vector<long> vals;
    std::size_t next = 0;
    if (match_box_with_n_ints(answer, i, /*count=*/4, vals, next)) {
      Box b;
      b.x1 = static_cast<double>(vals[0]) / 1000.0 * w;
      b.y1 = static_cast<double>(vals[1]) / 1000.0 * h;
      b.x2 = static_cast<double>(vals[2]) / 1000.0 * w;
      b.y2 = static_cast<double>(vals[3]) / 1000.0 * h;
      boxes.push_back(b);
      i = next;  // non-overlapping, like re.finditer
    } else {
      ++i;
    }
  }
  return boxes;
}

std::vector<Point> LocateAnythingWorker::parse_points(const std::string& answer,
                                                      int image_width,
                                                      int image_height) {
  std::vector<Point> points;
  const double w = static_cast<double>(image_width);
  const double h = static_cast<double>(image_height);
  std::size_t i = 0;
  while (i < answer.size()) {
    std::vector<long> vals;
    std::size_t next = 0;
    if (match_box_with_n_ints(answer, i, /*count=*/2, vals, next)) {
      Point p;
      p.x = static_cast<double>(vals[0]) / 1000.0 * w;
      p.y = static_cast<double>(vals[1]) / 1000.0 * h;
      points.push_back(p);
      i = next;
    } else {
      ++i;
    }
  }
  return points;
}

}  // namespace api
}  // namespace la
