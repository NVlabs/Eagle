// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
// la/api/worker.h
//
// LocateAnythingWorker-style public C++ API. This is the C++ analog of
// Embodied/locateanything_worker.py: a stateful object that, given an image and
// a task, builds the (load-bearing) task prompt, runs it through the chat
// template + tokenizer + vision + LLM decode pipeline, and parses the
// structured box/point output back into pixel coordinates.
//
// TRT-free vs engine-touching split (Phase 2 contract):
//   * Prompt building (la::api::prompts) and result parsing (parse_boxes /
//     parse_points) are PURE C++ and always available -- compiled and
//     unit-tested in the default (TRT-free) build.
//   * The end-to-end predict() body -- which wires tokenizer chat template ->
//     tokenize -> preprocess -> vision engine -> embed splice -> LLM decode
//     loop -> parse -- requires the TensorRT / TRT-LLM engines and is therefore
//     compiled ONLY when LA_BUILD_TRT is defined. In the default build the
//     engine-touching entry points throw a std::runtime_error stub so the
//     library still links and the API surface is stable.

#ifndef LA_API_WORKER_H
#define LA_API_WORKER_H

#include <string>
#include <vector>

#include "la/api/prompts.h"

namespace la {
namespace api {

// Generation mode, mirroring the worker's `generation_mode` string argument.
//   fast   -> MTP (Parallel Box Decoding) only
//   slow   -> NTP (autoregressive) only
//   hybrid -> MTP with AR fallback   (worker default)
enum class GenerationMode {
  kFast,
  kSlow,
  kHybrid,
};

// Decoding / generation parameters. Defaults mirror
// LocateAnythingWorker.predict() exactly.
struct GenerationParams {
  GenerationMode mode = GenerationMode::kHybrid;  // worker default "hybrid"
  int max_new_tokens = 2048;                      // worker default 2048
  // The worker passes do_sample=True, temperature=0.7, top_p=0.9. Per the
  // C++/TRT design (Section 2, "Resolved Decisions"): v1 decode is greedy +
  // repetition_penalty for tight parity; production sampling is not implemented
  // in v1. These fields record the reference values so the port can match
  // behaviour exactly once sampling is enabled.
  bool do_sample = true;             // worker default True
  float temperature = 0.7f;          // worker default 0.7
  float top_p = 0.9f;                // worker default 0.9
  float repetition_penalty = 1.1f;   // worker default 1.1
  bool verbose = true;               // worker default True
};

// A single parsed bounding box in pixel coordinates (xyxy). Mirrors the dict
// produced by LocateAnythingWorker.parse_boxes().
struct Box {
  double x1 = 0.0;
  double y1 = 0.0;
  double x2 = 0.0;
  double y2 = 0.0;
};

// A single parsed point in pixel coordinates. Mirrors the dict produced by
// LocateAnythingWorker.parse_points().
struct Point {
  double x = 0.0;
  double y = 0.0;
};

// Raw RGB image view handed to the worker. The C++ pipeline (preprocess module)
// consumes a tightly-packed HxWx3 uint8 RGB buffer. This struct is a
// non-owning view; the caller owns the pixel storage.
struct ImageRGB {
  const unsigned char* data = nullptr;  // row-major, HxWx3, 8-bit RGB
  int width = 0;
  int height = 0;
};

// Result of a predict() call. `answer` is the raw decoded string (containing
// <box>...</box> / <ref>...</ref> tokens); the parsed convenience views are
// derived from it.
struct PredictResult {
  std::string answer;
};

// LocateAnythingWorker-style API.
//
// Construction loads the engines (TRT build) or is a lightweight no-op holder of
// configuration (TRT-free build). The prompt-building + parsing methods are
// always usable regardless of build flavour.
class LocateAnythingWorker {
 public:
  // model_dir: path to the exported checkpoint / engine directory (tokenizer.json,
  // preprocessor_config.json, vision + llm engines, token-id constants).
  explicit LocateAnythingWorker(std::string model_dir);
  ~LocateAnythingWorker();

  LocateAnythingWorker(const LocateAnythingWorker&) = delete;
  LocateAnythingWorker& operator=(const LocateAnythingWorker&) = delete;
  LocateAnythingWorker(LocateAnythingWorker&&) noexcept;
  LocateAnythingWorker& operator=(LocateAnythingWorker&&) noexcept;

  // ---- Core entry point (engine-touching; stubbed when TRT is absent) ----
  //
  // Builds the chat-templated input for `question`, runs the full pipeline, and
  // returns the decoded answer. Throws std::runtime_error in a TRT-free build.
  PredictResult predict(const ImageRGB& image,
                        const std::string& question,
                        const GenerationParams& params = {});

  // ---- Convenience task methods (mirror the worker verbatim) ----
  // Each composes the appropriate (load-bearing) prompt via la::api::prompts
  // and forwards to predict(). They throw in a TRT-free build because predict()
  // does; the prompt strings they compose are still exercised by the pure-C++
  // prompt unit tests.

  PredictResult detect(const ImageRGB& image,
                       const std::vector<std::string>& categories,
                       const GenerationParams& params = {});

  PredictResult ground_single(const ImageRGB& image,
                              const std::string& phrase,
                              const GenerationParams& params = {});

  PredictResult ground_multi(const ImageRGB& image,
                             const std::string& phrase,
                             const GenerationParams& params = {});

  PredictResult ground_text(const ImageRGB& image,
                            const std::string& phrase,
                            const GenerationParams& params = {});

  PredictResult detect_text(const ImageRGB& image,
                            const GenerationParams& params = {});

  PredictResult ground_gui(const ImageRGB& image,
                           const std::string& phrase,
                           GuiOutputType output_type = GuiOutputType::kBox,
                           const GenerationParams& params = {});

  PredictResult point(const ImageRGB& image,
                      const std::string& phrase,
                      const GenerationParams& params = {});

  // ---- Output parsing (PURE C++; always available) ----
  // Verbatim ports of LocateAnythingWorker.parse_boxes / parse_points: scan the
  // answer for <box>...</box> wrappers and convert the per-mille [0,1000]
  // integer coordinates to pixels. Distinguished by integer-group count
  // (4 -> box, 2 -> point) inside the same wrapper.
  static std::vector<Box> parse_boxes(const std::string& answer,
                                      int image_width, int image_height);

  static std::vector<Point> parse_points(const std::string& answer,
                                         int image_width, int image_height);

 private:
  // Opaque engine/runtime state. Only populated in a LA_BUILD_TRT build.
  struct Impl;
  Impl* impl_ = nullptr;
  std::string model_dir_;
};

}  // namespace api
}  // namespace la

#endif  // LA_API_WORKER_H
