// SPDX-License-Identifier: Apache-2.0
// VisionEngine — TensorRT runtime wrapper for the LocateAnything MoonViT + mlp1
// vision tower (Phase 1).
//
// Phase 1 scope (see docs/superpowers/specs/2026-05-29-locateanything-cpp-trt-design.md
// Section 4): a single fixed-resolution (N=1) engine that maps preprocessed image
// patches to LLM-space visual embeddings.
//
//   pixel_values[L, 3, 14, 14] (FP16)  ->  vit_embeds[L/4, H_llm] (FP16)
//
// The canonical Phase 1 ONNX graph (cpp_trt/export/export_vision_onnx.py) has a
// SINGLE input (pixel_values) and a SINGLE output (vit_embeds). There is no
// grid_hws input at N=1 (cu_seqlens degenerates to [0, L]; pos-emb and RoPE are
// baked constants for the resolution bucket). This wrapper matches that contract.
//
// All TensorRT/CUDA dependencies are compiled only when LA_BUILD_TRT is defined.
// With LA_BUILD_TRT off (the default), this header still declares the full public
// interface so callers and tests compile, but there is no library body.
//
// Design notes:
//   * batch=1 only (the model asserts batch_size == 1).
//   * Engine I/O is FP16 on the device. Host callers pass FP32 and the wrapper
//     converts to/from FP16; or pass a device FP16 pointer via RunOptions.
//   * Engine bindings (by name): "pixel_values" (in), "vit_embeds" (out).
#ifndef LA_VISION_VISIONENGINE_H
#define LA_VISION_VISIONENGINE_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace la::vision {

// Logical description of one engine I/O tensor (no TensorRT dependency).
struct TensorSpec {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype;  // "fp16", "fp32", "int32"
};

// Result of a vision forward pass: the LLM-space visual embeddings.
struct VisionEmbeds {
  std::vector<int64_t> shape;  // [L/4, H_llm]
  std::vector<float> data;     // host copy (FP32, converted from engine FP16)
};

// Optional knobs for a single run().
struct RunOptions {
  bool input_is_device_ptr = false;  // pixel_values already on device as FP16
  bool output_to_host = true;        // copy vit_embeds back to host
};

class VisionEngine {
 public:
  // Construct from a serialized TRT engine file. Throws std::runtime_error on
  // failure (or if built without LA_BUILD_TRT).
  explicit VisionEngine(const std::string& engine_path);
  ~VisionEngine();

  VisionEngine(const VisionEngine&) = delete;
  VisionEngine& operator=(const VisionEngine&) = delete;

  // Run the vision tower. pixel_values is [L,3,14,14] row-major: FP32 on the host
  // by default, or an FP16 device pointer when opts.input_is_device_ptr is set.
  // Returns vit_embeds [L/4, H_llm].
  VisionEmbeds run(const float* pixel_values, int64_t num_patches,
                   RunOptions opts = RunOptions{});

  // Static description of the engine's expected I/O (from the spec, not the
  // loaded engine). Useful for validation/tests without a real engine.
  static std::vector<TensorSpec> io_spec();

 private:
#ifdef LA_BUILD_TRT
  struct Impl;
  std::unique_ptr<Impl> impl_;
#endif
};

}  // namespace la::vision

#endif  // LA_VISION_VISIONENGINE_H
