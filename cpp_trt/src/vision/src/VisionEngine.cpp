// SPDX-License-Identifier: Apache-2.0
// VisionEngine — TensorRT runtime implementation (Phase 1).
//
// Compiled only when LA_BUILD_TRT is defined; otherwise this translation unit
// provides throwing stubs so the default (TRT-free) build links if a caller
// references the symbols.
//
// NOTE: This TRT path has NOT been compiled or run in the development
// environment (no TensorRT / GPU engine / checkpoint available). It targets the
// TensorRT 10.x C++ API (enqueueV3 / setTensorAddress / setInputShape) and must
// be built + validated on a machine with TensorRT 10.11 and a real engine.
//
#include "la/vision/VisionEngine.h"

#ifdef LA_BUILD_TRT

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace la::vision {
namespace {

// TensorRT logger that forwards warnings and errors to stderr.
class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[TRT] %s\n", msg);
    }
  }
};

TrtLogger g_logger;

template <typename T>
struct TrtDeleter {
  void operator()(T* p) const noexcept {
    if (p) delete p;
  }
};
template <typename T>
using TrtUnique = std::unique_ptr<T, TrtDeleter<T>>;

void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("VisionEngine CUDA error: ") + what +
                             ": " + cudaGetErrorString(e));
  }
}

int64_t volume(const nvinfer1::Dims& d) {
  int64_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
  return v;
}

}  // namespace

struct VisionEngine::Impl {
  TrtUnique<nvinfer1::IRuntime> runtime;
  TrtUnique<nvinfer1::ICudaEngine> engine;
  TrtUnique<nvinfer1::IExecutionContext> context;
  cudaStream_t stream{nullptr};

  static constexpr const char* kInputPixelValues = "pixel_values";
  static constexpr const char* kOutputVitEmbeds = "vit_embeds";
  static constexpr int64_t kSpatialMergeTokens = 4;  // 2x2 patch merge

  ~Impl() {
    if (stream) cudaStreamDestroy(stream);
  }
};

// ----------------------------------------------------------------------------
// Construction / teardown
// ----------------------------------------------------------------------------

VisionEngine::VisionEngine(const std::string& engine_path) {
  impl_ = std::make_unique<Impl>();

  std::ifstream f(engine_path, std::ios::binary);
  if (!f) throw std::runtime_error("VisionEngine: cannot open engine file: " + engine_path);
  std::vector<char> blob((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
  if (blob.empty()) throw std::runtime_error("VisionEngine: empty engine file: " + engine_path);

  impl_->runtime.reset(nvinfer1::createInferRuntime(g_logger));
  if (!impl_->runtime) throw std::runtime_error("VisionEngine: createInferRuntime failed");

  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!impl_->engine) throw std::runtime_error("VisionEngine: deserializeCudaEngine failed");

  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) throw std::runtime_error("VisionEngine: createExecutionContext failed");

  cuda_check(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");
}

VisionEngine::~VisionEngine() = default;

// ----------------------------------------------------------------------------
// Inference: pixel_values[L,3,14,14] FP16 -> vit_embeds[L/4, H_llm] FP16
// ----------------------------------------------------------------------------

VisionEmbeds VisionEngine::run(const float* pixel_values, int64_t num_patches,
                               const RunOptions& opts) {
  Impl& impl = *impl_;

  // ---- Set the dynamic input shape and bind it ----
  const nvinfer1::Dims4 pv_dims(static_cast<int>(num_patches), 3, 14, 14);
  if (!impl.context->setInputShape(Impl::kInputPixelValues, pv_dims)) {
    throw std::runtime_error("VisionEngine: setInputShape(pixel_values) failed");
  }

  const int64_t pv_count = num_patches * 3 * 14 * 14;
  const size_t pv_bytes = static_cast<size_t>(pv_count) * sizeof(__half);

  // ---- Output shape is now resolved by the engine ----
  const nvinfer1::Dims out_dims = impl.context->getTensorShape(Impl::kOutputVitEmbeds);
  if (out_dims.nbDims < 1) {
    throw std::runtime_error("VisionEngine: could not resolve vit_embeds shape");
  }
  const int64_t out_count = volume(out_dims);
  const size_t out_bytes = static_cast<size_t>(out_count) * sizeof(__half);

  // ---- Device buffers ----
  void* d_pixel = nullptr;
  void* d_out = nullptr;
  cuda_check(cudaMalloc(&d_out, out_bytes), "cudaMalloc(vit_embeds)");

  std::vector<__half> h_pixel;  // staged FP16 host copy (host-input path)
  try {
    if (opts.input_is_device_ptr) {
      // Caller-owned FP16 device buffer; bind directly (no copy, no free here).
      d_pixel = const_cast<void*>(static_cast<const void*>(pixel_values));
    } else {
      // Convert host FP32 -> FP16 and upload.
      h_pixel.resize(static_cast<size_t>(pv_count));
      for (int64_t i = 0; i < pv_count; ++i) h_pixel[i] = __float2half(pixel_values[i]);
      cuda_check(cudaMalloc(&d_pixel, pv_bytes), "cudaMalloc(pixel_values)");
      cuda_check(cudaMemcpyAsync(d_pixel, h_pixel.data(), pv_bytes,
                                 cudaMemcpyHostToDevice, impl.stream),
                 "H2D pixel_values");
    }

    impl.context->setTensorAddress(Impl::kInputPixelValues, d_pixel);
    impl.context->setTensorAddress(Impl::kOutputVitEmbeds, d_out);

    if (!impl.context->enqueueV3(impl.stream)) {
      throw std::runtime_error("VisionEngine: enqueueV3 failed");
    }

    // ---- Retrieve output ----
    VisionEmbeds result;
    result.shape.assign(out_dims.d, out_dims.d + out_dims.nbDims);

    if (opts.output_to_host) {
      std::vector<__half> h_out(static_cast<size_t>(out_count));
      cuda_check(cudaMemcpyAsync(h_out.data(), d_out, out_bytes,
                                 cudaMemcpyDeviceToHost, impl.stream),
                 "D2H vit_embeds");
      cuda_check(cudaStreamSynchronize(impl.stream), "stream sync");
      result.data.resize(static_cast<size_t>(out_count));
      for (int64_t i = 0; i < out_count; ++i) result.data[i] = __half2float(h_out[i]);
    } else {
      cuda_check(cudaStreamSynchronize(impl.stream), "stream sync");
    }

    if (!opts.input_is_device_ptr && d_pixel) cudaFree(d_pixel);
    cudaFree(d_out);
    return result;
  } catch (...) {
    if (!opts.input_is_device_ptr && d_pixel) cudaFree(d_pixel);
    if (d_out) cudaFree(d_out);
    throw;
  }
}

// ----------------------------------------------------------------------------
// Static spec — single input, single output (N=1 Phase 1 contract)
// ----------------------------------------------------------------------------

std::vector<TensorSpec> VisionEngine::io_spec() {
  return {
      {"pixel_values", {-1, 3, 14, 14}, "fp16"},
      {"vit_embeds", {-1, -1}, "fp16"},
  };
}

}  // namespace la::vision

#else  // !LA_BUILD_TRT

#include <stdexcept>

namespace la::vision {

VisionEngine::VisionEngine(const std::string&) {
  throw std::runtime_error(
      "VisionEngine: built without LA_BUILD_TRT; rebuild with -DLA_BUILD_TRT=ON");
}
VisionEngine::~VisionEngine() = default;

VisionEmbeds VisionEngine::run(const float*, int64_t, const RunOptions&) {
  throw std::runtime_error("VisionEngine: built without LA_BUILD_TRT");
}

std::vector<TensorSpec> VisionEngine::io_spec() { return {}; }

}  // namespace la::vision

#endif  // LA_BUILD_TRT
