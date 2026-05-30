// SPDX-License-Identifier: Apache-2.0
//
// llm_engine_standalone_trt.cpp
//
// Standalone-TensorRT implementation of ILlmEngine for the Qwen2.5 decoder.
//
// THE ENTIRE FILE BODY IS BEHIND #ifdef LA_BUILD_TRT. In a TRT-free build this
// translation unit is not added to the la_llm target (see CMakeLists.txt); the
// guard is kept defensively so the file is always safe to compile.
//
// ---------------------------------------------------------------------------
// This is the backend that DOES support an arbitrary additive 4D mask, and is
// therefore the required path for Parallel Box Decoding (PBD) Phases 3-4
// (spec Section 5). The Qwen2.5 decoder is exported (ELSEWHERE -- see the
// Python export scripts; not run here) to a plain TensorRT engine whose
// network IO is:
//
//   inputs:
//     inputs_embeds : float32 [1, q_len, hidden]        (dynamic q_len)
//     attn_mask_4d  : float32 [1, 1, q_len, kv_len+q_len] additive (dynamic)
//     position_ids  : int64   [1, q_len]                 (dynamic)
//     past_key_*    : float   [1, n_kv_heads, kv_len, head_dim]  (per layer)
//     past_value_*  : float   [1, n_kv_heads, kv_len, head_dim]  (per layer)
//   outputs:
//     logits        : float32 [1, q_len, vocab]
//     present_key_*  / present_value_*  (per layer, length kv_len+q_len)
//
// Because the mask is a network INPUT, this backend honours
// StepInput::attn_mask_4d byte-for-byte (block-diagonal / banded PBD masks,
// MTP look-ahead masks, etc.). When attn_mask_4d is null we synthesize a causal
// additive mask ourselves so the Phase 2 contract (null == built-in causal) is
// satisfied with identical step() semantics to the TrtLlm backend.
//
// KV cache is held GPU-side as a ping/pong pair of device buffers per layer; on
// each step the engine reads past_* of length kv_len and writes present_* of
// length kv_len+q_len, after which we swap. rollback_kv() simply shrinks the
// logical length (no data move needed; the surplus tail is overwritten on the
// next step) -- this is what makes PBD speculate-and-accept cheap.
// ---------------------------------------------------------------------------

#include "la/llm/LlmEngine.h"

#ifdef LA_BUILD_TRT

#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// NOTE: TensorRT headers confined to this guarded region.
//   #include "NvInfer.h"
//   #include <cuda_runtime_api.h>
// Referenced via nvinfer1:: in the real implementation.

namespace la {
namespace llm {

namespace {

// Large negative used for masked positions in the synthesized causal mask.
constexpr float kMaskNegInf = -3.0e38f;  // ~ -FLT_MAX, additive "do not attend"

class StandaloneTrtEngine final : public ILlmEngine {
 public:
  explicit StandaloneTrtEngine(const LlmEngineConfig& cfg) : cfg_(cfg) {
    // --- Real implementation (build elsewhere; needs a TensorRT install):
    //   1. Read serialized engine bytes from cfg.engine_path.
    //   2. runtime_ = nvinfer1::createInferRuntime(logger);
    //      engine_  = runtime_->deserializeCudaEngine(bytes...);
    //      context_ = engine_->createExecutionContext();
    //   3. Discover hidden_, vocab_, n_layers_, n_kv_heads_, head_dim_ from the
    //      engine's IO tensor shapes; validate against cfg_.hidden / cfg_.vocab
    //      when those are > 0 (throw LlmEngineError on mismatch).
    //   4. cudaSetDevice(cfg.device_id); allocate per-layer KV device buffers
    //      sized for max_seq_len.
    //
    // Until wired against a concrete TensorRT install on the deployment machine,
    // construction is reported as unimplemented so it never silently misbehaves.
    throw LlmEngineError(
        "StandaloneTrtEngine: TensorRT engine integration is a build-elsewhere "
        "stub. Wire it against the installed TensorRT runtime (NvInfer) on the "
        "deployment machine. The interface and 4D-mask contract are final.");
  }

  StepOutput step(const StepInput& in) override {
    validate_step_input_(in);

    // Synthesize a causal additive mask if the caller passed null (Phase 2
    // contract). When a mask is provided (PBD), it is used verbatim.
    const int kv_total = in.kv_len + in.q_len;
    std::vector<float> causal_owned;
    const float* mask = in.attn_mask_4d;
    if (mask == nullptr) {
      causal_owned = build_causal_additive_mask_(in.q_len, in.kv_len);
      mask = causal_owned.data();
    }
    (void)mask;       // -> bound to the engine's attn_mask_4d input tensor
    (void)kv_total;   // -> sets the dynamic kv dimension on context_

    // --- Real step:
    //   * setInputShape for inputs_embeds [1,q_len,hidden], attn_mask_4d
    //     [1,1,q_len,kv_total], position_ids [1,q_len], past_*[...,kv_len,...].
    //   * H2D copy inputs_embeds, mask, position_ids.
    //   * bind past_* (current KV) and present_* (next KV) device buffers.
    //   * context_->enqueueV3(stream); cudaStreamSynchronize(stream).
    //   * D2H copy logits [q_len, vocab] into out.logits.
    //   * swap KV ping/pong; kv_len_ += q_len.
    StepOutput out;
    out.vocab = vocab_;
    out.logits.assign(static_cast<size_t>(in.q_len) * vocab_, 0.0f);
    kv_len_ += in.q_len;
    return out;
  }

  void reset_kv() override { kv_len_ = 0; }

  int kv_len() const override { return kv_len_; }

  void rollback_kv(int to_len) override {
    if (to_len < 0 || to_len > kv_len_) {
      std::ostringstream os;
      os << "StandaloneTrtEngine::rollback_kv: to_len=" << to_len
         << " out of range [0, " << kv_len_ << "]";
      throw LlmEngineError(os.str());
    }
    // KV is stored contiguously; shrinking the logical length is enough -- the
    // next step() overwrites the tail starting at to_len. No device copy needed.
    kv_len_ = to_len;
  }

  Backend backend() const override { return Backend::StandaloneTrt; }

  bool supports_dense_4d_mask() const override { return true; }

 private:
  // Build [1,1,q_len, kv_len+q_len] additive causal mask, row-major over
  // (q, kv). Query position i (0-based within this step) corresponds to absolute
  // position kv_len + i and may attend to all keys k with k <= kv_len + i.
  static std::vector<float> build_causal_additive_mask_(int q_len, int kv_len) {
    const int kv_total = kv_len + q_len;
    std::vector<float> m(static_cast<size_t>(q_len) * kv_total, 0.0f);
    for (int i = 0; i < q_len; ++i) {
      const int abs_q = kv_len + i;
      float* row = m.data() + static_cast<size_t>(i) * kv_total;
      for (int k = 0; k < kv_total; ++k) {
        row[k] = (k <= abs_q) ? 0.0f : kMaskNegInf;
      }
    }
    return m;
  }

  void validate_step_input_(const StepInput& in) const {
    if (in.inputs_embeds == nullptr || in.q_len <= 0) {
      throw LlmEngineError(
          "StandaloneTrtEngine::step: inputs_embeds null or q_len<=0");
    }
    if (in.position_ids == nullptr) {
      throw LlmEngineError("StandaloneTrtEngine::step: position_ids null");
    }
    if (in.kv_len != kv_len_) {
      std::ostringstream os;
      os << "StandaloneTrtEngine::step: in.kv_len=" << in.kv_len
         << " disagrees with engine kv_len()=" << kv_len_;
      throw LlmEngineError(os.str());
    }
    if (cfg_.hidden > 0 && in.hidden != cfg_.hidden) {
      std::ostringstream os;
      os << "StandaloneTrtEngine::step: in.hidden=" << in.hidden
         << " != configured hidden=" << cfg_.hidden;
      throw LlmEngineError(os.str());
    }
  }

  LlmEngineConfig cfg_;
  int hidden_ = 0;
  int vocab_ = 0;
  int n_layers_ = 0;
  int kv_len_ = 0;
  // nvinfer1::IRuntime*          runtime_ = nullptr;
  // nvinfer1::ICudaEngine*       engine_  = nullptr;
  // nvinfer1::IExecutionContext* context_ = nullptr;
};

}  // namespace

std::unique_ptr<ILlmEngine> make_standalone_trt_engine(const LlmEngineConfig& cfg) {
  return std::make_unique<StandaloneTrtEngine>(cfg);
}

}  // namespace llm
}  // namespace la

#endif  // LA_BUILD_TRT
