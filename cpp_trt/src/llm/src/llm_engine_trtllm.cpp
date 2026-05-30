// SPDX-License-Identifier: Apache-2.0
//
// llm_engine_trtllm.cpp
//
// TRT-LLM-backed implementation of ILlmEngine for the Qwen2.5 decoder.
//
// THE ENTIRE FILE BODY IS BEHIND #ifdef LA_BUILD_TRT. In a TRT-free build this
// translation unit is not even added to the la_llm target (see CMakeLists.txt),
// but the guard is kept defensively so the file is always safe to compile.
//
// ---------------------------------------------------------------------------
// Risk R5 / engine-count decision (spec Section 3) and the 4D-mask question
// (spec Section 5):
//
// The TRT-LLM C++ runtime (tensorrt_llm::runtime / the GptSession / executor
// APIs) is built around *causal* decoder-only generation. It manages the KV
// cache and the causal mask INTERNALLY from the (context_length, past_length)
// it tracks per request. As of the runtime versions we target it exposes:
//   - inputs_embeds / prompt-embedding-table style input (continuous batching),
//   - position ids (for RoPE),
//   - but NOT a per-step user-supplied dense additive 4D attention mask. The
//     packed-mask / attention-mask inputs that exist are derived from sequence
//     lengths (i.e. they encode causal + padding), not an arbitrary [q,kv]
//     additive matrix.
//
// WHERE THE LIMITATION BITES (documented precisely):
//   PBD (Phases 3-4) needs an arbitrary additive 4D mask so that parallel box
//   slots in one window can attend to the shared prefix but NOT to each other
//   (a block-diagonal / banded structure), and so MTP windows can implement
//   their look-ahead mask. The TRT-LLM runtime cannot accept that matrix at
//   step() time without a custom plugin/engine rebuild. THEREFORE:
//
//     * For Phase 2 (NTP/AR), the required mask is exactly causal, which is what
//       TRT-LLM applies by construction. So we IGNORE a null
//       StepInput::attn_mask_4d (the documented Phase 2 contract) and let the
//       runtime do causal masking. supports_dense_4d_mask() returns false.
//
//     * If a caller passes a NON-null attn_mask_4d to this backend, we do NOT
//       silently produce wrong results: we throw LlmEngineError telling the
//       caller to use Backend::StandaloneTrt for dense-4D-mask workloads (PBD).
//       This is the "write both mask paths defensively" contract: the decode
//       loop queries supports_dense_4d_mask() and routes accordingly; if it
//       mis-routes, it gets a loud error rather than a quiet correctness bug.
// ---------------------------------------------------------------------------

#include "la/llm/LlmEngine.h"

#ifdef LA_BUILD_TRT

#include <memory>
#include <sstream>
#include <string>
#include <vector>

// NOTE: TRT-LLM / TensorRT headers are intentionally confined to this guarded
// region. The exact include set depends on the installed TRT-LLM version; the
// canonical public C++ entry points are the executor API:
//
//   #include "tensorrt_llm/executor/executor.h"
//   #include "tensorrt_llm/runtime/common.h"
//
// They are referenced through the tllm:: aliases below. We deliberately keep the
// integration thin and well-isolated; the heavy lifting (KV cache management,
// causal masking, RoPE) is done by the runtime.
//
// #include "tensorrt_llm/executor/executor.h"
// namespace tllm = tensorrt_llm::executor;

namespace la {
namespace llm {

namespace {

class TrtLlmEngine final : public ILlmEngine {
 public:
  explicit TrtLlmEngine(const LlmEngineConfig& cfg) : cfg_(cfg) {
    // --- Load / construct the TRT-LLM executor for the engine at
    //     cfg.engine_path. Pseudocode for the target runtime version:
    //
    //   tllm::ExecutorConfig ec;
    //   ec.setKvCacheConfig(...);            // provision KV up to max_seq_len
    //   executor_ = std::make_unique<tllm::Executor>(
    //       cfg.engine_path, tllm::ModelType::kDECODER_ONLY, ec);
    //   hidden_ = <from engine config json>;
    //   vocab_  = <from engine config json>;
    //
    // We validate against cfg_.hidden / cfg_.vocab if those are > 0 and throw
    // LlmEngineError on mismatch. KV bookkeeping (kv_len_) is mirrored locally
    // because step() below drives the runtime one request/extension at a time
    // for the batch==1 streaming case.
    //
    // Until the integration is wired against a concrete TRT-LLM version on the
    // target machine, constructing this backend is reported as unimplemented so
    // it never silently misbehaves in a partially-built tree.
    throw LlmEngineError(
        "TrtLlmEngine: TRT-LLM C++ runtime integration is a build-elsewhere "
        "stub. Wire it against the installed tensorrt_llm executor API on the "
        "deployment machine. For dense-4D-mask (PBD) workloads use "
        "Backend::StandaloneTrt instead.");
  }

  StepOutput step(const StepInput& in) override {
    validate_step_input_(in);

    // Risk R5 contract: TRT-LLM only does causal masking. A null mask means
    // "use built-in causal", which is exactly what NTP/AR needs. A non-null
    // mask cannot be honoured here.
    if (in.attn_mask_4d != nullptr) {
      throw LlmEngineError(
          "TrtLlmEngine::step: a dense 4D attention mask was supplied, but the "
          "TRT-LLM runtime only applies built-in causal masking. Use "
          "Backend::StandaloneTrt for PBD / arbitrary-mask decoding.");
    }

    // --- Drive the runtime for one step:
    //   * prefill (kv_len==0): submit a Request whose inputEmbeds are the
    //     [q_len, hidden] block and positionIds are in.position_ids, asking for
    //     a single forward (no autoregressive expansion here -- we own the
    //     decode loop and sampling in pure C++).
    //   * extend (kv_len>0): submit the continuation with q_len new embeds; the
    //     runtime appends to its KV cache.
    //   * read back logits for all q_len positions into out.logits
    //     ([q_len, vocab]).
    // Local KV bookkeeping advances by q_len on success.
    StepOutput out;
    out.vocab = vocab_;
    out.logits.assign(static_cast<size_t>(in.q_len) * vocab_, 0.0f);
    kv_len_ += in.q_len;
    return out;
  }

  void reset_kv() override {
    // Cancel/replace the in-flight request so the next step() is a fresh
    // prefill; reset local bookkeeping.
    kv_len_ = 0;
  }

  int kv_len() const override { return kv_len_; }

  void rollback_kv(int to_len) override {
    if (to_len < 0 || to_len > kv_len_) {
      std::ostringstream os;
      os << "TrtLlmEngine::rollback_kv: to_len=" << to_len
         << " out of range [0, " << kv_len_ << "]";
      throw LlmEngineError(os.str());
    }
    // TRT-LLM's executor KV cache is not trivially truncatable mid-request;
    // PBD (which needs rollback) is a StandaloneTrt workload. We keep the local
    // counter consistent so a caller that only uses rollback in a causal
    // re-step (to_len == kv_len_) is fine, and throw if a true mid-cache
    // truncation is requested.
    if (to_len != kv_len_) {
      throw LlmEngineError(
          "TrtLlmEngine::rollback_kv: mid-cache truncation is not supported by "
          "the TRT-LLM runtime KV cache. Use Backend::StandaloneTrt for PBD.");
    }
  }

  Backend backend() const override { return Backend::TrtLlm; }

  bool supports_dense_4d_mask() const override { return false; }

 private:
  void validate_step_input_(const StepInput& in) const {
    if (in.inputs_embeds == nullptr || in.q_len <= 0) {
      throw LlmEngineError("TrtLlmEngine::step: inputs_embeds null or q_len<=0");
    }
    if (in.position_ids == nullptr) {
      throw LlmEngineError("TrtLlmEngine::step: position_ids null");
    }
    if (in.kv_len != kv_len_) {
      std::ostringstream os;
      os << "TrtLlmEngine::step: in.kv_len=" << in.kv_len
         << " disagrees with engine kv_len()=" << kv_len_;
      throw LlmEngineError(os.str());
    }
    if (cfg_.hidden > 0 && in.hidden != cfg_.hidden) {
      std::ostringstream os;
      os << "TrtLlmEngine::step: in.hidden=" << in.hidden
         << " != configured hidden=" << cfg_.hidden;
      throw LlmEngineError(os.str());
    }
  }

  LlmEngineConfig cfg_;
  int hidden_ = 0;
  int vocab_ = 0;
  int kv_len_ = 0;
  // std::unique_ptr<tllm::Executor> executor_;  // owned runtime handle
};

}  // namespace

std::unique_ptr<ILlmEngine> make_trtllm_engine(const LlmEngineConfig& cfg) {
  return std::make_unique<TrtLlmEngine>(cfg);
}

}  // namespace llm
}  // namespace la

#endif  // LA_BUILD_TRT
