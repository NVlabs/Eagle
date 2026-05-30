// SPDX-License-Identifier: Apache-2.0
//
// la/llm/LlmEngine.h
//
// Abstract, TRT-FREE interface for the Qwen2.5 LLM decoder used by the
// LocateAnything C++/TRT port.
//
// This header MUST NOT include any TensorRT or TRT-LLM headers. It is the only
// surface the decode loop (la_decode + the Phase 2/3 generation driver) sees, so
// the loop is agnostic to whether the underlying runtime accepts a runtime dense
// 4D attention mask (TRT-LLM, NTP/AR causal-only) or an explicit 4D mask
// (standalone TRT decoder, required for PBD Phases 3-4).
//
// See docs/superpowers/specs/2026-05-29-locateanything-cpp-trt-design.md
//   Section 3 (engine-count decision / Risk R5) and Section 5 (LLM + decode).
//
// Engine-count decision (Risk R5): we DO NOT bake the choice between
// "single TRT-LLM engine" and "standalone TRT decoder engine" into the decode
// loop. Instead we define ONE interface (ILlmEngine) and TWO implementations,
// selected at construction time via LlmEngineConfig::backend. This keeps the
// risky 4D-mask question (does TRT-LLM accept a runtime dense additive 4D mask?)
// isolated to a single implementation file; the StandaloneTrt backend is the
// guaranteed-correct path for arbitrary masks (PBD), while TrtLlm is the
// fast path for the causal NTP/AR generation in Phase 2.

#ifndef LA_LLM_LLMENGINE_H
#define LA_LLM_LLMENGINE_H

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace la {
namespace llm {

// ---------------------------------------------------------------------------
// Backend selection.
// ---------------------------------------------------------------------------
//
// TrtLlm        : the decoder is served by the TRT-LLM C++ runtime. This path is
//                 fast and battle-tested for plain causal (left-to-right)
//                 generation. It is the Phase 2 NTP/AR path. Whether TRT-LLM
//                 accepts a *runtime dense additive 4D mask* is uncertain (Risk
//                 R5); the implementation documents precisely where that bites
//                 and falls back to its built-in causal masking, which is all
//                 NTP/AR needs.
//
// StandaloneTrt : the Qwen2.5 decoder is exported to a plain TensorRT engine
//                 whose IO is { inputs_embeds, attn_mask_4d, position_ids, past
//                 KV } -> { logits, present KV }. This path DOES honour an
//                 arbitrary additive 4D mask, and is therefore the required
//                 backend for Parallel Box Decoding (PBD) Phases 3-4. For Phase
//                 2 it implements the same step() with a causal mask.
enum class Backend {
  TrtLlm = 0,
  StandaloneTrt = 1,
};

// ---------------------------------------------------------------------------
// Per-step input (batch == 1 only, by design for Phase 2/3).
// ---------------------------------------------------------------------------
//
// A "step" is one decoder forward pass. It covers BOTH:
//   * prefill          : q_len == prompt length, kv_len == 0 on entry
//   * incremental decode: q_len == 1 (NTP) or q_len == N (MTP/PBD window),
//                         kv_len == number of tokens already cached.
//
// The caller owns all pointers; the engine copies what it needs during step().
struct StepInput {
  // Token embeddings for the q_len query positions, row-major,
  // shape [q_len, hidden]. These are produced upstream by embedding lookup
  // (text tokens) merged with the projected vision features (image tokens).
  const float* inputs_embeds = nullptr;

  // Number of query positions in this step (>= 1).
  int q_len = 0;

  // Hidden size of the model (Qwen2.5: e.g. 3584 for 7B; the engine validates
  // against what the loaded engine expects).
  int hidden = 0;

  // Optional additive attention mask, shape [1, 1, q_len, kv_len + q_len],
  // row-major over (q, kv). Values are added to the attention logits before
  // softmax: 0.0 to attend, a large negative (e.g. -inf / -1e30f) to mask.
  //
  // If null, the engine MUST apply its built-in causal mask over the
  // [kv_len + q_len] keys for the q_len queries (standard left-to-right). The
  // null case is the Phase 2 NTP/AR path. A non-null mask is used by PBD
  // (Phases 3-4) and is only guaranteed to be honoured by the StandaloneTrt
  // backend; see make_llm_engine() / the per-backend docs.
  const float* attn_mask_4d = nullptr;

  // Number of keys already in the KV cache *before* this step (i.e. the number
  // of previously-processed positions). Total key length seen by attention is
  // kv_len + q_len. Must equal kv_len() of the engine at call time.
  int kv_len = 0;

  // Position ids for the q_len query positions, shape [q_len]. For plain causal
  // generation these are [kv_len, kv_len+1, ...]; for PBD windows they may be
  // non-contiguous (parallel slots share a base position). Supplied explicitly
  // so RoPE is identical to the reference implementation.
  const int64_t* position_ids = nullptr;
};

// ---------------------------------------------------------------------------
// Per-step output.
// ---------------------------------------------------------------------------
struct StepOutput {
  // Logits, row-major shape [rows, vocab] where rows == q_len. Implementations
  // return logits for ALL q_len query positions; callers that only need the
  // last row (NTP greedy/sampling) index the final row. Returning all rows is
  // required for MTP/PBD where every window slot produces a token.
  std::vector<float> logits;

  // Vocabulary size (number of columns in `logits`).
  int vocab = 0;
};

// ---------------------------------------------------------------------------
// Configuration passed to the factory.
// ---------------------------------------------------------------------------
struct LlmEngineConfig {
  // Which backend to construct.
  Backend backend = Backend::TrtLlm;

  // Primary engine artifact path.
  //   * TrtLlm        : path to the TRT-LLM engine directory (the folder
  //                     containing the serialized engine + config produced by
  //                     trtllm-build). May also be a single engine file
  //                     depending on the runtime version.
  //   * StandaloneTrt : path to the serialized .engine (or .plan) file for the
  //                     standalone Qwen2.5 decoder.
  std::string engine_path;

  // Optional secondary path. Reserved for backends that split artifacts (e.g. a
  // separate tokenizer/config json, or a second engine). Empty if unused.
  std::string aux_path;

  // Expected hidden size; if > 0 the engine validates the loaded engine against
  // it at load time and throws on mismatch. If 0, taken from the engine.
  int hidden = 0;

  // Expected vocab size; if > 0 validated at load time. If 0, taken from engine.
  int vocab = 0;

  // Maximum sequence length to provision KV cache / context for. If 0, the
  // backend uses the engine's own max.
  int max_seq_len = 0;

  // CUDA device ordinal to bind the engine to.
  int device_id = 0;
};

// ---------------------------------------------------------------------------
// Exception type thrown by TRT-free stub builds and by load/runtime failures.
// ---------------------------------------------------------------------------
class LlmEngineError : public std::runtime_error {
 public:
  explicit LlmEngineError(const std::string& what) : std::runtime_error(what) {}
};

// ---------------------------------------------------------------------------
// Abstract engine interface.
// ---------------------------------------------------------------------------
//
// Lifetime / threading: a single ILlmEngine instance is NOT thread-safe and
// owns one KV cache for one in-flight batch=1 sequence. Construct via
// make_llm_engine().
class ILlmEngine {
 public:
  virtual ~ILlmEngine() = default;

  // Run one decoder forward pass. See StepInput / StepOutput. On success the KV
  // cache grows by in.q_len (kv_len() increases by in.q_len). Throws
  // LlmEngineError on any failure.
  virtual StepOutput step(const StepInput& in) = 0;

  // Drop all cached keys/values; next step() is a fresh prefill (kv_len()==0).
  virtual void reset_kv() = 0;

  // Number of keys currently in the KV cache (tokens processed so far).
  virtual int kv_len() const = 0;

  // Truncate the KV cache back to exactly `to_len` keys. Required by PBD
  // (Phase 3): after speculating a window we accept a prefix and roll the cache
  // back to the accepted length before re-stepping. Precondition:
  // 0 <= to_len <= kv_len(). Throws LlmEngineError otherwise.
  virtual void rollback_kv(int to_len) = 0;

  // Which backend produced this instance (diagnostic).
  virtual Backend backend() const = 0;

  // Whether this instance honours a non-null StepInput::attn_mask_4d. The
  // decode loop can query this to decide whether it may rely on a dense 4D mask
  // (PBD) or must restrict itself to causal generation. StandaloneTrt returns
  // true; TrtLlm returns false (it falls back to built-in causal) unless a
  // future runtime is detected to support it.
  virtual bool supports_dense_4d_mask() const = 0;
};

// ---------------------------------------------------------------------------
// Factory.
// ---------------------------------------------------------------------------
//
// Constructs the engine selected by cfg.backend.
//
// In a TRT-free build (LA_BUILD_TRT not defined) the only linked translation
// unit is the stub factory, which throws LlmEngineError from this function so
// that the symbol exists and callers/tests link, but any attempt to actually
// build an engine fails loudly. The pure-C++ decode/sampling/parse code never
// calls this in unit tests.
std::unique_ptr<ILlmEngine> make_llm_engine(const LlmEngineConfig& cfg);

}  // namespace llm
}  // namespace la

#endif  // LA_LLM_LLMENGINE_H
