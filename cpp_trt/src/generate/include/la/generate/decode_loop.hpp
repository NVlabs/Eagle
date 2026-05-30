// SPDX-License-Identifier: Apache-2.0
//
// la/generate/decode_loop.hpp — the NTP / AR greedy decode driver (spec §5.4,
// the slow / pure-AR path: the Phase-2 internal correctness checkpoint).
//
// Ported from the slow/AR path of
//   Embodied/eaglevl/utils/locany/  generate() + generate_utils.py
// (greedy argmax + repetition penalty per the locked decision; spec §5.4).
//
// This file is TRT-FREE: the LLM forward is the abstract la::llm::ILlmEngine
// interface (the real TRT-LLM / standalone-TRT impls live behind LA_BUILD_TRT;
// tests use a host mock), and the embedding lookup is a host callback. Phase 3
// (PBD / MTP fast mode) and Phase 4 (hybrid MTP<->AR) branch from the documented
// points below; only `Slow` (pure AR) is implemented here.

#ifndef LA_GENERATE_DECODE_LOOP_HPP
#define LA_GENERATE_DECODE_LOOP_HPP

#include <cstdint>
#include <functional>
#include <vector>

#include "la/config/token_ids.hpp"
#include "la/llm/LlmEngine.h"

namespace la::generate {

using TokenIds = ::la::config::TokenIds;

enum class GenerationMode {
    Slow,    // pure AR (implemented here — Phase 2)
    Fast,    // pure MTP / PBD       (Phase 3 — NOT implemented here)
    Hybrid,  // MTP with AR fallback (Phase 4 — NOT implemented here)
};

// Host embedding lookup: given a token id, write `hidden` floats into `out` (the
// embedding-table row). The real table comes from the engine/checkpoint; tests
// supply a deterministic stub. Keeping this a callback is what lets this file
// stay TRT-free (spec requirement).
using EmbedLookup =
    std::function<void(std::int64_t token_id, float* out, std::int64_t hidden)>;

struct DecodeOptions {
    GenerationMode mode = GenerationMode::Slow;
    float repetition_penalty = 1.1f;     // spec §5.4 / generate_utils path
    std::int64_t max_new_tokens = 2048;  // spec §5.4
    // Upper bound on committed length (spec §5.4 stop: committed >=
    // min(model_max_length, S + max_new_tokens)). 0 disables the cap (only
    // max_new_tokens applies).
    std::int64_t model_max_length = 0;
    // Model hidden size for the per-step embedding buffer. If 0, taken from the
    // first StepOutput is not possible (we need it before the first step), so
    // the caller MUST set this (or it defaults to the prompt embeddings' hidden
    // via the prefill — see run_ar_decode docs). Required > 0.
    std::int64_t hidden = 0;
};

struct DecodeResult {
    // Generated ids (NOT including the prefill prompt ids), in commit order.
    std::vector<std::int64_t> generated_ids;
    bool hit_im_end = false;       // stopped on im_end (vs the length cap)
    std::int64_t steps = 0;        // engine forwards performed
};

// Run greedy AR decode.
//
// `engine`        : the already-prefilled la::llm::ILlmEngine. This function does
//                   NOT prefill — the caller runs the prefill step over
//                   inputs_embeds[S,H] (with spliced vision embeds), leaving the
//                   engine KV-cache populated with `prompt_len` keys
//                   (engine.kv_len() == prompt_len), and passes the prefill's
//                   last-position logits in `prefill_logits`.
// `prefill_logits`: logits of the LAST prefill position, length == vocab. The
//                   first generated token is argmax of these (after repetition
//                   penalty over the prompt ids). Mirrors HF generate.
// `prompt_ids`    : full prompt token ids (length S). Seeds the rep-penalty
//                   "seen" set and the committed KV length / first position id.
// `tok`           : token id constants (la::config::TokenIds).
// `embed`         : host embedding lookup for the single new token each AR step.
// `opts`          : decode options. opts.hidden MUST be > 0 (the model hidden
//                   size); greedy + repetition penalty only.
//
// Loop (spec §5.4 slow path), per step:
//   apply_repetition_penalty(logits, seen, penalty); id = greedy_argmax(logits)
//   append id; seen += id
//   if id == im_end -> stop (hit_im_end)
//   if committed+1 >= cap -> stop
//   embed(id); StepInput{inputs_embeds, q_len=1, hidden, attn_mask=nullptr
//     (engine built-in causal — the NTP/AR path; see LlmEngine.h),
//     kv_len=committed, position_ids=[committed]}
//   out = engine.step(in); committed += 1; logits = out.logits last row
//
// KV model (la/llm/LlmEngine.h): step() appends q_len keys, kv_len() grows by 1.
// In pure AR every appended key is permanent, so NO rollback_kv is needed. The
// MTP path (Phase 3) speculates a 6-wide window then rollback_kv() back to the
// accepted length before re-stepping.
//
// MASK: the NTP/AR path passes attn_mask_4d == nullptr so the engine applies its
// built-in causal mask (LlmEngine.h: the null case IS the Phase-2 NTP/AR path).
// la::decode::build_ar_mask is therefore NOT needed at runtime for AR; it exists
// for the standalone-TRT backend / bit-parity tests. (Phase 3 PBD MUST pass an
// explicit dense 4D mask from la::decode::build_mtp_window_mask.)
//
// === PHASE 3 / PHASE 4 BRANCH POINTS (documented, not implemented) ===
//   * Fast (MTP/PBD): instead of q_len=1, build the MTP input
//       [committed..., last_committed(dup), mask x5] (q_len=6); set
//       in.attn_mask_4d = la::decode::build_mtp_window_mask(kv_len, block=6)
//       (requires engine.supports_dense_4d_mask()); set in.position_ids =
//       la::decode::build_position_ids_mtp(start, total, n_future=6) (the -1
//       shift); slice out.logits[-6:]; run decode_bbox_avg / is_valid_box_frame
//       / handle_pattern; commit 6/4/3 tokens; engine.rollback_kv(accepted_len).
//   * Hybrid: start Fast; on handle_pattern -> error_box set use_mtp=false and
//       commit only the verified coord prefix; in AR, box_end -> use_mtp=true to
//       resume MTP; im_end -> stop.
//   These branch at the marked location in decode_loop.cpp.
DecodeResult run_ar_decode(::la::llm::ILlmEngine& engine,
                           const std::vector<float>& prefill_logits,
                           const std::vector<std::int64_t>& prompt_ids,
                           const TokenIds& tok,
                           const EmbedLookup& embed,
                           const DecodeOptions& opts);

}  // namespace la::generate

#endif  // LA_GENERATE_DECODE_LOOP_HPP
