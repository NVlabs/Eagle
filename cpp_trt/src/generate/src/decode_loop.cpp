// SPDX-License-Identifier: Apache-2.0
//
// la/generate/decode_loop.cpp — see decode_loop.hpp.

#include "la/generate/decode_loop.hpp"

#include <algorithm>
#include <cstddef>

#include "la/generate/sampling.hpp"

namespace la::generate {

DecodeResult run_ar_decode(::la::llm::ILlmEngine& engine,
                           const std::vector<float>& prefill_logits,
                           const std::vector<std::int64_t>& prompt_ids,
                           const TokenIds& tok,
                           const EmbedLookup& embed,
                           const DecodeOptions& opts) {
    DecodeResult result;

    const std::int64_t hidden = opts.hidden;
    const std::int64_t prompt_len = static_cast<std::int64_t>(prompt_ids.size());

    // Stop cap (spec §5.4): committed >= min(model_max_length, S + max_new).
    std::int64_t cap = prompt_len + opts.max_new_tokens;
    if (opts.model_max_length > 0) {
        cap = std::min(cap, opts.model_max_length);
    }

    // Seed the rep-penalty "seen" set with the prompt ids. The python applies
    // the penalty over the running set of *generated* ids; the reference
    // generate() accumulates from the prompt onward, so we seed with the prompt.
    std::vector<std::int64_t> seen = prompt_ids;

    // KV length currently materialized in the engine; == prompt_len after the
    // caller's prefill (engine.kv_len()).
    std::int64_t committed = prompt_len;

    // Working logit row for the next-token decision; starts as the prefill's
    // last-position logits (the first generated token comes from these).
    std::vector<float> logits = prefill_logits;

    // Reusable per-step embedding buffer for the single new token.
    std::vector<float> embed_buf(
        static_cast<std::size_t>(hidden > 0 ? hidden : 0));

    while (true) {
        // ---- Sampling: repetition penalty then greedy argmax (spec §5.4) ----
        apply_repetition_penalty(logits, seen, opts.repetition_penalty);
        const std::int64_t next_id = greedy_argmax(logits);
        if (next_id < 0) {
            break;  // empty logits — defensive.
        }

        result.generated_ids.push_back(next_id);
        seen.push_back(next_id);

        // Terminal token (spec §5.4 stop: im_end -> break).
        if (next_id == tok.im_end) {
            result.hit_im_end = true;
            break;
        }

        // Length cap (committed length after we would append this token).
        if (committed + 1 >= cap) {
            break;
        }

        if (hidden <= 0) {
            break;  // misconfigured: cannot embed the next token.
        }

        // ---- Build the next forward over the just-emitted token (q_len=1) ----
        // Embedding lookup (host callback — keeps this loop TRT-free).
        embed(next_id, embed_buf.data(), hidden);

        // AR step (spec §5.4): single new query at absolute position `committed`.
        // attn_mask_4d == nullptr -> engine applies its built-in causal mask
        // (LlmEngine.h: the null case IS the Phase-2 NTP/AR path).
        const std::int64_t pos = committed;

        ::la::llm::StepInput in;
        in.inputs_embeds = embed_buf.data();
        in.q_len = 1;
        in.hidden = static_cast<int>(hidden);
        in.attn_mask_4d = nullptr;             // built-in causal (AR)
        in.kv_len = static_cast<int>(committed);  // keys before this step
        in.position_ids = &pos;

        // =========================================================
        // PHASE 3 / 4 BRANCH POINT:
        //   if (opts.mode == Fast || (opts.mode == Hybrid && use_mtp)) {
        //       build the 6-wide MTP input here (dup last + 5 mask tokens),
        //       in.attn_mask_4d = build_mtp_window_mask(committed+6, 6).data()
        //         (requires engine.supports_dense_4d_mask()),
        //       in.position_ids = build_position_ids_mtp(start, total, 6).data(),
        //       slice out.logits[-6:], run decode_bbox_avg / handle_pattern,
        //       commit 6/4/3, engine.rollback_kv(accepted_len).
        //   }
        // The pure-AR path below is the Phase-2 slow checkpoint.
        // =========================================================

        ::la::llm::StepOutput out = engine.step(in);
        ++result.steps;
        committed += 1;  // engine appended one key (kv_len() grew by 1)

        // out.logits is [rows, vocab] with rows == q_len == 1; take the last
        // (only) row of `vocab` logits. (Handles rows>1 defensively.)
        const std::int64_t vocab = out.vocab;
        if (vocab <= 0 ||
            out.logits.size() < static_cast<std::size_t>(vocab)) {
            break;  // defensive: malformed engine output.
        }
        const std::size_t last_row =
            out.logits.size() - static_cast<std::size_t>(vocab);
        logits.assign(
            out.logits.begin() + static_cast<std::ptrdiff_t>(last_row),
            out.logits.end());
    }

    return result;
}

}  // namespace la::generate
