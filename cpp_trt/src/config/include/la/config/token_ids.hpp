// SPDX-License-Identifier: Apache-2.0
//
// la/config/token_ids.hpp
//
// Token-id configuration for the LocateAnything C++ port (Phase 0).
//
// PURPOSE
// -------
// The LocateAnything model extends a base LLM tokenizer with a set of
// "indicative" special tokens (image placeholder, box/ref delimiters, a
// contiguous band of 1001 coordinate tokens, etc.). The numeric ids of these
// tokens are NOT fixed by the architecture: they are assigned at
// `tokenizer.add_tokens(...)` time when the checkpoint is built, and therefore
// live in the released checkpoint's `config.json` / `tokenizer_config.json` /
// `added_tokens.json`. Downstream C++ inference code MUST read them from the
// checkpoint at runtime rather than baking magic numbers into the binary.
//
// This module provides:
//   * `TokenIds`            - a plain struct holding every id the pipeline needs.
//   * `LoadTokenIds(...)`   - parse a JSON config (std-only, no external deps)
//                             to populate a `TokenIds`.
//   * `TokenIds::Validate()`- assert the invariants the rest of the pipeline
//                             relies on, in particular that the coordinate band
//                             is contiguous and spans exactly 1001 ids
//                             (coord_end - coord_start == 1000).
//   * `TokenIds::Fallback()`- the indicative ids from spec Section 5.5, provided
//                             ONLY as a documented last-resort default for
//                             environments where the checkpoint config is not
//                             available. These are NOT authoritative.
//
// PORTED FROM (conceptually):
//   Embodied/eaglevl/train/constants.py            (token name constants)
//   Embodied/eaglevl/utils/locany/*                (tokenizer/config handling)
//   Embodied/eaglevl/model/locany/*                (id consumption)
//   Spec Section 5.5                               (indicative id table)
//
// NOTE: In the build environment used to author this file the Python reference
// tree was not readable, so the canonical JSON key names below are the ones the
// spec documents plus the conventional HuggingFace special-token spellings.
// `LoadTokenIds` is intentionally tolerant of several key spellings so it can
// bind to the real checkpoint config without code changes; see the loader docs.

#ifndef LA_CONFIG_TOKEN_IDS_HPP
#define LA_CONFIG_TOKEN_IDS_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace la::config {

// A token id. -1 means "unset / not present in the source config".
using token_id_t = std::int64_t;

inline constexpr token_id_t kUnsetTokenId = -1;

// Number of *intervals* between coord_start and coord_end. The coordinate band
// therefore contains kCoordBandSpan + 1 == 1001 distinct ids:
//   coord_start, coord_start+1, ..., coord_end  (inclusive)
// representing normalized coordinates 0/1000, 1/1000, ..., 1000/1000.
inline constexpr token_id_t kCoordBandSpan = 1000;

// Holds every special-token id the LocateAnything inference pipeline needs.
//
// All members default to kUnsetTokenId so a partially populated config is
// detectable. Use Validate() (or LoadTokenIds with require_all=true) to enforce
// completeness before the struct is handed to the inference pipeline.
struct TokenIds {
  // Image placeholder token (a.k.a. image_token_index). One per image patch
  // slot in the prompt; replaced by image embeddings at runtime.
  token_id_t image_token = kUnsetTokenId;

  // Bounding-box span delimiters: <box> ... </box>.
  token_id_t box_start = kUnsetTokenId;
  token_id_t box_end = kUnsetTokenId;

  // Referring-expression span delimiters: <ref> ... </ref>.
  token_id_t ref_start = kUnsetTokenId;
  token_id_t ref_end = kUnsetTokenId;

  // Segmentation-mask token.
  token_id_t mask = kUnsetTokenId;

  // Coordinate band delimiters. The ids in the inclusive range
  // [coord_start, coord_end] are the 1001 quantized coordinate tokens.
  // Invariant (enforced by Validate): coord_end - coord_start == 1000.
  token_id_t coord_start = kUnsetTokenId;
  token_id_t coord_end = kUnsetTokenId;

  // "none" sentinel (e.g. for an absent/empty answer). Note: in the indicative
  // table this id is small (4064), i.e. it lives in the *base* vocab, not in
  // the added-token range -- do not assume it is contiguous with the others.
  token_id_t none = kUnsetTokenId;

  // Explicit null sentinel for the LocateAnything heads.
  token_id_t null_token = kUnsetTokenId;

  // End-of-message / end-of-sequence (im_end == eos for this family).
  token_id_t im_end = kUnsetTokenId;

  // Task/mode switch token.
  token_id_t switch_token = kUnsetTokenId;

  // ---- Helpers ------------------------------------------------------------

  // True iff `id` is one of the 1001 ids in the inclusive coordinate band.
  // Returns false if the band is unset.
  bool IsCoordToken(token_id_t id) const noexcept {
    return coord_start != kUnsetTokenId && coord_end != kUnsetTokenId &&
           id >= coord_start && id <= coord_end;
  }

  // Map a coordinate token id to its quantized bin index in [0, 1000].
  // Returns -1 if `id` is not in the band.
  token_id_t CoordBin(token_id_t id) const noexcept {
    return IsCoordToken(id) ? (id - coord_start) : kUnsetTokenId;
  }

  // Map a quantized bin index in [0, 1000] back to its token id.
  // Returns -1 if the band is unset or `bin` is out of range.
  token_id_t CoordTokenForBin(token_id_t bin) const noexcept {
    if (coord_start == kUnsetTokenId || bin < 0 || bin > kCoordBandSpan) {
      return kUnsetTokenId;
    }
    return coord_start + bin;
  }

  // True iff every member has been assigned (no kUnsetTokenId remains).
  bool IsComplete() const noexcept;

  // Assert the pipeline invariants. Throws std::runtime_error on violation:
  //   * coord_start and coord_end must both be set;
  //   * coord_end - coord_start == 1000 (the band spans exactly 1001 ids);
  //   * coord_start >= 0 (band is non-negative / contiguous & well-formed);
  //   * box/ref start ids precede their matching end ids when both are set.
  // If `require_all` is true, also asserts IsComplete().
  void Validate(bool require_all = false) const;

  // The indicative ids documented in spec Section 5.5.
  // These are a *fallback only*; prefer LoadTokenIds() against the real
  // checkpoint config. Kept here so callers have a deterministic default and so
  // the documented values live in exactly one place.
  static TokenIds Fallback() noexcept;
};

// Parse a JSON document (text) and populate a TokenIds.
//
// Accepted shapes (the loader is deliberately permissive so it can bind to the
// real checkpoint without source changes):
//
//   1. A flat object mapping the canonical field names to integer ids, e.g.
//        { "image_token_index": 151667, "box_start": 151668, ... }
//      Several key spellings are accepted per field (see .cpp for the table),
//      including HuggingFace special-token spellings such as "<box>", "<ref>".
//
//   2. An object with a nested "token_ids" (or "locany_token_ids") object of
//      the same flat form; the nested object is used if present.
//
// `text` need not be a complete checkpoint config; unknown keys are ignored, so
// passing the whole config.json is fine.
//
// Throws std::runtime_error on malformed JSON. Does NOT throw on missing
// fields unless `require_all` is true (in which case Validate(require_all=true)
// is invoked before returning). When `require_all` is false the returned struct
// may contain kUnsetTokenId members; callers should call Validate() themselves.
TokenIds LoadTokenIds(std::string_view json_text, bool require_all = false);

// Convenience: read a JSON file from `path` and parse it as above.
// Throws std::runtime_error if the file cannot be opened.
TokenIds LoadTokenIdsFromFile(const std::string& path, bool require_all = false);

}  // namespace la::config

#endif  // LA_CONFIG_TOKEN_IDS_HPP
