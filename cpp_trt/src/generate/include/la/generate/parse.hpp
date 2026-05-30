// SPDX-License-Identifier: Apache-2.0
//
// la/generate/parse.hpp — output-stream parsing (spec §6), pure C++.
//
// The decode loop produces a flat sequence of token ids. We scan it for
// <ref>...</ref> labels and <box>...</box> frames and classify each box by its
// integer coordinate-group count: 4 groups -> Box (xyxy), 2 groups -> Point
// (xy), <box>none</box> -> None (spec §6). This is the id-native scanner
// (preferred: we have ids, not text — spec §6). A text fallback parser matching
// the worker's `<box><x1><y1><x2><y2></box>` regex is also provided for
// golden-string tests.
//
// Ported from the worker box/point distinction in
//   Embodied/eaglevl/utils/locany/processing_locateanything.py  and the box FSM
//   in generate_utils.py (handle_pattern: coord_box=4 / point_box=2 / empty=none).
//
// COORDINATE ORDERING (spec §6, Phase-0 item): decode_bbox_avg comments say
// x1,x2,y1,y2 while handle_pattern / the template / the worker regex imply
// <x1><y1><x2><y2> (xyxy). The model does not reorder. This parser preserves the
// on-wire order verbatim into `coords_permille` and does NOT reinterpret it; the
// caller decides the final xyxy mapping once the convention is confirmed in
// Phase 0. We document but do not bake the assumption.

#ifndef LA_GENERATE_PARSE_HPP
#define LA_GENERATE_PARSE_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "la/config/token_ids.hpp"

namespace la::generate {

// The canonical token-id struct is owned by la::config (spec §5.5). la_generate
// consumes it directly so there is a single source of truth.
using TokenIds = ::la::config::TokenIds;

enum class DetectionKind {
    None,   // <box>none</box>
    Box,    // 4 coordinate groups
    Point,  // 2 coordinate groups
};

struct Detection {
    std::optional<std::string> label;  // text from the preceding <ref>...</ref>
    DetectionKind kind = DetectionKind::None;
    // Coordinates as per-mille integers in [0,1000] (token_id - coord_start),
    // preserved in on-wire order. Empty for None. The sentinel value 0
    // (kUncertainCoordId in coord.hpp) may appear for hybrid-uncertain slots.
    std::vector<std::int64_t> coords_permille;
};

// Raw ref ids captured for a detection (caller decodes them to text via the
// tokenizer; la_generate does not depend on the tokenizer).
struct RefLabel {
    std::vector<std::int64_t> ids;  // ids strictly between <ref> and </ref>
};

struct ParseResult {
    std::vector<Detection> detections;
};

// Scan an output token-id stream into structured detections.
// A <ref>...</ref> immediately preceding a <box>...</box> attaches as its label.
// `<box>none</box>` -> Detection{kind=None}. A box with 4 coord groups -> Box;
// with 2 -> Point. Coordinate tokens are converted to per-mille via
// (id - coord_start) using TokenIds::CoordBin. Non-coord, non-structural ids
// inside a box are ignored (matches the FSM's "take coord tokens" behavior).
// Scanning stops at im_end.
//
// `label_ids_out` (optional, may be null) receives, parallel to detections, the
// raw ref ids (empty if no label) so the caller can run the tokenizer.
ParseResult parse_token_stream(const std::vector<std::int64_t>& ids,
                               const TokenIds& tok,
                               std::vector<RefLabel>* label_ids_out = nullptr);

// Text fallback. Parses a decoded string for the worker's box/point pattern.
// Mirrors the worker regex family: an optional <ref>label</ref> prefix, then a
// <box>...</box> containing either 4 or 2 <int> coordinate groups, or the
// literal `<box>none</box>`. Integers are interpreted as per-mille [0,1000].
// This exists for golden-string parity tests; the id-native scanner is
// authoritative in production.
ParseResult parse_text(const std::string& text);

}  // namespace la::generate

#endif  // LA_GENERATE_PARSE_HPP
