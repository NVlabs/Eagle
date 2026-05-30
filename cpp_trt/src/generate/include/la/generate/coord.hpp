// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef LA_GENERATE_COORD_HPP
#define LA_GENERATE_COORD_HPP

#include <cstdint>

// Coordinate detokenization, spec §6 "Output decode".
//
// A coordinate token id in the contiguous band [coord_start, coord_end] maps to
// a normalized value: value = (id - coord_start) / 1000.0. This value is in
// [0, 1] relative to the resized+padded canvas the model saw; mapping back to
// original-image pixels is the inverse of the preprocessing resize transform.
//
// NOTE: la::preprocess (la/preprocess/resize.hpp) owns the canonical resize
// transform used by the forward pipeline. We keep a small local ResizeTransform
// here so la_generate stays self-contained (testable with no sibling). The
// integrator wires la::preprocess's scale/canvas/offset into this struct; the
// inverse-mapping math is the single source of truth for coord->pixel.

namespace la::generate {

// Sentinel coord id meaning "uncertain coordinate" (spec §5.4 hybrid ambiguity
// rule emits id 0). Callers should treat a parsed coordinate of this id as
// missing/uncertain.
inline constexpr std::int64_t kUncertainCoordId = 0;

// value = (id - coord_start) / 1000.0  (spec §6, decode_bbox path).
// `id` is assumed to already be a coordinate token in the band; callers gate on
// TokenIds::is_coord(). For the uncertain sentinel (id 0) this returns a
// negative value; callers should check kUncertainCoordId before mapping.
[[nodiscard]] double coord_token_to_value(std::int64_t id,
                                          std::int64_t coord_start) noexcept;

// Inverse pixel mapping. The preprocessing pipeline (spec §6) does:
//   resized = round(orig * scale)   (BICUBIC), then ceil-to-28 padding.
// A model coordinate is per-mille of the *padded canvas*. To recover the
// original-image pixel we strip the canvas scaling.
//
// Given the normalized model value v in [0,1], the padded-canvas dimension
// `canvas_dim` (px) along that axis, and the resize `scale` that maps original
// -> resized (resized_dim = orig_dim * scale), the original-image pixel is:
//
//     canvas_px = v * canvas_dim          // pixel on the padded canvas
//     orig_px   = (canvas_px - offset) / scale  // undo pad anchor + resize
//
// `offset` allows a non-zero top-left pad anchor if a future preprocessing
// variant centers the image. Default 0 matches the current top-left-anchored
// ceil-to-28 padding.
struct ResizeTransform {
    double scale = 1.0;        // orig_dim * scale == resized_dim
    double canvas_dim = 0.0;   // padded canvas size (px) along this axis
    double offset = 0.0;       // top-left pad offset (px), usually 0
};

// Map a normalized model value (0..1, per-mille/1000) to an original-image
// pixel coordinate along one axis.
[[nodiscard]] double value_to_pixel(double value,
                                    const ResizeTransform& t) noexcept;

// Convenience: coord token id -> original-image pixel along one axis.
[[nodiscard]] double coord_token_to_pixel(std::int64_t id,
                                          std::int64_t coord_start,
                                          const ResizeTransform& t) noexcept;

}  // namespace la::generate

#endif  // LA_GENERATE_COORD_HPP
