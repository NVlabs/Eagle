// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "la/generate/coord.hpp"

namespace la::generate {

double coord_token_to_value(std::int64_t id, std::int64_t coord_start) noexcept {
    return static_cast<double>(id - coord_start) / 1000.0;
}

double value_to_pixel(double value, const ResizeTransform& t) noexcept {
    const double canvas_px = value * t.canvas_dim;
    const double anchored = canvas_px - t.offset;
    if (t.scale == 0.0) {
        return anchored;
    }
    return anchored / t.scale;
}

double coord_token_to_pixel(std::int64_t id, std::int64_t coord_start,
                            const ResizeTransform& t) noexcept {
    return value_to_pixel(coord_token_to_value(id, coord_start), t);
}

}  // namespace la::generate
