// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// PIL-faithful BICUBIC resize (CPU reference).
//
// This reproduces Pillow's resize(..., Image.Resampling.BICUBIC) semantics,
// which LocateAnything's rescale() relies on. Getting this exactly right
// matters because downstream bounding-box coordinates are mapped through the
// resize transform.
//
// Pillow algorithm (see Pillow's src/libImaging/Resample.c):
//   * Separable two-pass filtering: horizontal pass then vertical pass.
//   * Bicubic kernel = cubic convolution with a = -0.5 (Catmull-Rom / Keys):
//       k(x) = (a+2)|x|^3 - (a+3)|x|^2 + 1                for |x| <= 1
//       k(x) = a|x|^3 - 5a|x|^2 + 8a|x| - 4a              for 1 < |x| < 2
//       k(x) = 0                                          otherwise
//     The kernel's nominal support radius is 2.0.
//   * Coordinate convention: output sample j maps to input center
//       center = (j + 0.5) * scale,   scale = in_size / out_size.
//   * Antialiasing on DOWNSCALE: filterscale = max(scale, 1.0). The kernel is
//     stretched by filterscale (support *= filterscale) and arguments are
//     divided by filterscale, then weights are normalized to sum to 1. On
//     upscale (scale < 1) filterscale == 1 so the kernel is the plain bicubic.
//   * Taps span [center - support, center + support); xmin/xmax are clipped to
//     [0, in_size). Edge pixels are effectively clamped via clipping + weight
//     renormalization.
//   * Pillow precomputes weights as fixed-point for uint8 images; here we keep
//     float weights and round at the end (round-half-away-from-zero via
//     floor(x + 0.5)) and clamp to [0,255]. This matches Pillow's
//     clip8()/rounding for the uint8 path to within +-1 LSB in rare ties.

#ifndef LA_PREPROCESS_BICUBIC_RESIZE_H_
#define LA_PREPROCESS_BICUBIC_RESIZE_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace la {
namespace preprocess {
namespace detail {

// Bicubic kernel with a = -0.5.
inline double BicubicKernel(double x) {
  const double a = -0.5;
  x = std::fabs(x);
  if (x < 1.0) {
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  }
  if (x < 2.0) {
    return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  }
  return 0.0;
}

// Precomputed per-output-sample filter taps for one axis.
struct AxisFilter {
  // For output index i, taps live at bounds[i].first .. bounds[i].first+ksize[i]-1
  // (input coordinates), with weights at weights[i*max_k + t].
  std::vector<int> start;     // first input index contributing to output i
  std::vector<int> ksize;     // number of taps for output i
  std::vector<double> weights;  // flattened, max_k per output row
  int max_k = 0;
};

// Build PIL bicubic taps for resizing one axis from in_size -> out_size.
inline AxisFilter BuildAxisFilter(int in_size, int out_size) {
  AxisFilter f;
  const double support_radius = 2.0;  // bicubic
  const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
  const double filterscale = scale >= 1.0 ? scale : 1.0;
  const double support = support_radius * filterscale;
  const int max_k = static_cast<int>(std::ceil(support)) * 2 + 1;

  f.max_k = max_k;
  f.start.resize(out_size);
  f.ksize.resize(out_size);
  f.weights.assign(static_cast<std::size_t>(out_size) * max_k, 0.0);

  for (int xx = 0; xx < out_size; ++xx) {
    const double center = (xx + 0.5) * scale;
    double ww = 0.0;
    // PIL: xmin = (int)(center - support + 0.5), clipped to >= 0.
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    // xmax = (int)(center + support + 0.5), clipped to <= in_size.
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    int k = xmax - xmin;
    if (k > max_k) k = max_k;

    double* row = &f.weights[static_cast<std::size_t>(xx) * max_k];
    for (int x = 0; x < k; ++x) {
      const double w = BicubicKernel((x + xmin - center + 0.5) / filterscale);
      row[x] = w;
      ww += w;
    }
    if (ww != 0.0) {
      for (int x = 0; x < k; ++x) row[x] /= ww;
    }
    f.start[xx] = xmin;
    f.ksize[xx] = k;
  }
  return f;
}

inline std::uint8_t Clip8(double v) {
  // PIL clip8 with round-half-away-from-zero.
  double r = std::floor(v + 0.5);
  if (r < 0.0) r = 0.0;
  if (r > 255.0) r = 255.0;
  return static_cast<std::uint8_t>(r);
}

// Resize an interleaved uint8 image (HWC) using PIL bicubic semantics.
// in:  src, src_w x src_h, `channels` interleaved channels.
// out: dst (resized), dst_w x dst_h, same channel count, caller-allocated to
//      dst_w*dst_h*channels.
inline void ResizeBicubicU8(const std::uint8_t* src, int src_w, int src_h,
                            std::uint8_t* dst, int dst_w, int dst_h,
                            int channels) {
  // Pass 1: horizontal resize src(src_w x src_h) -> tmp(dst_w x src_h),
  // accumulating in double to mirror Pillow's two-pass precision.
  const AxisFilter fx = BuildAxisFilter(src_w, dst_w);
  std::vector<double> tmp(static_cast<std::size_t>(dst_w) * src_h * channels);

  for (int y = 0; y < src_h; ++y) {
    const std::uint8_t* srow = src + static_cast<std::size_t>(y) * src_w * channels;
    double* trow = &tmp[static_cast<std::size_t>(y) * dst_w * channels];
    for (int x = 0; x < dst_w; ++x) {
      const int xmin = fx.start[x];
      const int k = fx.ksize[x];
      const double* w = &fx.weights[static_cast<std::size_t>(x) * fx.max_k];
      for (int c = 0; c < channels; ++c) {
        double acc = 0.0;
        for (int t = 0; t < k; ++t) {
          acc += w[t] * static_cast<double>(srow[(xmin + t) * channels + c]);
        }
        trow[x * channels + c] = acc;
      }
    }
  }

  // Pass 2: vertical resize tmp(dst_w x src_h) -> dst(dst_w x dst_h).
  const AxisFilter fy = BuildAxisFilter(src_h, dst_h);
  for (int y = 0; y < dst_h; ++y) {
    const int ymin = fy.start[y];
    const int k = fy.ksize[y];
    const double* w = &fy.weights[static_cast<std::size_t>(y) * fy.max_k];
    std::uint8_t* drow = dst + static_cast<std::size_t>(y) * dst_w * channels;
    for (int x = 0; x < dst_w; ++x) {
      for (int c = 0; c < channels; ++c) {
        double acc = 0.0;
        for (int t = 0; t < k; ++t) {
          acc += w[t] *
                 tmp[(static_cast<std::size_t>(ymin + t) * dst_w + x) * channels + c];
        }
        drow[x * channels + c] = Clip8(acc);
      }
    }
  }
}

}  // namespace detail
}  // namespace preprocess
}  // namespace la

#endif  // LA_PREPROCESS_BICUBIC_RESIZE_H_
