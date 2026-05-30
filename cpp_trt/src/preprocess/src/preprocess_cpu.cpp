// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// Pure-C++ CPU reference for the LocateAnything preprocessing pipeline.
// Ported from Embodied/eaglevl/utils/locany/image_processing_locateanything.py
//   (class LocateAnythingImageProcessor: rescale / to_tensor / normalize /
//    patchify / _preprocess).

#include "la/preprocess/preprocess.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "bicubic_resize.h"

namespace la {
namespace preprocess {

namespace {

// math.ceil(a / b) * b for positive integers, computed without float error.
inline int CeilMul(int a, int b) { return ((a + b - 1) / b) * b; }

// Replicates the integer/scaling math of rescale() WITHOUT touching pixels.
// Sets shrunk_{w,h}, target_{w,h}, scales, grid, L. Returns false on the
// "Exceed pos emb" guard.
bool ComputeGeometry(int orig_w, int orig_h, const PreprocessConfig& cfg,
                     ResizeTransform* xf, int* grid_h, int* grid_w,
                     int* num_patches) {
  const int patch = cfg.patch_size;
  const int pad_h = cfg.merge_kernel_h * patch;
  const int pad_w = cfg.merge_kernel_w * patch;

  xf->orig_w = orig_w;
  xf->orig_h = orig_h;

  int new_w = orig_w;
  int new_h = orig_h;
  xf->shrunk = false;

  // in_token_limit guard: tokens = (w // patch) * (h // patch).
  const long tokens =
      static_cast<long>(orig_w / patch) * static_cast<long>(orig_h / patch);
  if (tokens > cfg.in_token_limit) {
    const double scale =
        std::sqrt(static_cast<double>(cfg.in_token_limit) /
                  static_cast<double>(tokens));
    // Python int() truncates toward zero; values are positive so cast is fine.
    new_w = static_cast<int>(orig_w * scale);
    new_h = static_cast<int>(orig_h * scale);
    xf->shrunk = true;
  }
  xf->shrunk_w = new_w;
  xf->shrunk_h = new_h;

  const int target_w = CeilMul(new_w, pad_w);
  const int target_h = CeilMul(new_h, pad_h);
  xf->target_w = target_w;
  xf->target_h = target_h;

  // Exceed pos emb guard (on the FINAL size, matching the Python check after
  // the second resize): w // patch >= 512 || h // patch >= 512.
  if (target_w / patch >= 512 || target_h / patch >= 512) {
    return false;
  }

  *grid_w = target_w / patch;
  *grid_h = target_h / patch;
  *num_patches = (*grid_w) * (*grid_h);

  // Affine map from final network-input space back to original pixels.
  xf->scale_x = static_cast<float>(orig_w) / static_cast<float>(target_w);
  xf->scale_y = static_cast<float>(orig_h) / static_cast<float>(target_h);
  return true;
}

// Convert an interleaved RGB(A) uint8 buffer to interleaved RGB uint8.
// RGBA -> composite over a solid background (white per spec): out = a*fg +
// (1-a)*bg, alpha in [0,1]. RGB is copied through.
void ToRGB(const std::uint8_t* data, int w, int h, int channels,
           const PreprocessConfig& cfg, std::vector<std::uint8_t>* rgb) {
  rgb->resize(static_cast<std::size_t>(w) * h * 3);
  if (channels == 3) {
    std::memcpy(rgb->data(), data, rgb->size());
    return;
  }
  // channels == 4
  const std::uint8_t bg[3] = {cfg.rgba_bg_r, cfg.rgba_bg_g, cfg.rgba_bg_b};
  for (std::size_t p = 0; p < static_cast<std::size_t>(w) * h; ++p) {
    const std::uint8_t* src = data + p * 4;
    std::uint8_t* dst = rgb->data() + p * 3;
    const double a = src[3] / 255.0;
    for (int c = 0; c < 3; ++c) {
      const double v = a * src[c] + (1.0 - a) * bg[c];
      double r = std::floor(v + 0.5);
      if (r < 0.0) r = 0.0;
      if (r > 255.0) r = 255.0;
      dst[c] = static_cast<std::uint8_t>(r);
    }
  }
}

}  // namespace

bool ComputeTargetSize(int orig_w, int orig_h, const PreprocessConfig& cfg,
                       ResizeTransform* out, int* grid_h, int* grid_w,
                       int* num_patches) {
  if (orig_w <= 0 || orig_h <= 0 || cfg.patch_size <= 0 ||
      cfg.merge_kernel_h <= 0 || cfg.merge_kernel_w <= 0) {
    return false;
  }
  return ComputeGeometry(orig_w, orig_h, cfg, out, grid_h, grid_w, num_patches);
}

bool PreprocessCPU(const std::uint8_t* data, int width, int height,
                   int channels, const PreprocessConfig& cfg,
                   PreprocessResult* out) {
  if (data == nullptr || out == nullptr) return false;
  if (width <= 0 || height <= 0) return false;
  if (channels != 3 && channels != 4) return false;
  if (cfg.patch_size <= 0 || cfg.merge_kernel_h <= 0 || cfg.merge_kernel_w <= 0) {
    return false;
  }

  const int patch = cfg.patch_size;

  // Step 1: RGB conversion (RGBA -> white composite).
  std::vector<std::uint8_t> rgb;
  ToRGB(data, width, height, channels, cfg, &rgb);

  // Step 2+3: geometry (in_token_limit guard + ceil-to-multiple).
  ResizeTransform xf;
  int grid_h = 0, grid_w = 0, L = 0;
  if (!ComputeGeometry(width, height, cfg, &xf, &grid_h, &grid_w, &L)) {
    return false;
  }

  // Apply resizes to reach target_{w,h}. Mirror the Python control flow:
  //   - first BICUBIC shrink to (shrunk_w, shrunk_h) if the guard fired,
  //   - second BICUBIC to (target_w, target_h) if size differs from current.
  std::vector<std::uint8_t> cur = std::move(rgb);
  int cur_w = width, cur_h = height;

  if (xf.shrunk && (xf.shrunk_w != cur_w || xf.shrunk_h != cur_h)) {
    std::vector<std::uint8_t> shr(
        static_cast<std::size_t>(xf.shrunk_w) * xf.shrunk_h * 3);
    detail::ResizeBicubicU8(cur.data(), cur_w, cur_h, shr.data(), xf.shrunk_w,
                            xf.shrunk_h, 3);
    cur = std::move(shr);
    cur_w = xf.shrunk_w;
    cur_h = xf.shrunk_h;
  }

  if (xf.target_w != cur_w || xf.target_h != cur_h) {
    std::vector<std::uint8_t> tgt(
        static_cast<std::size_t>(xf.target_w) * xf.target_h * 3);
    detail::ResizeBicubicU8(cur.data(), cur_w, cur_h, tgt.data(), xf.target_w,
                            xf.target_h, 3);
    cur = std::move(tgt);
    cur_w = xf.target_w;
    cur_h = xf.target_h;
  }

  const int H = cur_h;  // == target_h
  const int W = cur_w;  // == target_w

  // Step 4+5+6 fused: to_tensor ([0,1], HWC->CHW) + normalize (x*2-1) +
  // patchify into (L, 3, patch, patch).
  //
  // patchify reference:
  //   image (C,H,W) -> reshape(C, H/p, p, W/p, p) -> permute(1,3,0,2,4)
  //                 -> view(L, C, p, p),  L = (H/p)*(W/p).
  // Patch ordering is row-major over (gy, gx): patch index l = gy*grid_w + gx.
  // to_tensor + normalize fused: value = (u/255 - mean)/std = a*u + b,
  //   a = (1/255)/std ; b = -mean/std.
  // For mean=std=0.5: a = 2/255, b = -1, i.e. value = (u/255)*2 - 1 in [-1,1].
  const float inv255 = 1.0f / 255.0f;
  const float a = inv255 / cfg.std;
  const float b = -cfg.mean / cfg.std;

  out->pixel_values.assign(static_cast<std::size_t>(L) * 3 * patch * patch, 0.0f);
  float* pv = out->pixel_values.data();

  for (int gy = 0; gy < grid_h; ++gy) {
    for (int gx = 0; gx < grid_w; ++gx) {
      const int l = gy * grid_w + gx;
      float* patch_base =
          pv + static_cast<std::size_t>(l) * 3 * patch * patch;
      for (int c = 0; c < 3; ++c) {
        float* cb = patch_base + static_cast<std::size_t>(c) * patch * patch;
        for (int py = 0; py < patch; ++py) {
          const int sy = gy * patch + py;
          const std::uint8_t* srow =
              cur.data() + (static_cast<std::size_t>(sy) * W) * 3;
          float* orow = cb + static_cast<std::size_t>(py) * patch;
          for (int px = 0; px < patch; ++px) {
            const int sx = gx * patch + px;
            const std::uint8_t u = srow[sx * 3 + c];
            orow[px] = a * static_cast<float>(u) + b;
          }
        }
      }
    }
  }

  out->grid_hws = {grid_h, grid_w};
  out->num_patches = L;
  out->grid_h = grid_h;
  out->grid_w = grid_w;
  out->patch_size = patch;
  out->transform = xf;
  return true;
}

}  // namespace preprocess
}  // namespace la
