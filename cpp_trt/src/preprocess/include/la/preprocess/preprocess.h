// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// LocateAnything image-preprocessing module (C++ port).
// Ported from Embodied/eaglevl/utils/locany/image_processing_locateanything.py

#ifndef LA_PREPROCESS_PREPROCESS_H_
#define LA_PREPROCESS_PREPROCESS_H_

#include <cstdint>
#include <vector>

namespace la {
namespace preprocess {

struct PreprocessConfig {
  int patch_size = 14;
  int merge_kernel_h = 2;
  int merge_kernel_w = 2;
  int in_token_limit = 25600;  // preprocessor_config.json value (not the 4096 class default)
  float mean = 0.5f;
  float std = 0.5f;
  std::uint8_t rgba_bg_r = 255;
  std::uint8_t rgba_bg_g = 255;
  std::uint8_t rgba_bg_b = 255;
};

struct ResizeTransform {
  int orig_w = 0;
  int orig_h = 0;
  int shrunk_w = 0;
  int shrunk_h = 0;
  int target_w = 0;
  int target_h = 0;
  float scale_x = 1.0f;  // x_orig = x_input * scale_x
  float scale_y = 1.0f;  // y_orig = y_input * scale_y
  bool shrunk = false;
};

struct PreprocessResult {
  std::vector<float> pixel_values;  // (L,3,patch,patch) row-major
  std::vector<int> grid_hws;        // (1,2): {grid_h, grid_w}
  int num_patches = 0;
  int grid_h = 0;
  int grid_w = 0;
  int patch_size = 14;
  ResizeTransform transform;
};

bool ComputeTargetSize(int orig_w, int orig_h, const PreprocessConfig& cfg,
                       ResizeTransform* out, int* grid_h, int* grid_w,
                       int* num_patches);

bool PreprocessCPU(const std::uint8_t* data, int width, int height,
                   int channels, const PreprocessConfig& cfg,
                   PreprocessResult* out);

#ifdef LA_BUILD_CUDA
bool PreprocessCUDA(const std::uint8_t* data, int width, int height,
                    int channels, const PreprocessConfig& cfg,
                    PreprocessResult* out);
#endif

}  // namespace preprocess
}  // namespace la

#endif  // LA_PREPROCESS_PREPROCESS_H_
