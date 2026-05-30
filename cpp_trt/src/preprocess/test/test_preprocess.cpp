// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
//
// Unit tests for the la_preprocess CPU reference.

#include "la/preprocess/preprocess.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

// Pull in the internal bicubic header to test the kernel directly.
#include "bicubic_resize.h"

namespace {

using la::preprocess::ComputeTargetSize;
using la::preprocess::PreprocessConfig;
using la::preprocess::PreprocessCPU;
using la::preprocess::PreprocessResult;
using la::preprocess::ResizeTransform;

// ---------------------------------------------------------------------------
// Bicubic kernel coefficients (a = -0.5, Catmull-Rom).
// ---------------------------------------------------------------------------
TEST(Bicubic, KernelValues) {
  using la::preprocess::detail::BicubicKernel;
  EXPECT_NEAR(BicubicKernel(0.0), 1.0, 1e-12);
  // k(1) should be 0 for both branches meeting.
  EXPECT_NEAR(BicubicKernel(1.0), 0.0, 1e-12);
  EXPECT_NEAR(BicubicKernel(2.0), 0.0, 1e-12);
  EXPECT_NEAR(BicubicKernel(3.0), 0.0, 1e-12);
  // k(0.5): (a+2)x^3-(a+3)x^2+1 with a=-0.5 -> 1.5*0.125 - 2.5*0.25 + 1 = 0.5625
  EXPECT_NEAR(BicubicKernel(0.5), 0.5625, 1e-12);
  // k(1.5): a x^3 -5a x^2 +8a x -4a, a=-0.5, x=1.5 ->
  //   -0.5*3.375 +2.5*2.25 -4*1.5 +2 = -1.6875+5.625-6+2 = -0.0625
  EXPECT_NEAR(BicubicKernel(1.5), -0.0625, 1e-12);
  // Symmetry.
  EXPECT_NEAR(BicubicKernel(0.7), BicubicKernel(-0.7), 1e-12);
}

// Filter weights for any output sample must sum to ~1.
TEST(Bicubic, WeightsNormalized) {
  for (int in : {10, 33, 100, 7}) {
    for (int out : {5, 28, 56, 13}) {
      auto f = la::preprocess::detail::BuildAxisFilter(in, out);
      for (int i = 0; i < out; ++i) {
        double s = 0.0;
        const double* w = &f.weights[static_cast<size_t>(i) * f.max_k];
        for (int t = 0; t < f.ksize[i]; ++t) s += w[t];
        EXPECT_NEAR(s, 1.0, 1e-9) << "in=" << in << " out=" << out << " i=" << i;
      }
    }
  }
}

// Resizing to the same size should be (near) identity for a flat image and
// preserve a constant image exactly.
TEST(Bicubic, ConstantImagePreserved) {
  const int w = 16, h = 12, c = 3;
  std::vector<uint8_t> src(static_cast<size_t>(w) * h * c, 137);
  std::vector<uint8_t> dst(static_cast<size_t>(w) * h * c, 0);
  la::preprocess::detail::ResizeBicubicU8(src.data(), w, h, dst.data(), w, h, c);
  for (auto v : dst) EXPECT_EQ(v, 137);
  // Upscale a constant image: still constant.
  std::vector<uint8_t> up(static_cast<size_t>(32) * 24 * c, 0);
  la::preprocess::detail::ResizeBicubicU8(src.data(), w, h, up.data(), 32, 24, c);
  for (auto v : up) EXPECT_EQ(v, 137);
}

// ---------------------------------------------------------------------------
// Geometry: ComputeTargetSize matches the Python rescale() integer math.
// ---------------------------------------------------------------------------
TEST(Geometry, NoShrinkCeilToMultiple) {
  // Small image, well under in_token_limit. pad = 2*14 = 28.
  PreprocessConfig cfg;  // in_token_limit=25600
  ResizeTransform xf;
  int gh = 0, gw = 0, L = 0;
  // 100x50 -> ceil(100/28)*28 = 112, ceil(50/28)*28 = 56.
  ASSERT_TRUE(ComputeTargetSize(100, 50, cfg, &xf, &gh, &gw, &L));
  EXPECT_FALSE(xf.shrunk);
  EXPECT_EQ(xf.target_w, 112);
  EXPECT_EQ(xf.target_h, 56);
  EXPECT_EQ(gw, 112 / 14);
  EXPECT_EQ(gh, 56 / 14);
  EXPECT_EQ(L, gw * gh);
  EXPECT_EQ(xf.shrunk_w, 100);
  EXPECT_EQ(xf.shrunk_h, 50);
}

TEST(Geometry, AlreadyMultiple) {
  PreprocessConfig cfg;
  ResizeTransform xf;
  int gh = 0, gw = 0, L = 0;
  // 56x28 are already multiples of 28; target == input.
  ASSERT_TRUE(ComputeTargetSize(56, 28, cfg, &xf, &gh, &gw, &L));
  EXPECT_EQ(xf.target_w, 56);
  EXPECT_EQ(xf.target_h, 28);
  EXPECT_EQ(gw, 4);
  EXPECT_EQ(gh, 2);
  EXPECT_EQ(L, 8);
}

TEST(Geometry, ShrinkGuardFires) {
  // Force the in_token_limit branch with a small limit.
  PreprocessConfig cfg;
  cfg.in_token_limit = 100;  // tokens = (w/14)*(h/14)
  ResizeTransform xf;
  int gh = 0, gw = 0, L = 0;
  // 1000x1000: tokens = (1000/14)*(1000/14) = 71*71 = 5041 > 100.
  // scale = sqrt(100/5041) = 10/71 ~= 0.140845
  // new_w = int(1000*0.140845) = int(140.845) = 140 (both axes)
  ASSERT_TRUE(ComputeTargetSize(1000, 1000, cfg, &xf, &gh, &gw, &L));
  EXPECT_TRUE(xf.shrunk);
  EXPECT_EQ(xf.shrunk_w, 140);
  EXPECT_EQ(xf.shrunk_h, 140);
  // target = ceil(140/28)*28 = 140.
  EXPECT_EQ(xf.target_w, 140);
  EXPECT_EQ(xf.target_h, 140);
  EXPECT_EQ(gw, 10);
  EXPECT_EQ(gh, 10);
  EXPECT_EQ(L, 100);
}

TEST(Geometry, PosEmbGuardRejects) {
  PreprocessConfig cfg;  // limit 25600, so no shrink for moderate sizes
  ResizeTransform xf;
  int gh = 0, gw = 0, L = 0;
  // Need target_w/14 >= 512 -> target_w >= 7168. Input 7200 wide, tiny height.
  // tokens for 7200x28 = (7200/14)*(28/14)=514*2=1028 < 25600 (no shrink).
  // target_w = ceil(7200/28)*28 = 7224; 7224/14 = 516 >= 512 -> reject.
  EXPECT_FALSE(ComputeTargetSize(7200, 28, cfg, &xf, &gh, &gw, &L));
}

TEST(Geometry, InvalidInputs) {
  PreprocessConfig cfg;
  ResizeTransform xf;
  int gh = 0, gw = 0, L = 0;
  EXPECT_FALSE(ComputeTargetSize(0, 50, cfg, &xf, &gh, &gw, &L));
  EXPECT_FALSE(ComputeTargetSize(50, -1, cfg, &xf, &gh, &gw, &L));
}

// scale_x/scale_y invert coordinates from network input space to original.
TEST(Geometry, InverseScales) {
  PreprocessConfig cfg;
  ResizeTransform xf;
  int gh = 0, gw = 0, L = 0;
  ASSERT_TRUE(ComputeTargetSize(100, 50, cfg, &xf, &gh, &gw, &L));
  // A point at the right/bottom edge of the network input maps near orig edge.
  const float x_in = static_cast<float>(xf.target_w);
  const float y_in = static_cast<float>(xf.target_h);
  EXPECT_NEAR(x_in * xf.scale_x, 100.0f, 1e-4);
  EXPECT_NEAR(y_in * xf.scale_y, 50.0f, 1e-4);
}

// ---------------------------------------------------------------------------
// Full pipeline: shapes, normalization range, patch layout.
// ---------------------------------------------------------------------------
TEST(Pipeline, ShapesAndNormalization) {
  const int w = 56, h = 28, c = 3;  // already multiples -> no resize
  std::vector<uint8_t> img(static_cast<size_t>(w) * h * c);
  for (size_t i = 0; i < img.size(); ++i) img[i] = static_cast<uint8_t>(i % 256);

  PreprocessConfig cfg;
  PreprocessResult res;
  ASSERT_TRUE(PreprocessCPU(img.data(), w, h, c, cfg, &res));

  EXPECT_EQ(res.grid_h, 2);
  EXPECT_EQ(res.grid_w, 4);
  EXPECT_EQ(res.num_patches, 8);
  ASSERT_EQ(res.grid_hws.size(), 2u);
  EXPECT_EQ(res.grid_hws[0], 2);  // grid_h
  EXPECT_EQ(res.grid_hws[1], 4);  // grid_w
  EXPECT_EQ(res.pixel_values.size(),
            static_cast<size_t>(8) * 3 * 14 * 14);

  // Normalization: value = u/255*2 - 1, so in [-1, 1].
  for (float v : res.pixel_values) {
    EXPECT_GE(v, -1.0001f);
    EXPECT_LE(v, 1.0001f);
  }
  // u=0 -> -1, u=255 -> 1, u=127.5 ~ 0. Check exact endpoints exist.
  // (channel 0, patch 0, pixel 0) reads src[0]=0 -> -1.
  EXPECT_NEAR(res.pixel_values[0], -1.0f, 1e-5);
}

// Patchify layout: verify a known pixel lands in the right (l,c,y,x) slot.
TEST(Pipeline, PatchifyLayout) {
  const int patch = 14;
  const int w = 28, h = 28, c = 3;  // grid 2x2, no resize
  std::vector<uint8_t> img(static_cast<size_t>(w) * h * c, 0);

  // Mark pixel at (sx=15, sy=3), channel 1 with value 200.
  // That is grid cell gx=1 (15/14), gy=0; within-patch px=1, py=3.
  const int sx = 15, sy = 3, ch = 1;
  img[(static_cast<size_t>(sy) * w + sx) * 3 + ch] = 200;

  PreprocessConfig cfg;
  PreprocessResult res;
  ASSERT_TRUE(PreprocessCPU(img.data(), w, h, c, cfg, &res));
  ASSERT_EQ(res.grid_h, 2);
  ASSERT_EQ(res.grid_w, 2);

  const int gx = 1, gy = 0, px = 1, py = 3;
  const int l = gy * res.grid_w + gx;  // = 1
  const size_t idx =
      ((static_cast<size_t>(l) * 3 + ch) * patch + py) * patch + px;
  const float expected = 200.0f / 255.0f * 2.0f - 1.0f;
  EXPECT_NEAR(res.pixel_values[idx], expected, 1e-5);

  // A different channel at the same spatial location should be -1 (value 0).
  const int other_ch = 0;
  const size_t idx0 =
      ((static_cast<size_t>(l) * 3 + other_ch) * patch + py) * patch + px;
  EXPECT_NEAR(res.pixel_values[idx0], -1.0f, 1e-5);
}

// RGBA -> white composite: alpha=0 yields white (value 1.0 after normalize).
TEST(Pipeline, RgbaWhiteComposite) {
  const int w = 28, h = 28, c = 4;
  std::vector<uint8_t> img(static_cast<size_t>(w) * h * c, 0);
  // Fully transparent everywhere (alpha=0): foreground RGB ignored, -> white.
  // Leave RGB=0, alpha=0.
  PreprocessConfig cfg;
  PreprocessResult res;
  ASSERT_TRUE(PreprocessCPU(img.data(), w, h, c, cfg, &res));
  // White (255) -> normalize -> +1.
  for (float v : res.pixel_values) EXPECT_NEAR(v, 1.0f, 1e-5);

  // Fully opaque red (alpha=255, R=255): R->1, G,B-> -1.
  std::fill(img.begin(), img.end(), 0);
  for (int p = 0; p < w * h; ++p) {
    img[p * 4 + 0] = 255;  // R
    img[p * 4 + 3] = 255;  // A
  }
  ASSERT_TRUE(PreprocessCPU(img.data(), w, h, c, cfg, &res));
  // patch 0, channel R, pixel 0 -> +1 ; channel G -> -1.
  EXPECT_NEAR(res.pixel_values[(0 * 3 + 0) * 14 * 14 + 0], 1.0f, 1e-5);
  EXPECT_NEAR(res.pixel_values[(0 * 3 + 1) * 14 * 14 + 0], -1.0f, 1e-5);
}

// Pipeline rejects oversized inputs (pos emb guard) without crashing.
TEST(Pipeline, RejectsPosEmbExceed) {
  const int w = 7200, h = 28, c = 3;
  std::vector<uint8_t> img(static_cast<size_t>(w) * h * c, 10);
  PreprocessConfig cfg;
  PreprocessResult res;
  EXPECT_FALSE(PreprocessCPU(img.data(), w, h, c, cfg, &res));
}

// Pipeline with a real resize (non-multiple dims) produces consistent shapes.
TEST(Pipeline, ResizePathConsistent) {
  const int w = 100, h = 60, c = 3;
  std::vector<uint8_t> img(static_cast<size_t>(w) * h * c);
  for (size_t i = 0; i < img.size(); ++i)
    img[i] = static_cast<uint8_t>((i * 7) % 256);

  PreprocessConfig cfg;
  PreprocessResult res;
  ASSERT_TRUE(PreprocessCPU(img.data(), w, h, c, cfg, &res));
  // target_w = ceil(100/28)*28 = 112; target_h = ceil(60/28)*28 = 84.
  EXPECT_EQ(res.transform.target_w, 112);
  EXPECT_EQ(res.transform.target_h, 84);
  EXPECT_EQ(res.grid_w, 8);
  EXPECT_EQ(res.grid_h, 6);
  EXPECT_EQ(res.num_patches, 48);
  EXPECT_EQ(res.pixel_values.size(),
            static_cast<size_t>(48) * 3 * 14 * 14);
  for (float v : res.pixel_values) {
    EXPECT_GE(v, -1.0001f);
    EXPECT_LE(v, 1.0001f);
  }
}

}  // namespace
