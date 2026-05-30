// Unit tests for la_preprocess (geometry, normalize, RGBA composite, patchify,
// and a small BICUBIC resize case).
//
// Expected values are computed INDEPENDENTLY inside each test:
//   - target-size / L formula reimplemented with ceil/floor math,
//   - normalize reimplemented as (u/255 - mean)/std,
//   - RGBA->white composite reimplemented per-pixel,
//   - BICUBIC verified against an independent Pillow-style separable resampler.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "la/preprocess/preprocess.h"

namespace {

using la::preprocess::PreprocessConfig;
using la::preprocess::PreprocessResult;
using la::preprocess::ResizeTransform;
using la::preprocess::ComputeTargetSize;
using la::preprocess::PreprocessCPU;

// Independent target-size reference (matches spec formula).
struct Geo {
  bool ok;
  int target_w, target_h, grid_h, grid_w, num_patches;
  double scale_x, scale_y;
};

Geo RefGeometry(int w, int h, const PreprocessConfig& cfg) {
  Geo g{};
  g.ok = false;
  if (w <= 0 || h <= 0) return g;
  const int patch = cfg.patch_size;          // 14
  const int pad_w = cfg.merge_kernel_w * patch;  // 28
  const int pad_h = cfg.merge_kernel_h * patch;  // 28
  long tokens = static_cast<long>(w / patch) * static_cast<long>(h / patch);
  int new_w = w, new_h = h;
  if (tokens > cfg.in_token_limit) {
    double scale = std::sqrt(static_cast<double>(cfg.in_token_limit) /
                             static_cast<double>(tokens));
    new_w = static_cast<int>(w * scale);  // truncation == Python int()
    new_h = static_cast<int>(h * scale);
  }
  auto ceil_to = [](int v, int m) { return ((v + m - 1) / m) * m; };
  g.target_w = ceil_to(new_w, pad_w);
  g.target_h = ceil_to(new_h, pad_h);
  g.grid_w = g.target_w / patch;
  g.grid_h = g.target_h / patch;
  // Position-embedding guard.
  if (g.grid_w >= 512 || g.grid_h >= 512) return g;
  g.num_patches = g.grid_w * g.grid_h;
  g.scale_x = static_cast<double>(w) / g.target_w;
  g.scale_y = static_cast<double>(h) / g.target_h;
  g.ok = true;
  return g;
}

}  // namespace

TEST(Geometry, CeilToMultipleOf28AndGrid) {
  PreprocessConfig cfg;  // defaults: patch=14, merge 2x2, limit 25600
  struct Case { int w, h; };
  // None exceed the token limit (25600 tokens), so no shrink.
  for (Case c : {Case{14, 14}, Case{29, 15}, Case{100, 50}, Case{28, 28},
                 Case{200, 137}, Case{13, 13}}) {
    ResizeTransform tr{};
    int gh = 0, gw = 0, np = 0;
    bool ok = ComputeTargetSize(c.w, c.h, cfg, &tr, &gh, &gw, &np);
    Geo ref = RefGeometry(c.w, c.h, cfg);
    ASSERT_EQ(ok, ref.ok) << "w=" << c.w << " h=" << c.h;
    if (!ok) continue;
    EXPECT_EQ(tr.target_w, ref.target_w) << "w=" << c.w << " h=" << c.h;
    EXPECT_EQ(tr.target_h, ref.target_h) << "w=" << c.w << " h=" << c.h;
    EXPECT_EQ(gw, ref.grid_w);
    EXPECT_EQ(gh, ref.grid_h);
    EXPECT_EQ(np, ref.num_patches);
    EXPECT_EQ(np, gw * gh);
    // target dims are multiples of 28.
    EXPECT_EQ(tr.target_w % 28, 0);
    EXPECT_EQ(tr.target_h % 28, 0);
    // inverse scale map. ResizeTransform stores scale_{x,y} as float, so the
    // tolerance must reflect float32 precision (~1e-7 relative), not double.
    EXPECT_NEAR(tr.scale_x, ref.scale_x, 1e-6);
    EXPECT_NEAR(tr.scale_y, ref.scale_y, 1e-6);
  }
}

// Token count L == grid_h*grid_w, and (for an image already a multiple of the
// patch grid) equals (target_w/patch)*(target_h/patch). Also verifies the
// (h*w)//4 merged-token relationship downstream uses.
TEST(Geometry, TokenCountFormula) {
  PreprocessConfig cfg;
  ResizeTransform tr{};
  int gh = 0, gw = 0, np = 0;
  ASSERT_TRUE(ComputeTargetSize(56, 84, cfg, &tr, &gh, &gw, &np));
  // 56/14=4 wide, 84/14=6 tall, both already multiples of 28.
  EXPECT_EQ(gw, 4);
  EXPECT_EQ(gh, 6);
  EXPECT_EQ(np, 24);
  // merged context-token count = (gh*gw)/4
  EXPECT_EQ((gh * gw) / 4, 6);
}

// Normalize: with mean=std=0.5 the fused transform must be value = u/255*2 - 1.
TEST(Normalize, MapsToMinusOneToOne) {
  PreprocessConfig cfg;  // mean=std=0.5
  // A 28x28 RGB image (one full pad tile) with a known ramp; check a few pixels
  // after the full pipeline. We pick a flat image so resize is identity and the
  // only transform on interior pixels is normalize.
  const int W = 28, H = 28, C = 3;
  std::vector<uint8_t> img(W * H * C);
  // Constant value 0, 128, 255 in three vertical bands? Use a single constant so
  // bicubic on a flat image is exactly the constant (independent fact).
  for (uint8_t val : {uint8_t{0}, uint8_t{128}, uint8_t{255}}) {
    std::fill(img.begin(), img.end(), val);
    PreprocessResult out;
    ASSERT_TRUE(PreprocessCPU(img.data(), W, H, C, cfg, &out));
    float expected = static_cast<float>(val) / 255.0f * 2.0f - 1.0f;
    // All pixel_values should equal expected for a flat image.
    for (size_t i = 0; i < out.pixel_values.size(); ++i) {
      ASSERT_NEAR(out.pixel_values[i], expected, 1e-4)
          << "val=" << static_cast<int>(val) << " i=" << i;
    }
  }
}

// Endpoint check independent of pipeline: 0 -> -1, 255 -> +1, 128 -> ~0.0039.
TEST(Normalize, EndpointValues) {
  auto norm = [](uint8_t u) { return static_cast<float>(u) / 255.0f * 2.0f - 1.0f; };
  EXPECT_NEAR(norm(0), -1.0f, 1e-6);
  EXPECT_NEAR(norm(255), 1.0f, 1e-6);
  EXPECT_NEAR(norm(128), 128.0f / 255.0f * 2.0f - 1.0f, 1e-6);
}

// RGBA -> white composite: out = a*fg + (1-a)*255, then normalized.
TEST(RgbaComposite, WhiteBackground) {
  PreprocessConfig cfg;  // rgba_bg default white (255)
  const int W = 28, H = 28;
  // Flat RGBA: fg color (10,20,30), alpha 100/255.
  const uint8_t fr = 10, fg = 20, fb = 30, fa = 100;
  std::vector<uint8_t> img(W * H * 4);
  for (int i = 0; i < W * H; ++i) {
    img[i * 4 + 0] = fr;
    img[i * 4 + 1] = fg;
    img[i * 4 + 2] = fb;
    img[i * 4 + 3] = fa;
  }
  PreprocessResult out;
  ASSERT_TRUE(PreprocessCPU(img.data(), W, H, 4, cfg, &out));

  // Independent composite-then-normalize reference.
  double a = fa / 255.0;
  auto composite = [&](uint8_t fgv) {
    double c = a * fgv + (1.0 - a) * 255.0;  // white bg
    // round-half-away-from-zero to uint8 (composite happens in uint8 domain)
    int iv = static_cast<int>(std::floor(c + 0.5));
    iv = std::clamp(iv, 0, 255);
    return iv;
  };
  int cr = composite(fr), cg = composite(fg), cb = composite(fb);
  auto norm = [](int u) { return static_cast<float>(u) / 255.0f * 2.0f - 1.0f; };
  float er = norm(cr), eg = norm(cg), eb = norm(cb);

  // pixel_values layout (L,3,patch,patch): idx(l,c,y,x)=((l*3+c)*patch+y)*patch+x
  const int patch = out.patch_size;
  const int L = out.num_patches;
  // Allow a small tolerance because the impl may composite in float before
  // bicubic; flat image keeps values constant so check channel means.
  for (int l = 0; l < L; ++l) {
    for (int y = 0; y < patch; ++y) {
      for (int x = 0; x < patch; ++x) {
        float r = out.pixel_values[((l * 3 + 0) * patch + y) * patch + x];
        float g = out.pixel_values[((l * 3 + 1) * patch + y) * patch + x];
        float b = out.pixel_values[((l * 3 + 2) * patch + y) * patch + x];
        EXPECT_NEAR(r, er, 2e-2) << "l=" << l << " channel R";
        EXPECT_NEAR(g, eg, 2e-2) << "l=" << l << " channel G";
        EXPECT_NEAR(b, eb, 2e-2) << "l=" << l << " channel B";
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Independent Pillow-style BICUBIC resampler for a small known case.
// ---------------------------------------------------------------------------
namespace {

double CubicKernel(double x) {
  const double a = -0.5;
  x = std::fabs(x);
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  return 0.0;
}

// 1-D Pillow resample along an axis of length in_size -> out_size.
// Produces, for each output index, a list of (src_index, weight).
struct Tap { int idx; double w; };
std::vector<std::vector<Tap>> BuildTaps(int in_size, int out_size) {
  double scale = static_cast<double>(in_size) / out_size;
  double filterscale = std::max(scale, 1.0);
  double support = 2.0 * filterscale;  // bicubic support radius 2
  std::vector<std::vector<Tap>> taps(out_size);
  for (int xx = 0; xx < out_size; ++xx) {
    double center = (xx + 0.5) * scale;
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    std::vector<Tap> row;
    double total = 0.0;
    for (int x = xmin; x < xmax; ++x) {
      double w = CubicKernel((x - center + 0.5) / filterscale);
      row.push_back({x, w});
      total += w;
    }
    if (total != 0.0) {
      for (auto& t : row) t.w /= total;
    }
    taps[xx] = std::move(row);
  }
  return taps;
}

// Resample a single-channel double image (row-major HxW) to out_h x out_w,
// Pillow's separable H-then-V double-precision path.
std::vector<double> RefBicubic(const std::vector<double>& in, int in_w, int in_h,
                               int out_w, int out_h) {
  auto htaps = BuildTaps(in_w, out_w);
  std::vector<double> tmp(static_cast<size_t>(in_h) * out_w);
  for (int y = 0; y < in_h; ++y) {
    for (int ox = 0; ox < out_w; ++ox) {
      double acc = 0.0;
      for (const auto& t : htaps[ox]) acc += in[y * in_w + t.idx] * t.w;
      tmp[y * out_w + ox] = acc;
    }
  }
  auto vtaps = BuildTaps(in_h, out_h);
  std::vector<double> out(static_cast<size_t>(out_h) * out_w);
  for (int oy = 0; oy < out_h; ++oy) {
    for (int ox = 0; ox < out_w; ++ox) {
      double acc = 0.0;
      for (const auto& t : vtaps[oy]) acc += tmp[t.idx * out_w + ox] * t.w;
      out[oy * out_w + ox] = acc;
    }
  }
  return out;
}

}  // namespace

// Small known BICUBIC case: upscale a flat-then-ramp image and confirm that the
// preprocess pipeline's resampled values match an independent Pillow-style
// resampler (within a few LSB, accounting for the uint8 round + normalize).
//
// We craft an image whose native size is NOT a multiple of 28 so the pipeline
// must bicubic-resize up to the next multiple of 28, exercising the resampler.
TEST(Bicubic, MatchesIndependentResamplerSmallCase) {
  PreprocessConfig cfg;
  const int W = 20, H = 20, C = 3;  // -> target 28x28 (ceil to 28)
  // Horizontal ramp 0..255 across width; same on all rows and channels.
  std::vector<uint8_t> img(W * H * C);
  std::vector<double> chan(W * H);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      uint8_t v = static_cast<uint8_t>(std::min(255, x * 13));
      chan[y * W + x] = v;
      for (int c = 0; c < C; ++c) img[(y * W + x) * C + c] = v;
    }
  }

  PreprocessResult out;
  ASSERT_TRUE(PreprocessCPU(img.data(), W, H, C, cfg, &out));
  ASSERT_EQ(out.transform.target_w, 28);
  ASSERT_EQ(out.transform.target_h, 28);

  // Independent reference: resize 20x20 -> 28x28, round to uint8, normalize.
  std::vector<double> ref = RefBicubic(chan, W, H, 28, 28);
  auto norm = [](double u8) {
    return static_cast<float>(u8 / 255.0 * 2.0 - 1.0);
  };

  // Reconstruct a full 28x28 normalized image from the patchified output to
  // compare against the reference grid. patch=14, grid 2x2.
  const int patch = out.patch_size;       // 14
  const int grid_w = out.grid_w;          // 2
  // de-patchify channel 0 (all channels equal here).
  std::vector<float> full(28 * 28);
  for (int l = 0; l < out.num_patches; ++l) {
    int gy = l / grid_w;
    int gx = l % grid_w;
    for (int py = 0; py < patch; ++py) {
      for (int px = 0; px < patch; ++px) {
        float val = out.pixel_values[((l * 3 + 0) * patch + py) * patch + px];
        int Y = gy * patch + py;
        int X = gx * patch + px;
        full[Y * 28 + X] = val;
      }
    }
  }

  // Compare. Pillow's internal fixed-point uint8 weights vs our double path can
  // differ by ~1 LSB (post-round), which after normalize is ~2/255 ~= 0.0078;
  // allow a small tolerance.
  for (int y = 0; y < 28; ++y) {
    for (int x = 0; x < 28; ++x) {
      double rv = std::clamp(ref[y * 28 + x], 0.0, 255.0);
      // round half away from zero, matching the impl's documented behavior.
      int ru = static_cast<int>(std::floor(rv + 0.5));
      ru = std::clamp(ru, 0, 255);
      float expected = norm(ru);
      EXPECT_NEAR(full[y * 28 + x], expected, 3.0f / 255.0f)
          << "y=" << y << " x=" << x;
    }
  }
}
