// SPDX-License-Identifier: Apache-2.0
//
// OPTIONAL CUDA kernel for MoonViT real-arithmetic 2D-RoPE.
//
// This file is ONLY compiled when the CMake option LA_VISION_ROPE_WITH_CUDA is
// ON (default OFF). It is intentionally kept out of the always-buildable CPU
// path because the local nvcc/gcc combination may be mismatched. The kernel is
// a direct, element-for-element transliteration of the CPU reference in
// vision_rope.cpp so that results are bit-comparable up to FP ordering.
//
// It is NOT validated in this environment (no working CUDA toolchain confirmed
// for this build). The CPU reference in vision_rope.cpp is the source of truth.

#include "la/vision_rope/vision_rope.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace la {
namespace vision_rope {
namespace {

// One thread per (token, head, complex-slot). Applies the interleaved complex
// multiply:  out[2s]   = a*c - b*d ;  out[2s+1] = a*d + b*c
// where (a,b) = (vec[2s], vec[2s+1]), (c,d) = (cos[t,s], sin[t,s]).
__global__ void apply_rope_real_kernel(float* __restrict__ q,
                                       const float* __restrict__ cos,
                                       const float* __restrict__ sin,
                                       std::size_t seq_len, std::size_t n_heads,
                                       std::size_t head_dim,
                                       std::size_t half_dim) {
  const std::size_t total =
      seq_len * n_heads * half_dim;  // number of complex slots to rotate
  for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const std::size_t s = idx % half_dim;
    const std::size_t hth = idx / half_dim;          // token*heads + head
    const std::size_t h = hth % n_heads;
    const std::size_t t = hth / n_heads;
    (void)h;  // head index does not affect the broadcast table lookup

    float* vec = q + hth * head_dim;
    const float a = vec[2 * s];
    const float b = vec[2 * s + 1];
    const float c = cos[t * half_dim + s];
    const float d = sin[t * half_dim + s];
    vec[2 * s] = a * c - b * d;
    vec[2 * s + 1] = a * d + b * c;
  }
}

}  // namespace

// Host launcher operating on DEVICE pointers. Provided so a TRT plugin can call
// it; declared here (not in the public header) to avoid leaking CUDA types into
// the always-built CPU API.
void apply_rope_real_cuda(float* d_q, std::size_t seq_len, std::size_t n_heads,
                          std::size_t head_dim, const float* d_cos,
                          const float* d_sin, void* stream_v) {
  if (head_dim == 0 || (head_dim % 2) != 0) {
    throw std::invalid_argument(
        "apply_rope_real_cuda: head_dim must be a positive even number");
  }
  const std::size_t half_dim = head_dim / 2;
  const std::size_t total = seq_len * n_heads * half_dim;
  if (total == 0) return;

  cudaStream_t stream = static_cast<cudaStream_t>(stream_v);
  const int block = 256;
  const int grid = static_cast<int>((total + block - 1) / block);
  apply_rope_real_kernel<<<grid, block, 0, stream>>>(
      d_q, d_cos, d_sin, seq_len, n_heads, head_dim, half_dim);
}

}  // namespace vision_rope
}  // namespace la
