# SPDX-License-Identifier: Apache-2.0
#
# la_vision_rope : pure-C++ real-arithmetic 2D-RoPE reference for MoonViT.
#
# This target builds with the C++ toolchain ALONE. It has NO dependency on
# Torch, TensorRT, TRT-LLM, or CUDA. An optional CUDA kernel build is gated
# behind LA_VISION_ROPE_WITH_CUDA (default OFF) so that a possibly-mismatched
# nvcc/gcc pair can never break the always-buildable CPU reference.

cmake_minimum_required(VERSION 3.20)

# Allow building this directory standalone (e.g. for the test agent) as well as
# via add_subdirectory() from a parent project.
if (NOT DEFINED PROJECT_NAME)
  project(la_vision_rope LANGUAGES CXX)
endif()

option(LA_VISION_ROPE_WITH_CUDA
       "Build the optional CUDA kernel for la_vision_rope (default OFF)" OFF)

# ---------------------------------------------------------------------------
# Always-built CPU static library.
# ---------------------------------------------------------------------------
add_library(la_vision_rope STATIC
  src/vision_rope.cpp
)

target_include_directories(la_vision_rope
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(la_vision_rope PUBLIC cxx_std_17)

set_target_properties(la_vision_rope PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

if (MSVC)
  target_compile_options(la_vision_rope PRIVATE /W4)
else()
  target_compile_options(la_vision_rope PRIVATE -Wall -Wextra)
endif()

# Provide a namespaced alias for clean consumption via target_link_libraries.
add_library(la::vision_rope ALIAS la_vision_rope)

# ---------------------------------------------------------------------------
# Optional CUDA kernel (OFF by default; never required for Phase 0 targets).
# ---------------------------------------------------------------------------
if (LA_VISION_ROPE_WITH_CUDA)
  enable_language(CUDA)
  target_sources(la_vision_rope PRIVATE src/vision_rope_cuda.cu)
  target_compile_definitions(la_vision_rope PUBLIC LA_VISION_ROPE_HAS_CUDA=1)
  set_target_properties(la_vision_rope PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )
  message(STATUS "la_vision_rope: optional CUDA kernel ENABLED")
else()
  message(STATUS "la_vision_rope: CUDA kernel disabled (CPU reference only)")
endif()
