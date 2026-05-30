# cpp_trt/src/decode/CMakeLists.txt
#
# Builds the static library `la_decode` containing the pure-C++ decode-mask and
# position-id builders for Parallel Box Decoding (Phase 0).
#
# Design constraints (Phase 0 environment):
#   * MUST compile with the C++ toolchain alone -- no TensorRT, no TRT-LLM,
#     no PyTorch, no CUDA required.
#   * Optional CUDA kernels are gated behind LA_DECODE_WITH_CUDA (default OFF)
#     because nvcc 13.1 + gcc 15 may mismatch. The pure-C++ reference always
#     builds and is always the source of truth.
#
# This file is self-contained: it can be configured directly as the top of a
# build tree, or add_subdirectory()'d from a parent project.

cmake_minimum_required(VERSION 3.20)

# Only declare project() when configured standalone (so add_subdirectory works).
if(NOT DEFINED PROJECT_NAME)
  project(la_decode LANGUAGES CXX)
endif()

option(LA_DECODE_WITH_CUDA "Build optional CUDA kernels for la_decode" OFF)
option(LA_DECODE_BUILD_TESTS "Build la_decode gtest unit tests" ON)

# --------------------------------------------------------------------------
# Core static library (pure C++ -- always builds)
# --------------------------------------------------------------------------
add_library(la_decode STATIC
  src/mask.cpp
  src/position_ids.cpp
)

target_include_directories(la_decode PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(la_decode PUBLIC cxx_std_17)

set_target_properties(la_decode PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
  target_compile_options(la_decode PRIVATE /W4)
else()
  target_compile_options(la_decode PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --------------------------------------------------------------------------
# Optional CUDA kernels (default OFF). Pure-C++ reference is always present, so
# enabling this is purely additive and never required for Phase 0 targets.
# --------------------------------------------------------------------------
if(LA_DECODE_WITH_CUDA)
  enable_language(CUDA)
  message(STATUS "la_decode: CUDA kernels ENABLED (LA_DECODE_WITH_CUDA=ON)")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/mask_cuda.cu")
    target_sources(la_decode PRIVATE src/mask_cuda.cu)
    target_compile_definitions(la_decode PUBLIC LA_DECODE_HAVE_CUDA=1)
    set_target_properties(la_decode PROPERTIES CUDA_STANDARD 17)
  else()
    message(WARNING "la_decode: LA_DECODE_WITH_CUDA=ON but no src/mask_cuda.cu present; "
                    "building CPU reference only.")
  endif()
else()
  message(STATUS "la_decode: CUDA kernels DISABLED (pure-C++ reference only)")
endif()

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
if(LA_DECODE_BUILD_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND OR TARGET GTest::gtest_main)
    enable_testing()
    add_executable(la_decode_tests
      tests/test_mask.cpp
      tests/test_position_ids.cpp
    )
    target_link_libraries(la_decode_tests PRIVATE la_decode GTest::gtest_main)
    set_target_properties(la_decode_tests PROPERTIES
      CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    include(GoogleTest)
    gtest_discover_tests(la_decode_tests)
  else()
    message(STATUS "la_decode: GTest not found; skipping tests "
                   "(set LA_DECODE_BUILD_TESTS=OFF to silence).")
  endif()
endif()
