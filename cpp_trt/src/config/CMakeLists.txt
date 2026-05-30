# SPDX-License-Identifier: Apache-2.0
#
# la_config: token-id configuration module for the LocateAnything C++ port.
#
# Builds a STATIC library `la_config` from pure C++17 with NO external
# dependencies (no TensorRT, no TRT-LLM, no PyTorch, no JSON library). It is
# safe to build with the C++ toolchain alone.
#
# CUDA: this module contains no device code. A LA_CONFIG_ENABLE_CUDA option is
# provided for consistency with the wider Phase 0 convention (CUDA must be
# OPTIONAL and default OFF so nvcc/gcc mismatches never break the C++ build).
# It currently gates nothing here; the pure-C++ path always builds.

cmake_minimum_required(VERSION 3.20)

# Allow this directory to be configured standalone (for unit testing in
# isolation) or as add_subdirectory() from the parent cpp_trt build.
if(NOT DEFINED PROJECT_NAME)
  project(la_config LANGUAGES CXX)
endif()

option(LA_CONFIG_ENABLE_CUDA
       "Build optional CUDA paths for la_config (none today; reserved)" OFF)

add_library(la_config STATIC
  src/token_ids.cpp
)

target_include_directories(la_config
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(la_config PUBLIC cxx_std_17)

set_target_properties(la_config PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
  target_compile_options(la_config PRIVATE /W4)
else()
  target_compile_options(la_config PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Provide an alias so consumers can link via a namespaced target.
add_library(la::config ALIAS la_config)

# ---------------------------------------------------------------------------
# Optional unit tests (GoogleTest). Enabled when LA_CONFIG_BUILD_TESTS is ON
# (default ON only when configured standalone). The test agent owns the actual
# test sources; this just wires the target if a tests/ dir exists.
# ---------------------------------------------------------------------------
if(CMAKE_PROJECT_NAME STREQUAL "la_config")
  set(_la_config_default_tests ON)
else()
  set(_la_config_default_tests OFF)
endif()
option(LA_CONFIG_BUILD_TESTS "Build la_config unit tests"
       ${_la_config_default_tests})

if(LA_CONFIG_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  enable_testing()
  add_subdirectory(tests)
endif()
