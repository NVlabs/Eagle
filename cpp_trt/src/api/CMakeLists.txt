# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# cpp_trt/src/api/CMakeLists.txt
#
# Builds the static library `la_api` (alias `la::api`): the public
# LocateAnythingWorker-style API plus the verbatim task prompt builders.
#
# Design constraints (Phase 2 environment):
#   * `la_api` ALWAYS builds with the C++ toolchain alone -- the prompt builders
#     and the result parsers (parse_boxes / parse_points) are pure C++ with NO
#     TensorRT, TRT-LLM, PyTorch, or CUDA dependency, so they compile and are
#     unit-tested in the default build.
#   * The engine-touching end-to-end predict() body is compiled only when
#     LA_BUILD_TRT is defined (CMake option below, default OFF). In the default
#     build predict() is a throwing stub, keeping the library linkable and the
#     API surface stable.
#
# Self-contained: configurable standalone or add_subdirectory()'d from a parent.

cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED PROJECT_NAME)
  project(la_api LANGUAGES CXX)
endif()

# Global option (default OFF) gating all TensorRT / TRT-LLM engine code. Declared
# here so the module is usable standalone; a parent project may also declare it.
option(LA_BUILD_TRT "Build TensorRT / TRT-LLM engine-touching code paths" OFF)
option(LA_API_BUILD_TESTS "Build la_api gtest unit tests" ON)

# --------------------------------------------------------------------------
# Core static library (pure C++ -- always builds)
# --------------------------------------------------------------------------
add_library(la_api STATIC
  src/prompts.cpp
  src/worker.cpp
)
add_library(la::api ALIAS la_api)

target_include_directories(la_api PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(la_api PUBLIC cxx_std_20)

set_target_properties(la_api PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
  target_compile_options(la_api PRIVATE /W4)
else()
  target_compile_options(la_api PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --------------------------------------------------------------------------
# Engine-touching code (default OFF). Only when LA_BUILD_TRT=ON do we compile
# the predict() pipeline body and link the engine-bearing sibling modules. The
# pure-C++ prompt builders + parsers are always present, so enabling this is
# purely additive and never required for the default Phase 2 build to be green.
# --------------------------------------------------------------------------
if(LA_BUILD_TRT)
  message(STATUS "la_api: LA_BUILD_TRT=ON -- compiling engine pipeline")
  target_compile_definitions(la_api PUBLIC LA_BUILD_TRT=1)

  # Link the engine-bearing sibling modules only in the TRT build. These targets
  # are expected to exist when configured as part of the full cpp_trt tree.
  foreach(_dep la_tokenizer la_config la_preprocess la_vision la_decode la_generate)
    if(TARGET ${_dep})
      target_link_libraries(la_api PUBLIC ${_dep})
    else()
      message(WARNING "la_api: LA_BUILD_TRT=ON but target '${_dep}' not found; "
                      "the predict() pipeline will not link until it is added.")
    endif()
  endforeach()
else()
  message(STATUS "la_api: LA_BUILD_TRT=OFF -- prompt builders + parsers only "
                 "(predict() is a throwing stub)")
endif()

# --------------------------------------------------------------------------
# Tests (pure C++; always runnable in the default build)
# --------------------------------------------------------------------------
if(LA_API_BUILD_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND OR TARGET GTest::gtest_main)
    enable_testing()
    add_executable(la_api_tests
      tests/test_prompts.cpp
      tests/test_worker.cpp
    )
    target_link_libraries(la_api_tests PRIVATE la_api GTest::gtest_main)
    set_target_properties(la_api_tests PROPERTIES
      CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    include(GoogleTest)
    gtest_discover_tests(la_api_tests)
  else()
    message(STATUS "la_api: GTest not found; skipping tests "
                   "(set LA_API_BUILD_TESTS=OFF to silence).")
  endif()
endif()
