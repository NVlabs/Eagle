# ============================================================================
# locateanything_trt — root build file
#
# C++/TensorRT port of NVIDIA LocateAnything.
#
# Phase 0 (this scaffolding) builds ONLY pure-C++ targets. There is no
# dependency on TensorRT, TRT-LLM, PyTorch, CUDA, or any model checkpoint at
# build time. CUDA kernels are gated behind LA_BUILD_CUDA (default OFF) and
# every CUDA path has a pure-C++ CPU reference that always builds, because the
# local toolchain (nvcc 13.1 + gcc 15) is not guaranteed to interoperate.
# ============================================================================
cmake_minimum_required(VERSION 3.20)

project(locateanything_trt
  VERSION 0.0.1
  DESCRIPTION "C++/TensorRT port of NVIDIA LocateAnything"
  LANGUAGES CXX)

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
option(LA_BUILD_CUDA "build CUDA kernels (Phase 1+; requires nvcc)" OFF)
option(LA_BUILD_TRT "build TensorRT-dependent targets" OFF)
option(BUILD_TESTING "build unit tests" ON)

# ----------------------------------------------------------------------------
# Global C++ settings
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

# Local cmake modules (FindTensorRT.cmake lives here).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ----------------------------------------------------------------------------
# Optional TensorRT discovery
#   TensorRT is OFF by default. The default (Phase 0) build is fully TRT-free:
#   FindTensorRT.cmake is ONLY invoked when LA_BUILD_TRT=ON, so a host with no
#   TensorRT installed still configures and builds the 76 unit tests. The
#   TRT-dependent targets live under src/vision/ and are guarded by this option
#   there as well.
# ----------------------------------------------------------------------------
if(LA_BUILD_TRT)
  find_package(TensorRT REQUIRED)
  message(STATUS "LA_BUILD_TRT=ON: TensorRT ${TensorRT_VERSION} found; "
                 "building TRT-dependent targets.")
else()
  message(STATUS "LA_BUILD_TRT=OFF: TRT-dependent targets are excluded.")
endif()

# ----------------------------------------------------------------------------
# Optional CUDA language enablement
#   Only enabled when LA_BUILD_CUDA=ON. Modules still provide a CPU reference
#   that compiles regardless of this flag.
# ----------------------------------------------------------------------------
if(LA_BUILD_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  message(STATUS "LA_BUILD_CUDA=ON: CUDA kernels will be built.")
else()
  message(STATUS "LA_BUILD_CUDA=OFF: building pure-C++ CPU references only.")
endif()

# ----------------------------------------------------------------------------
# Testing
# ----------------------------------------------------------------------------
include(CTest)
enable_testing()

# ----------------------------------------------------------------------------
# Subdirectories
# ----------------------------------------------------------------------------
add_subdirectory(src)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
