# FindTensorRT.cmake
# -----------------------------------------------------------------------------
# Locate an NVIDIA TensorRT installation (TRT 10.x, e.g. 10.11 in the project
# Docker image).
#
# This module is ONLY invoked when the top-level option LA_BUILD_TRT is ON.
# The default Phase 0 build is TRT-free and never includes this file, so a host
# with no TensorRT installed still configures and builds the unit tests.
#
# Search order for each component:
#   1. The TensorRT_ROOT CMake/cache variable or environment variable.
#   2. The CUDA toolkit root (CUDAToolkit_ROOT / CUDA_PATH) as a hint.
#   3. Standard system locations (/usr, /usr/local, /usr/local/tensorrt).
#
# Result variables:
#   TensorRT_FOUND          - TRUE if all required pieces were located.
#   TensorRT_INCLUDE_DIRS   - Include directory containing NvInfer.h.
#   TensorRT_LIBRARIES      - All located TensorRT import libraries.
#   TensorRT_VERSION        - Version string parsed from NvInferVersion.h
#                             (e.g. "10.11.0"), when available.
#
# Imported target:
#   TensorRT::TensorRT      - INTERFACE target carrying the include dirs and
#                             all located libraries. Link against this.
#
# Components:
#   nvinfer         (always required)
#   nvonnxparser    (required by default; needed to parse exported ONNX)
#   nvinfer_plugin  (optional; many builds use it, but it is not strictly
#                    required to instantiate a runtime from a serialized engine)
#
# Usage:
#   find_package(TensorRT REQUIRED)
#   target_link_libraries(my_target PRIVATE TensorRT::TensorRT)
# -----------------------------------------------------------------------------

# Honor an explicit root from cache or environment.
if(NOT TensorRT_ROOT AND DEFINED ENV{TensorRT_ROOT})
  set(TensorRT_ROOT "$ENV{TensorRT_ROOT}")
endif()
if(NOT TensorRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
  set(TensorRT_ROOT "$ENV{TENSORRT_ROOT}")
endif()

# CUDA toolkit can act as a hint (TRT is frequently co-located with CUDA, and
# pip-installed TRT wheels live under the Python site-packages tensorrt_libs).
set(_TRT_HINTS)
if(TensorRT_ROOT)
  list(APPEND _TRT_HINTS "${TensorRT_ROOT}")
endif()
if(DEFINED CUDAToolkit_ROOT)
  list(APPEND _TRT_HINTS "${CUDAToolkit_ROOT}")
endif()
if(DEFINED ENV{CUDA_PATH})
  list(APPEND _TRT_HINTS "$ENV{CUDA_PATH}")
endif()

set(_TRT_SYSTEM_PATHS
  /usr
  /usr/local
  /usr/local/tensorrt
  /opt/tensorrt
  /opt/nvidia/tensorrt
)

# ---- headers ----------------------------------------------------------------
find_path(TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ${_TRT_HINTS}
  PATHS ${_TRT_SYSTEM_PATHS}
  PATH_SUFFIXES include include/x86_64-linux-gnu
  DOC "TensorRT include directory containing NvInfer.h"
)

# ---- libraries --------------------------------------------------------------
# Common per-arch lib suffixes; covers both tarball and apt layouts.
set(_TRT_LIB_SUFFIXES
  lib
  lib64
  lib/x86_64-linux-gnu
  targets/x86_64-linux-gnu/lib
)

find_library(TensorRT_nvinfer_LIBRARY
  NAMES nvinfer
  HINTS ${_TRT_HINTS}
  PATHS ${_TRT_SYSTEM_PATHS}
  PATH_SUFFIXES ${_TRT_LIB_SUFFIXES}
  DOC "TensorRT nvinfer library"
)

find_library(TensorRT_nvonnxparser_LIBRARY
  NAMES nvonnxparser
  HINTS ${_TRT_HINTS}
  PATHS ${_TRT_SYSTEM_PATHS}
  PATH_SUFFIXES ${_TRT_LIB_SUFFIXES}
  DOC "TensorRT nvonnxparser library"
)

# nvinfer_plugin is optional.
find_library(TensorRT_nvinfer_plugin_LIBRARY
  NAMES nvinfer_plugin
  HINTS ${_TRT_HINTS}
  PATHS ${_TRT_SYSTEM_PATHS}
  PATH_SUFFIXES ${_TRT_LIB_SUFFIXES}
  DOC "TensorRT nvinfer_plugin library (optional)"
)

# ---- version ----------------------------------------------------------------
# TRT 8.x kept the version macros in NvInferVersion.h; TRT 10.x still ships it.
set(TensorRT_VERSION "")
foreach(_trt_ver_hdr NvInferVersion.h NvInfer.h)
  if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/${_trt_ver_hdr}")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/${_trt_ver_hdr}" _trt_ver_lines
      REGEX "^#define (NV_TENSORRT_MAJOR|NV_TENSORRT_MINOR|NV_TENSORRT_PATCH) ")
    if(_trt_ver_lines)
      string(REGEX REPLACE ".*NV_TENSORRT_MAJOR ([0-9]+).*" "\\1"
        _trt_major "${_trt_ver_lines}")
      string(REGEX REPLACE ".*NV_TENSORRT_MINOR ([0-9]+).*" "\\1"
        _trt_minor "${_trt_ver_lines}")
      string(REGEX REPLACE ".*NV_TENSORRT_PATCH ([0-9]+).*" "\\1"
        _trt_patch "${_trt_ver_lines}")
      if(_trt_major MATCHES "^[0-9]+$")
        set(TensorRT_VERSION "${_trt_major}.${_trt_minor}.${_trt_patch}")
        break()
      endif()
    endif()
  endif()
endforeach()

# ---- standard handling ------------------------------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
  REQUIRED_VARS
    TensorRT_INCLUDE_DIR
    TensorRT_nvinfer_LIBRARY
    TensorRT_nvonnxparser_LIBRARY
  VERSION_VAR TensorRT_VERSION
)

if(TensorRT_FOUND)
  set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}")
  set(TensorRT_LIBRARIES
    "${TensorRT_nvinfer_LIBRARY}"
    "${TensorRT_nvonnxparser_LIBRARY}"
  )
  if(TensorRT_nvinfer_plugin_LIBRARY)
    list(APPEND TensorRT_LIBRARIES "${TensorRT_nvinfer_plugin_LIBRARY}")
  endif()

  if(NOT TARGET TensorRT::TensorRT)
    add_library(TensorRT::TensorRT INTERFACE IMPORTED)
    set_target_properties(TensorRT::TensorRT PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${TensorRT_LIBRARIES}"
    )
  endif()
endif()

mark_as_advanced(
  TensorRT_INCLUDE_DIR
  TensorRT_nvinfer_LIBRARY
  TensorRT_nvonnxparser_LIBRARY
  TensorRT_nvinfer_plugin_LIBRARY
)
