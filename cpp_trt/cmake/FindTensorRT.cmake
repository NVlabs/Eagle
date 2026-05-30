# ============================================================================
# FindTensorRT.cmake
#
# STATUS: STUB / UNUSED IN PHASE 0.
#
# Phase 0 of the LocateAnything C++ port builds pure-C++ targets only and does
# NOT link TensorRT. This module is provided so that later phases (engine
# build / inference runtime) have a single, documented place to discover an
# installed TensorRT. It is intentionally non-fatal: it never calls
# find_package_handle_standard_args with REQUIRED semantics so that including
# it during Phase 0 cannot break a TRT-less build.
#
# Later-phase usage (Phase 1+):
#   find_package(TensorRT)          # optional discovery
#   if(TensorRT_FOUND)
#     target_link_libraries(<tgt> PRIVATE TensorRT::TensorRT)
#   endif()
#
# Discovery inputs (set any of these to help the search):
#   TENSORRT_ROOT       (CMake var or environment var)
#   CMAKE_PREFIX_PATH
#
# Outputs on success:
#   TensorRT_FOUND
#   TensorRT_INCLUDE_DIRS
#   TensorRT_LIBRARIES
#   TensorRT_VERSION            (parsed from NvInferVersion.h when available)
#   imported target TensorRT::TensorRT
#
# The reference deployment container is nvcr.io/nvidia/tensorrt:25.06-py3,
# where headers live under /usr/include/x86_64-linux-gnu and libraries under
# /usr/lib/x86_64-linux-gnu. TRT-LLM 0.20 @ 7c828d7 is a separate later-phase
# dependency and is NOT discovered here.
# ============================================================================

set(_la_trt_roots
  "${TENSORRT_ROOT}"
  "$ENV{TENSORRT_ROOT}"
  "/usr"
  "/usr/local/tensorrt"
  "/opt/tensorrt")

# --- headers --------------------------------------------------------------
find_path(TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ${_la_trt_roots}
  PATH_SUFFIXES include include/x86_64-linux-gnu
  DOC "TensorRT include directory (contains NvInfer.h)")

# --- core libraries -------------------------------------------------------
find_library(TensorRT_nvinfer_LIBRARY
  NAMES nvinfer
  HINTS ${_la_trt_roots}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
  DOC "TensorRT nvinfer library")

find_library(TensorRT_nvonnxparser_LIBRARY
  NAMES nvonnxparser
  HINTS ${_la_trt_roots}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
  DOC "TensorRT nvonnxparser library")

# --- version parse --------------------------------------------------------
set(TensorRT_VERSION "")
if(TensorRT_INCLUDE_DIR)
  set(_la_trt_ver_hdr "")
  if(EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    set(_la_trt_ver_hdr "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  elseif(EXISTS "${TensorRT_INCLUDE_DIR}/NvInfer.h")
    set(_la_trt_ver_hdr "${TensorRT_INCLUDE_DIR}/NvInfer.h")
  endif()
  if(_la_trt_ver_hdr)
    file(STRINGS "${_la_trt_ver_hdr}" _la_trt_major REGEX "define NV_TENSORRT_MAJOR")
    file(STRINGS "${_la_trt_ver_hdr}" _la_trt_minor REGEX "define NV_TENSORRT_MINOR")
    file(STRINGS "${_la_trt_ver_hdr}" _la_trt_patch REGEX "define NV_TENSORRT_PATCH")
    string(REGEX REPLACE "[^0-9]" "" _la_trt_major "${_la_trt_major}")
    string(REGEX REPLACE "[^0-9]" "" _la_trt_minor "${_la_trt_minor}")
    string(REGEX REPLACE "[^0-9]" "" _la_trt_patch "${_la_trt_patch}")
    if(_la_trt_major)
      set(TensorRT_VERSION "${_la_trt_major}.${_la_trt_minor}.${_la_trt_patch}")
    endif()
  endif()
endif()

# --- aggregate ------------------------------------------------------------
set(TensorRT_FOUND FALSE)
if(TensorRT_INCLUDE_DIR AND TensorRT_nvinfer_LIBRARY)
  set(TensorRT_FOUND TRUE)
  set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}")
  set(TensorRT_LIBRARIES "${TensorRT_nvinfer_LIBRARY}")
  if(TensorRT_nvonnxparser_LIBRARY)
    list(APPEND TensorRT_LIBRARIES "${TensorRT_nvonnxparser_LIBRARY}")
  endif()

  if(NOT TARGET TensorRT::TensorRT)
    add_library(TensorRT::TensorRT INTERFACE IMPORTED)
    set_target_properties(TensorRT::TensorRT PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${TensorRT_LIBRARIES}")
  endif()

  message(STATUS "FindTensorRT: found TensorRT ${TensorRT_VERSION} at ${TensorRT_INCLUDE_DIR}")
else()
  # Non-fatal by design — Phase 0 does not need TensorRT.
  if(TensorRT_FIND_REQUIRED)
    message(FATAL_ERROR "FindTensorRT: TensorRT requested as REQUIRED but not found.")
  else()
    message(STATUS "FindTensorRT: TensorRT not found (expected & harmless in Phase 0).")
  endif()
endif()

mark_as_advanced(
  TensorRT_INCLUDE_DIR
  TensorRT_nvinfer_LIBRARY
  TensorRT_nvonnxparser_LIBRARY)

unset(_la_trt_roots)
unset(_la_trt_ver_hdr)
