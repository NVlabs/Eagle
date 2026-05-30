# SPDX-License-Identifier: Apache-2.0
#
# src/llm/CMakeLists.txt  ->  static lib la_llm  (alias la::llm)
#
# The Qwen2.5 LLM engine abstraction for the LocateAnything C++/TRT port.
#
# Build modes (Risk R5 / engine-count decision, spec Section 3):
#
#   LA_BUILD_TRT = OFF (default)  -- TRT-free build (CI / no-TensorRT machines).
#       Compiles ONLY the throwing factory stub + the public header. No
#       TensorRT, no TRT-LLM, no CUDA runtime dependency. The pure-C++ decode /
#       sampling / detokenize / parse code (in la_decode etc.) links against
#       this so the symbol make_llm_engine() exists; calling it throws
#       LlmEngineError.
#
#   LA_BUILD_TRT = ON            -- compiles both backend implementations
#       (TrtLlm + StandaloneTrt) and links TensorRT (and TRT-LLM if found). All
#       TensorRT / TRT-LLM includes live inside #ifdef LA_BUILD_TRT in those
#       .cpp files.
#
# LA_BUILD_TRT is declared as a project-wide option here only if the parent
# project has not already declared it, so this module is safe to add standalone.

if(NOT DEFINED LA_BUILD_TRT)
  option(LA_BUILD_TRT
         "Build TensorRT / TRT-LLM backends (requires TensorRT install)" OFF)
endif()

if(LA_BUILD_TRT)
  # ---- Full build: both real backends + factory dispatch. -------------------
  add_library(la_llm STATIC
    src/llm_engine_factory.cpp
    src/llm_engine_trtllm.cpp
    src/llm_engine_standalone_trt.cpp
  )
  target_compile_definitions(la_llm PUBLIC LA_BUILD_TRT)

  # TensorRT (required for both backends).
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} /usr/include/x86_64-linux-gnu /usr/local/tensorrt/include)
  find_library(TENSORRT_NVINFER nvinfer
    HINTS ${TENSORRT_ROOT} /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib)
  if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER)
    message(FATAL_ERROR
      "LA_BUILD_TRT=ON but TensorRT not found. Set TENSORRT_ROOT to your "
      "TensorRT install (the dir containing include/NvInfer.h and lib/).")
  endif()
  target_include_directories(la_llm PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(la_llm PRIVATE ${TENSORRT_NVINFER})

  # CUDA runtime (KV cache device buffers, H2D/D2H copies).
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    target_link_libraries(la_llm PRIVATE CUDA::cudart)
  endif()

  # TRT-LLM C++ runtime (optional; only the TrtLlm backend uses it). If absent,
  # that backend's construction throws at runtime but the build still links.
  find_library(TRTLLM_LIB
    NAMES tensorrt_llm nvinfer_plugin_tensorrt_llm
    HINTS ${TRTLLM_ROOT} /usr/local/tensorrt_llm/lib)
  find_path(TRTLLM_INCLUDE_DIR tensorrt_llm/executor/executor.h
    HINTS ${TRTLLM_ROOT}/include /usr/local/tensorrt_llm/include)
  if(TRTLLM_LIB AND TRTLLM_INCLUDE_DIR)
    target_include_directories(la_llm PRIVATE ${TRTLLM_INCLUDE_DIR})
    target_link_libraries(la_llm PRIVATE ${TRTLLM_LIB})
    target_compile_definitions(la_llm PRIVATE LA_HAVE_TRTLLM)
    message(STATUS "la_llm: TRT-LLM found (${TRTLLM_LIB})")
  else()
    message(STATUS
      "la_llm: TRT-LLM not found; TrtLlm backend will throw at construction. "
      "StandaloneTrt backend (plain TensorRT) is fully built.")
  endif()
else()
  # ---- TRT-free build: factory stub only. ----------------------------------
  add_library(la_llm STATIC
    src/llm_engine_factory.cpp
  )
endif()

target_include_directories(la_llm
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(la_llm PUBLIC cxx_std_20)

add_library(la::llm ALIAS la_llm)
