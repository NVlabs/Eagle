# cpp_trt/src/generate/CMakeLists.txt
#
# Builds the static library `la_generate` (alias la::generate): the Phase-2
# NTP/AR decode loop, greedy + repetition-penalty sampling, coordinate
# detokenization, and box/point parsing for the LocateAnything C++/TRT port.
#
# Design constraints (matches src/decode, src/config conventions):
#   * PURE C++ -- no TensorRT, no TRT-LLM, no PyTorch, no CUDA required. The
#     decode loop drives the abstract la::llm::LlmEngine interface (header-only,
#     src/llm/include/la/llm/LlmEngine.h) and a host embedding callback, so the
#     concrete engine (behind LA_BUILD_TRT) is never linked here. The default
#     build and these gtests compile with the C++ toolchain alone.
#   * Self-contained: can be configured standalone or add_subdirectory()'d.
#
# Module convention: src/<m>/{CMakeLists.txt, include/la/<m>/*.hpp, src/*.cpp}
#   -> static lib la_<m>, alias la::<m>.
#
# Dependencies (spec): la::config (TokenIds), la::decode (build_ar_mask /
# build_mtp_window_mask / build_position_ids), la::llm (LlmEngine interface
# only). la_config + la_decode are real STATIC targets and propagate their
# include dirs. la::llm is a header-only stub with no CMakeLists/target yet, so
# we add its include directory directly.

cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED PROJECT_NAME)
  project(la_generate LANGUAGES CXX)
endif()

option(LA_GENERATE_BUILD_TESTS "Build la_generate gtest unit tests" ON)

add_library(la_generate STATIC
  src/sampling.cpp
  src/coord.cpp
  src/parse.cpp
  src/decode_loop.cpp
)

target_include_directories(la_generate PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(la_generate PUBLIC cxx_std_20)

set_target_properties(la_generate PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
  target_compile_options(la_generate PRIVATE /W4)
else()
  target_compile_options(la_generate PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(la::generate ALIAS la_generate)

# --- Dependencies -----------------------------------------------------------
# Real library targets: link for headers + symbols.
#   la_config : TokenIds (+ out-of-line Fallback/Validate symbols).
#   la_llm    : the ILlmEngine interface header (LlmEngine.h) + the TRT-free
#               make_llm_engine() stub. ALWAYS builds (factory stub) so linking
#               it keeps the default build TRT-free.
# la_decode is NOT linked: the AR path passes a null mask (engine built-in
# causal) and does not call build_ar_mask/build_position_ids at runtime; the
# Phase-3 MTP path will add the la_decode dependency when it lands.
foreach(_dep la_config la_llm)
  if(TARGET ${_dep})
    target_link_libraries(la_generate PUBLIC ${_dep})
  endif()
endforeach()

# Fallbacks for standalone configuration (module built outside the full tree):
# add sibling include dirs directly when their library targets are absent.
if(NOT TARGET la_config)
  set(_inc "${CMAKE_CURRENT_SOURCE_DIR}/../config/include")
  if(EXISTS "${_inc}/la/config/token_ids.hpp")
    target_include_directories(la_generate PUBLIC $<BUILD_INTERFACE:${_inc}>)
  endif()
endif()
if(NOT TARGET la_llm)
  set(_inc "${CMAKE_CURRENT_SOURCE_DIR}/../llm/include")
  if(EXISTS "${_inc}/la/llm/LlmEngine.h")
    target_include_directories(la_generate PUBLIC $<BUILD_INTERFACE:${_inc}>)
  else()
    message(WARNING "la_generate: la/llm/LlmEngine.h not found; decode_loop "
                    "will fail to compile until la::llm is present.")
  endif()
endif()

# --- Tests (pure C++, no TRT). Mirrors src/decode/CMakeLists.txt. -----------
if(LA_GENERATE_BUILD_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND OR TARGET GTest::gtest_main)
    enable_testing()
    add_executable(la_generate_tests
      tests/test_sampling.cpp
      tests/test_coord.cpp
      tests/test_parse.cpp
      tests/test_decode_loop.cpp
    )
    target_link_libraries(la_generate_tests PRIVATE la_generate GTest::gtest_main)
    set_target_properties(la_generate_tests PROPERTIES
      CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    include(GoogleTest)
    gtest_discover_tests(la_generate_tests)
  else()
    message(STATUS "la_generate: GTest not found; skipping tests "
                   "(set LA_GENERATE_BUILD_TESTS=OFF to silence).")
  endif()
endif()
