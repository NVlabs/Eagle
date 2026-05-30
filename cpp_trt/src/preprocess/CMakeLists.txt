# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# la_preprocess: LocateAnything image-preprocessing static library.
#
# Always builds a pure-C++ CPU reference. The CUDA/NPP path is OPTIONAL and
# compiled only when LA_BUILD_CUDA=ON (default OFF), because nvcc 13.1 + gcc 15
# may mismatch in this environment.

cmake_minimum_required(VERSION 3.18)

# Standalone-friendly: if this directory is configured on its own, declare a
# project so it can be built without the parent tree.
if(NOT DEFINED PROJECT_NAME)
  project(la_preprocess LANGUAGES CXX)
endif()

option(LA_BUILD_CUDA "Build the optional CUDA/NPP preprocessing path" OFF)
option(LA_BUILD_TESTS "Build la_preprocess unit tests" ON)

set(LA_PREPROCESS_SOURCES
  src/preprocess_cpu.cpp
)

add_library(la_preprocess STATIC ${LA_PREPROCESS_SOURCES})

target_include_directories(la_preprocess
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(la_preprocess PUBLIC cxx_std_17)
set_target_properties(la_preprocess PROPERTIES
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

if(LA_BUILD_CUDA)
  enable_language(CUDA)
  target_sources(la_preprocess PRIVATE cuda/preprocess_cuda.cu)
  set_target_properties(la_preprocess PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )
  target_compile_definitions(la_preprocess PUBLIC LA_BUILD_CUDA)
  # CUDA sources need the public headers too.
  target_include_directories(la_preprocess PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(LA_BUILD_TESTS)
  enable_testing()
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    # Fall back to system libs (Ubuntu: /usr/lib/x86_64-linux-gnu/libgtest.a).
    find_library(GTEST_LIB NAMES gtest)
    find_library(GTEST_MAIN_LIB NAMES gtest_main)
    find_path(GTEST_INC gtest/gtest.h)
  endif()

  add_executable(la_preprocess_tests test/test_preprocess.cpp)
  target_link_libraries(la_preprocess_tests PRIVATE la_preprocess)
  # Tests reach into the internal bicubic header (src/) to test it directly.
  target_include_directories(la_preprocess_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

  if(GTest_FOUND)
    target_link_libraries(la_preprocess_tests PRIVATE GTest::gtest GTest::gtest_main)
  elseif(GTEST_LIB AND GTEST_MAIN_LIB)
    target_include_directories(la_preprocess_tests PRIVATE ${GTEST_INC})
    target_link_libraries(la_preprocess_tests PRIVATE
      ${GTEST_MAIN_LIB} ${GTEST_LIB} pthread)
  else()
    message(WARNING "gtest not found; la_preprocess_tests will not link.")
  endif()

  add_test(NAME la_preprocess_tests COMMAND la_preprocess_tests)
endif()
