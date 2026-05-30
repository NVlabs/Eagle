cmake_minimum_required(VERSION 3.24)

# ---------------------------------------------------------------------------
# la_vision — Vision Transformer (ViT) TensorRT engine wrapper (Phase 1)
# ---------------------------------------------------------------------------
# Two configurations, selected by the top-level option LA_BUILD_TRT:
#
#   LA_BUILD_TRT = OFF (default)
#       Header-only INTERFACE target. No TensorRT, no CUDA, nothing that could
#       fail to configure on a host without TRT. Callers can still
#       #include "la/vision/VisionEngine.h" and compile. (Linking against the
#       VisionEngine *implementation* without TRT is intentionally unsupported.)
#
#   LA_BUILD_TRT = ON
#       Static library la_vision that compiles VisionEngine.cpp, finds and links
#       TensorRT (via cmake/FindTensorRT.cmake) and the CUDA runtime.
#
# In both cases the alias la::vision is provided so downstream code links the
# same target name regardless of configuration.
# ---------------------------------------------------------------------------

if(LA_BUILD_TRT)
    # -----------------------------------------------------------------------
    # Full TensorRT build.
    # -----------------------------------------------------------------------
    find_package(TensorRT REQUIRED)
    find_package(CUDAToolkit REQUIRED)

    add_library(la_vision STATIC
        src/VisionEngine.cpp
    )
    add_library(la::vision ALIAS la_vision)

    target_include_directories(la_vision
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )

    target_compile_features(la_vision PUBLIC cxx_std_20)

    # LA_BUILD_TRT must be visible to the .cpp so its #ifdef body is active.
    target_compile_definitions(la_vision PUBLIC LA_BUILD_TRT)

    target_link_libraries(la_vision
        PUBLIC
            TensorRT::TensorRT
            CUDA::cudart
    )

    target_compile_options(la_vision PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    )

else()
    # -----------------------------------------------------------------------
    # Default (no TRT): header-only interface target. Configures cleanly with
    # no TensorRT or CUDA present, so the Phase 0 build and its 76 gtest cases
    # are unaffected.
    # -----------------------------------------------------------------------
    add_library(la_vision INTERFACE)
    add_library(la::vision ALIAS la_vision)

    target_include_directories(la_vision
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )

    target_compile_features(la_vision INTERFACE cxx_std_20)
endif()

# No gtest targets here: the VisionEngine tests require a live TensorRT engine
# and GPU, which are out of scope for the sandbox build. Pure-C++ unit tests for
# this module would have to mock the entire TRT runtime, providing little value.
