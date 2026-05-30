# SPDX-License-Identifier: Apache-2.0
#
# la_tokenizer - tokenizer module for the LocateAnything C++ port (Phase 0).
#
# Builds a static library `la_tokenizer` containing:
#   * chat_template.cpp  - pure C++ chat-template assembler (always built)
#   * bpe_tokenizer.cpp  - RAII wrapper over the OPTIONAL Rust FFI backend
#
# The Rust FFI backend (HuggingFace tokenizers, loading tokenizer.json) is
# OFF by default so the module always builds without cargo / a checkpoint.
# Enable with -DLA_BUILD_TOKENIZER_FFI=ON (requires cargo on PATH).
#
# This file is self-contained: it can be added via add_subdirectory() from a
# parent project, or configured standalone for development/testing.

cmake_minimum_required(VERSION 3.20)

# Allow standalone configuration of just this module.
if(NOT DEFINED PROJECT_NAME)
  project(la_tokenizer LANGUAGES CXX)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(LA_BUILD_TOKENIZER_FFI
       "Build and link the Rust FFI tokenizer backend (requires cargo)" OFF)
option(LA_TOKENIZER_BUILD_TESTS
       "Build la_tokenizer unit tests (requires GTest)" ON)

# ---------------------------------------------------------------------------
# Core static library (always builds, no external deps)
# ---------------------------------------------------------------------------
add_library(la_tokenizer STATIC
  src/chat_template.cpp
  src/bpe_tokenizer.cpp
)
add_library(la::tokenizer ALIAS la_tokenizer)

target_include_directories(la_tokenizer
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(la_tokenizer PUBLIC cxx_std_17)
set_target_properties(la_tokenizer PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_EXTENSIONS OFF
)

if(MSVC)
  target_compile_options(la_tokenizer PRIVATE /W4)
else()
  target_compile_options(la_tokenizer PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# Optional Rust FFI backend
# ---------------------------------------------------------------------------
if(LA_BUILD_TOKENIZER_FFI)
  set(LA_RUST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rust")

  # Prefer Corrosion if available (clean, dependency-tracked integration).
  find_package(Corrosion QUIET)

  if(Corrosion_FOUND)
    message(STATUS "la_tokenizer: using Corrosion for Rust FFI integration")
    corrosion_import_crate(MANIFEST_PATH "${LA_RUST_DIR}/Cargo.toml")
    # Corrosion exposes the staticlib as an imported target named after the
    # crate's [lib] name: la_tokenizer_ffi.
    target_link_libraries(la_tokenizer PUBLIC la_tokenizer_ffi)
  else()
    # ----- Manual cargo fallback (documented) -----
    # Build the staticlib with `cargo build --release` and link the resulting
    # archive directly. This requires `cargo` on PATH.
    message(STATUS "la_tokenizer: Corrosion not found; using manual cargo build")
    find_program(CARGO_EXECUTABLE cargo)
    if(NOT CARGO_EXECUTABLE)
      message(FATAL_ERROR
        "LA_BUILD_TOKENIZER_FFI=ON but 'cargo' was not found on PATH. "
        "Install Rust or set -DLA_BUILD_TOKENIZER_FFI=OFF.")
    endif()

    set(LA_RUST_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/rust-target")
    set(LA_RUST_STATICLIB
        "${LA_RUST_TARGET_DIR}/release/libla_tokenizer_ffi.a")

    add_custom_command(
      OUTPUT "${LA_RUST_STATICLIB}"
      COMMAND ${CMAKE_COMMAND} -E env
              CARGO_TARGET_DIR=${LA_RUST_TARGET_DIR}
              ${CARGO_EXECUTABLE} build --release
              --manifest-path "${LA_RUST_DIR}/Cargo.toml"
      WORKING_DIRECTORY "${LA_RUST_DIR}"
      DEPENDS
        "${LA_RUST_DIR}/Cargo.toml"
        "${LA_RUST_DIR}/src/lib.rs"
      COMMENT "Building Rust FFI staticlib la_tokenizer_ffi (cargo --release)"
      VERBATIM
    )
    add_custom_target(la_tokenizer_ffi_build DEPENDS "${LA_RUST_STATICLIB}")

    add_library(la_tokenizer_ffi STATIC IMPORTED GLOBAL)
    set_target_properties(la_tokenizer_ffi PROPERTIES
      IMPORTED_LOCATION "${LA_RUST_STATICLIB}")
    add_dependencies(la_tokenizer_ffi la_tokenizer_ffi_build)

    target_link_libraries(la_tokenizer PUBLIC la_tokenizer_ffi)

    # The Rust staticlib needs the platform libs it links against.
    if(UNIX AND NOT APPLE)
      target_link_libraries(la_tokenizer PUBLIC pthread dl m)
    endif()
  endif()

  target_compile_definitions(la_tokenizer PUBLIC LA_BUILD_TOKENIZER_FFI=1)
  message(STATUS "la_tokenizer: Rust FFI backend ENABLED")
else()
  message(STATUS
    "la_tokenizer: Rust FFI backend DISABLED "
    "(set -DLA_BUILD_TOKENIZER_FFI=ON to enable)")
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(LA_TOKENIZER_BUILD_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    enable_testing()
    add_executable(la_tokenizer_tests
      test/test_chat_template.cpp
      test/test_bpe_tokenizer.cpp
    )
    target_link_libraries(la_tokenizer_tests
      PRIVATE la_tokenizer GTest::gtest GTest::gtest_main)
    target_compile_features(la_tokenizer_tests PRIVATE cxx_std_17)
    include(GoogleTest)
    gtest_discover_tests(la_tokenizer_tests)
    message(STATUS "la_tokenizer: tests ENABLED")
  else()
    message(STATUS
      "la_tokenizer: GTest not found; tests DISABLED")
  endif()
endif()
