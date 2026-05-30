# Op-level unit tests for the LocateAnything C++/TensorRT port (Phase 0).
#
# Owned by the test agent. Each test executable links exactly one la_* static
# library plus GTest::gtest_main, and is registered with add_test()
# (via gtest_discover_tests when available).
#
# These tests are pure-C++ (no CUDA, no TensorRT, no Torch). They check
# internal consistency and spec conformance against independent in-test
# reference implementations -- NOT PyTorch parity (impossible without a
# checkpoint / torch in this environment).

find_package(GTest REQUIRED)
include(GoogleTest OPTIONAL)

# Define a test executable that links one module library + GTest.
#   la_add_test(<test_name> <source> <module_lib_target>)
function(la_add_test test_name source module_lib)
  add_executable(${test_name} ${source})
  target_link_libraries(${test_name}
    PRIVATE ${module_lib} GTest::gtest GTest::gtest_main)
  target_compile_features(${test_name} PRIVATE cxx_std_17)
  if(COMMAND gtest_discover_tests)
    gtest_discover_tests(${test_name})
  else()
    add_test(NAME ${test_name} COMMAND ${test_name})
  endif()
endfunction()

# Resolve a module target, preferring the la::<module> alias and falling back
# to the plain la_<module> library name.
function(la_resolve_target out_var alias plain)
  if(TARGET ${alias})
    set(${out_var} ${alias} PARENT_SCOPE)
  elseif(TARGET ${plain})
    set(${out_var} ${plain} PARENT_SCOPE)
  else()
    message(FATAL_ERROR
      "Neither '${alias}' nor '${plain}' exists; configure src/ before tests/.")
  endif()
endfunction()

la_resolve_target(LA_DECODE_TGT      la::decode      la_decode)
la_resolve_target(LA_VISION_ROPE_TGT la::vision_rope la_vision_rope)
la_resolve_target(LA_PREPROCESS_TGT  la::preprocess  la_preprocess)
la_resolve_target(LA_CONFIG_TGT      la::config      la_config)
la_resolve_target(LA_TOKENIZER_TGT   la::tokenizer   la_tokenizer)

la_add_test(test_decode      test_decode.cpp      ${LA_DECODE_TGT})
la_add_test(test_vision_rope test_vision_rope.cpp ${LA_VISION_ROPE_TGT})
la_add_test(test_preprocess  test_preprocess.cpp  ${LA_PREPROCESS_TGT})
la_add_test(test_config      test_config.cpp      ${LA_CONFIG_TGT})
la_add_test(test_tokenizer   test_tokenizer.cpp   ${LA_TOKENIZER_TGT})

# ---------------------------------------------------------------------------
# Phase-2 PURE-C++ tests for la_generate (sampling / coord / parse) and la_api
# (prompts / parse_boxes / parse_points). Added by the Phase-2 test task.
#
# Uses the existing la_add_test(<test_name> <source> <module_lib>) helper +
# la_resolve_target convention above. These link only TRT-free static libs
# (la_generate, which PUBLIC-links la_config; and la_api) so they build in the
# default LA_BUILD_TRT=OFF configuration.
#
# These tests need C++20 (la_generate / la_api are cxx_std_20), so we bump the
# standard after la_add_test (which sets cxx_std_17) creates the targets.
# ---------------------------------------------------------------------------
la_resolve_target(LA_GENERATE_TGT la::generate la_generate)
la_resolve_target(LA_API_TGT      la::api      la_api)

la_add_test(test_generate test_generate.cpp ${LA_GENERATE_TGT})
la_add_test(test_api      test_api.cpp      ${LA_API_TGT})

# la_generate / la_api are compiled as C++20; their public headers (and the
# tests) require it. Override the cxx_std_17 the helper sets.
set_target_properties(test_generate test_api PROPERTIES
  CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
