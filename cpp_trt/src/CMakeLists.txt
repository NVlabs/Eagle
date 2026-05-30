# ============================================================================
# src/ aggregator
#
# Each module is an independent subdirectory that exposes its public headers
# under  src/<module>/include/la/<module>/  and (typically) defines a static
# library target named  la_<module>.
#
# These subdirectories are owned by other agents. This aggregator only wires
# them into the build. If a module subdirectory does not yet contain its own
# CMakeLists.txt the configure step will fail — that is expected until the
# owning agent has populated it.
# ============================================================================

add_subdirectory(config)
add_subdirectory(tokenizer)
add_subdirectory(preprocess)
add_subdirectory(decode)
add_subdirectory(vision_rope)

# LLM engine abstraction (la_llm / la::llm). ALWAYS builds: in the default
# TRT-free build only the throwing make_llm_engine() factory stub + the public
# interface header (la/llm/LlmEngine.h) are compiled, so nothing TensorRT is
# linked. la_generate (the decode loop) depends on this interface.
add_subdirectory(llm)

# Generation driver (la_generate / la::generate): NTP/AR decode loop, greedy +
# repetition-penalty sampling, coordinate detokenize, box/point id-stream
# parsing. Pure C++; drives la::llm::ILlmEngine + la::config::TokenIds. Added
# after llm + config so those targets exist.
add_subdirectory(generate)

# Public worker-style API + prompt builders. la_api ALWAYS builds (prompt
# builders + result parsers are pure C++); only the predict() engine pipeline
# inside it is guarded by LA_BUILD_TRT, so the default TRT-free build links
# nothing extra. Added after the Phase 0 modules it composes.
add_subdirectory(api)

# TensorRT-dependent vision targets (engine wrapper + end-to-end pipeline).
# Added AFTER the Phase 0 modules above so targets like la_vision_rope are
# already defined and can be linked by the vision targets.
#
# The vision/ subdirectory is owned by a separate agent and its own
# CMakeLists.txt is responsible for honoring LA_BUILD_TRT (i.e. it should
# only declare TRT-linked targets when LA_BUILD_TRT is ON). We still guard
# the add_subdirectory() here so that:
#   * the default TRT-free build never descends into TRT code, and
#   * the configure step does not fail before the directory has landed.
if(LA_BUILD_TRT)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vision/CMakeLists.txt")
    add_subdirectory(vision)
  else()
    message(WARNING
      "LA_BUILD_TRT=ON but src/vision/CMakeLists.txt is missing; "
      "skipping TensorRT vision targets.")
  endif()
endif()
