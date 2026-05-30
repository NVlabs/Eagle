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
