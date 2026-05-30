// SPDX-License-Identifier: Apache-2.0
//
// llm_engine_factory.cpp
//
// make_llm_engine() dispatch.
//
//   * When LA_BUILD_TRT is defined, this dispatches to the two real
//     implementations (declared below, defined in llm_engine_trtllm.cpp and
//     llm_engine_standalone_trt.cpp).
//   * When LA_BUILD_TRT is NOT defined (the default, TRT-free build used for the
//     gtest suite on a machine without TensorRT/TRT-LLM), this is the ONLY
//     translation unit in la_llm. It provides a throwing stub so the
//     make_llm_engine symbol exists and callers/tests link, but constructing an
//     actual engine fails loudly with LlmEngineError.
//
// This file contains NO TensorRT / TRT-LLM includes at all; all such includes
// live inside the two impl .cpp files behind #ifdef LA_BUILD_TRT.

#include "la/llm/LlmEngine.h"

#include <memory>
#include <string>

namespace la {
namespace llm {

#ifdef LA_BUILD_TRT

// Defined in the backend-specific translation units (also guarded by
// LA_BUILD_TRT). Declared here to keep all TRT headers out of this file.
std::unique_ptr<ILlmEngine> make_trtllm_engine(const LlmEngineConfig& cfg);
std::unique_ptr<ILlmEngine> make_standalone_trt_engine(const LlmEngineConfig& cfg);

std::unique_ptr<ILlmEngine> make_llm_engine(const LlmEngineConfig& cfg) {
  switch (cfg.backend) {
    case Backend::TrtLlm:
      return make_trtllm_engine(cfg);
    case Backend::StandaloneTrt:
      return make_standalone_trt_engine(cfg);
  }
  throw LlmEngineError("make_llm_engine: unknown Backend enum value");
}

#else  // !LA_BUILD_TRT  -- TRT-free stub.

std::unique_ptr<ILlmEngine> make_llm_engine(const LlmEngineConfig& cfg) {
  const char* name = (cfg.backend == Backend::TrtLlm) ? "TrtLlm" : "StandaloneTrt";
  throw LlmEngineError(
      std::string("make_llm_engine: this binary was built WITHOUT LA_BUILD_TRT, "
                  "so no LLM engine backend is available (requested backend = ") +
      name +
      ", engine_path = '" + cfg.engine_path +
      "'). Rebuild with -DLA_BUILD_TRT=ON on a machine that has TensorRT "
      "(and optionally TRT-LLM) to construct a real engine.");
}

#endif  // LA_BUILD_TRT

}  // namespace la::llm  (closed below)
}  // namespace la
