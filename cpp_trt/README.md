# LocateAnything — C++/TensorRT Port

A C++/TensorRT port of NVIDIA **LocateAnything**.

This directory (`cpp_trt/`) contains the native implementation. It is organized
so that the **Phase 0** targets — the deterministic, model-free building blocks
(config, tokenizer, image preprocessing, output decoding, vision RoPE) — build
and test with **only a C++ toolchain**: no TensorRT, no TRT-LLM, no PyTorch, and
no model checkpoint are required.

CUDA kernels are optional and gated behind a CMake option (default **OFF**).
Every CUDA path has a pure-C++ CPU reference that always builds, so the project
configures and compiles even where `nvcc` and the host compiler do not
interoperate.

---

## Project layout

```
cpp_trt/
├── CMakeLists.txt          # root: project(locateanything_trt), C++20, options, testing
├── cmake/
│   └── FindTensorRT.cmake  # discovery module for later phases (UNUSED in Phase 0)
├── docker/
│   └── Dockerfile          # FROM nvcr.io/nvidia/tensorrt:25.06-py3 + cmake/g++/cargo
├── src/
│   ├── CMakeLists.txt      # aggregator: add_subdirectory for each module below
│   ├── config/             # (owned by another agent)
│   ├── tokenizer/          # (owned by another agent)
│   ├── preprocess/         # (owned by another agent)
│   ├── decode/             # (owned by another agent)
│   └── vision_rope/        # (owned by another agent)
├── tests/
│   └── CMakeLists.txt      # placeholder (test agent owns the rest of tests/)
├── .gitignore
└── README.md               # this file
```

### Modules (`src/`)

| Module        | Purpose |
|---------------|---------|
| `config`      | Model / runtime configuration: hyperparameters, image and grid sizes, special token ids, paths. The single source of truth other modules read. |
| `tokenizer`   | Text tokenization / detokenization for the prompt and model output (encode/decode, special tokens). |
| `preprocess`  | Image preprocessing: resize, normalize, layout conversion (HWC→CHW), and packing into the model input tensor. |
| `decode`      | Output decoding: turning raw model logits / coordinate predictions into structured detections (boxes / points), including any coordinate-token parsing. |
| `vision_rope` | Vision rotary positional embeddings (RoPE) used by the vision tower — frequency tables and the rotation applied to patch embeddings. |

Each module is an independent static library (conventionally `la_<module>`) and
may ship a CPU reference plus, in later phases, an optional CUDA kernel.

---

## Include directory convention

Every module exposes its **public** headers under:

```
cpp_trt/src/<module>/include/la/<module>/
```

A module's `CMakeLists.txt` therefore does:

```cmake
target_include_directories(la_<module> PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
```

and consumers include headers with the fully-qualified path, e.g.:

```cpp
#include "la/config/config.hpp"
#include "la/tokenizer/tokenizer.hpp"
#include "la/preprocess/preprocess.hpp"
#include "la/decode/decode.hpp"
#include "la/vision_rope/vision_rope.hpp"
```

The `la/<module>/` prefix keeps the include namespace collision-free and makes
the owning module obvious at every use site.

---

## Build instructions (Phase 0, no TensorRT)

Phase 0 targets need only CMake (≥ 3.20) and a C++20 compiler. GoogleTest is
required to configure the test tree (`BUILD_TESTING=ON` by default).

From inside `cpp_trt/`:

```bash
# Configure
cmake -B build -S .

# Build
cmake --build build

# Test
ctest --test-dir build --output-on-failure
```

To skip tests entirely:

```bash
cmake -B build -S . -DBUILD_TESTING=OFF
cmake --build build
```

### CMake options

| Option          | Default | Meaning |
|-----------------|---------|---------|
| `LA_BUILD_CUDA` | `OFF`   | Build optional CUDA kernels (Phase 1+; requires `nvcc`). When OFF, only the pure-C++ CPU references build. |
| `BUILD_TESTING` | `ON`    | Build and register the unit tests. |

> Enabling `LA_BUILD_CUDA` calls `enable_language(CUDA)`. On a host where
> `nvcc 13.1` and `gcc 15` do not interoperate, leave this **OFF** — the CPU
> references provide identical behavior for Phase 0.

---

## Phase 0 vs later phases

- **Phase 0 (this scaffolding):** pure-C++ modules; deterministic, unit-tested
  in isolation. No TensorRT / TRT-LLM / PyTorch / checkpoint dependency.
- **Phase 1+:** ONNX export, TensorRT engine build, and the inference runtime.
  These use `cmake/FindTensorRT.cmake`, optionally `-DLA_BUILD_CUDA=ON`, and the
  TensorRT container in `docker/Dockerfile`. Building **TRT-LLM 0.20**
  (pinned at commit `7c828d7`) is part of that later phase, not Phase 0.

---

## Docker

`docker/Dockerfile` is based on `nvcr.io/nvidia/tensorrt:25.06-py3` and adds the
C++ toolchain (`cmake`, `build-essential`), `ninja`, and Rust/`cargo`. It builds
the Phase 0 targets as-is and provides TensorRT for later phases. TRT-LLM is not
built in this image (later phase).

```bash
docker build -t locateanything-trt -f cpp_trt/docker/Dockerfile .
```
