# LocateAnything C++/TensorRT Port

Phase 0 scaffold: CPU-only, TRT-free building blocks with tests.
Phase 1 adds an opt-in TensorRT path (default OFF) plus the Python export
pipeline that produces the ONNX / engine artifacts consumed by the C++ runtime.

## Layout

```
cpp_trt/
  src/
    config/      # runtime config structs + JSON parse
    tokenizer/   # BPE/SentencePiece-ish tokenizer port
    preprocess/  # image preprocessing (letterbox, normalize)
    decode/      # box/point decode utilities
    vision_rope/ # 2D rotary position embedding tables
    vision/      # TensorRT engine wrapper + end-to-end pipeline (LA_BUILD_TRT)
  tests/         # gtest unit tests (76 cases)
  export/        # Python export pipeline (PyTorch -> ONNX -> TRT engine)
  cmake/         # FindTensorRT.cmake
  docker/        # build image (TRT 10.11)
```

## Build (default: TRT-free)

```bash
cmake -B build -S . -DLA_BUILD_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

All 76 tests pass with no TensorRT, CUDA, or PyTorch installed. This is the
default configuration: `LA_BUILD_TRT` is OFF, so `cmake/FindTensorRT.cmake` is
never invoked and `src/vision/` is not descended into.

## Build options

| Option          | Default | Effect                                                              |
| --------------- | ------- | ------------------------------------------------------------------- |
| `BUILD_TESTING` | `ON`    | Build the gtest unit tests (76 cases).                              |
| `LA_BUILD_CUDA` | `OFF`   | Enable the CUDA language and CUDA-dependent targets.                |
| `LA_BUILD_TRT`  | `OFF`   | Find TensorRT and build the TRT-dependent vision targets.           |

### LA_BUILD_TRT

When `LA_BUILD_TRT=ON` the root `CMakeLists.txt` runs
`find_package(TensorRT REQUIRED)` (see `cmake/FindTensorRT.cmake`) and
`src/CMakeLists.txt` descends into `src/vision/` (added AFTER the Phase 0
modules so targets such as `la_vision_rope` are available to link). When OFF,
none of the TensorRT machinery is touched and the build stays exactly as in
Phase 0.

```bash
# TRT-enabled build (requires a TensorRT 10.x install)
cmake -B build -S . \
  -DLA_BUILD_CUDA=ON \
  -DLA_BUILD_TRT=ON \
  -DTensorRT_ROOT=/usr/local/tensorrt \
  -DBUILD_TESTING=ON
cmake --build build -j
```

`FindTensorRT.cmake` locates `nvinfer`, `nvonnxparser`, and (optionally)
`nvinfer_plugin` plus `NvInfer.h`, honoring `-DTensorRT_ROOT=...` (or the
`TensorRT_ROOT` / `TENSORRT_ROOT` environment variables) before falling back to
the CUDA toolkit root and standard system locations. On success it sets
`TensorRT_FOUND`, `TensorRT_INCLUDE_DIRS`, `TensorRT_LIBRARIES`, and the
imported target `TensorRT::TensorRT`.

## Export pipeline (Python)

The `export/` directory holds the Python code that converts the LocateAnything
checkpoint into the artifacts the C++/TRT runtime loads (ONNX, then a
serialized TensorRT engine). It is written against the real checkpoint and is
intended to be run on a machine that has PyTorch, the checkpoint, and TensorRT
available — not as part of the C++ build.

See `export/README.md` for the full, authoritative run order, required
environment, and the exact commands for each stage of the
PyTorch -> ONNX -> TensorRT engine conversion.

## Status

- [x] Phase 0: CPU building blocks + tests (76 cases, TRT-free default).
- [ ] Phase 1: TensorRT engine wrapper + end-to-end pipeline (`LA_BUILD_TRT`),
      Python export pipeline (`export/`).
- [ ] Later phase: TensorRT-LLM integration.
```
