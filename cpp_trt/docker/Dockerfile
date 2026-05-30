# Build image for the LocateAnything C++/TensorRT port.
#
# Base: NGC TensorRT 10.11 container (CUDA + TensorRT 10.x headers/libs).
# This image supports BOTH builds:
#   * the default TRT-free Phase 0 build (LA_BUILD_TRT=OFF), and
#   * the opt-in TRT build (LA_BUILD_TRT=ON), which links nvinfer /
#     nvonnxparser / nvinfer_plugin found under the container's TensorRT install.
#
# Note: TensorRT-LLM is a later phase and is intentionally NOT installed here.
FROM nvcr.io/nvidia/tensorrt:25.05-py3

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build g++ git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

# The NGC TensorRT image installs TensorRT under /usr (apt layout), so
# FindTensorRT.cmake locates it without an explicit TensorRT_ROOT. Override
# with -DTensorRT_ROOT=... if you mount a custom TensorRT tarball.

# Default build: TRT-free, builds and runs the 76 unit tests. This is the
# verification gate and must stay green.
RUN cmake -B build -S . -DLA_BUILD_CUDA=OFF -DBUILD_TESTING=ON \
    && cmake --build build -j \
    && ctest --test-dir build --output-on-failure

# TRT-enabled build (Phase 1). Lives in a separate build dir so it never
# disturbs the default build above. Enabled by passing the build arg.
#   docker build --build-arg LA_BUILD_TRT=ON .
ARG LA_BUILD_TRT=OFF
RUN if [ "$LA_BUILD_TRT" = "ON" ]; then \
      cmake -B build-trt -S . \
        -DLA_BUILD_CUDA=ON -DLA_BUILD_TRT=ON -DBUILD_TESTING=ON \
      && cmake --build build-trt -j ; \
    fi
