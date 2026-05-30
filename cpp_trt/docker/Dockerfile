# ============================================================================
# LocateAnything C++/TensorRT — build & runtime image
#
# Base: NVIDIA TensorRT container. This already provides TensorRT, CUDA, and
# cuDNN, which are needed in Phase 1+ (engine build / inference runtime).
#
# Phase 0 (current) only needs the C++ toolchain (cmake, g++) and cargo; the
# scaffolding targets do NOT link TensorRT. They will, however, build fine
# inside this image as well.
#
# NOTE: Building TRT-LLM 0.20 (pinned at commit 7c828d7) is a SEPARATE,
# LATER PHASE. It is intentionally NOT done here to keep the Phase 0 image
# small and fast. A future Dockerfile stage / overlay will add it.
# ============================================================================
FROM nvcr.io/nvidia/tensorrt:25.06-py3

ARG DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# Toolchain: cmake, build-essential (g++, make), git, and Rust/cargo.
# build-essential gives a host C++ compiler; the base image's CUDA toolkit
# provides nvcc for the optional LA_BUILD_CUDA path in later phases.
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        ca-certificates \
        curl \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Rust / cargo (used by tooling; e.g. tokenizer fixture generation in later
# phases). Installed via rustup; PATH updated for all subsequent layers.
# ----------------------------------------------------------------------------
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal \
    && cargo --version

WORKDIR /workspace

# ----------------------------------------------------------------------------
# Phase 0 build (pure C++; no TensorRT linkage):
#   cmake -B build -S cpp_trt -G Ninja
#   cmake --build build
#   ctest --test-dir build --output-on-failure
#
# Phase 1+ will additionally:
#   - build TRT-LLM 0.20 @ 7c828d7
#   - configure with -DLA_BUILD_CUDA=ON and link TensorRT::TensorRT
# ----------------------------------------------------------------------------
CMD ["/bin/bash"]
