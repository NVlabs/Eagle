# la_tokenizer

Tokenizer module for the LocateAnything C++ port (Phase 0).

## Layout

```
src/tokenizer/
  CMakeLists.txt                       # builds static lib `la_tokenizer`
  include/la/tokenizer/
    chat_template.hpp                  # pure-C++ chat-template assembler API
    bpe_tokenizer.hpp                  # RAII wrapper over optional Rust FFI
    tokenizer_ffi.h                    # stable C ABI (Rust mirrors this)
  src/
    chat_template.cpp                  # pure C++, no deps (testable now)
    bpe_tokenizer.cpp                  # FFI wrapper + no-backend stub
  rust/
    Cargo.toml                         # crate `la_tokenizer_ffi`
    src/lib.rs                         # C ABI binding to HF `tokenizers`
  test/
    test_chat_template.cpp             # GTest, runs without checkpoint
    test_bpe_tokenizer.cpp            # GTest, degradation contract
```

## What is testable now (no checkpoint / PyTorch / TRT)

The **chat-template assembler** (`chat_template.hpp` / `.cpp`) is pure C++ with
no external dependencies. It reproduces the reference string layout from
`Embodied/eaglevl/utils/locany/processing_locateanything.py`
(`py_apply_chat_template` + the `<image-k>` expansion). Spec Section 6.

Key behavior:

- `<|im_start|>`/`<|im_end|>` framing, one turn per message.
- Optional default-system injection (`ApplyOptions::inject_default_system`),
  suppressed when the conversation already starts with a `System` message.
- Optional trailing generation prompt `"<|im_start|>assistant\n"`.
- Image expansion: `<image-k>` ->
  `<img>` + `(<IMG_CONTEXT> * ((h*w)//4))` + `</img>`, where `(h,w)` is the
  per-image feature grid. If a turn has images but no `<image-k>` markers, the
  image blocks are prepended (each on its own line).

All special-token strings are configurable via `ChatTemplateConfig` so a caller
holding the real `tokenizer_config.json` can guarantee byte-exact parity.

## Building

Default (no Rust, always builds with the C++ toolchain alone):

```bash
cmake -S src/tokenizer -B build -DLA_BUILD_TOKENIZER_FFI=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

## Optional Rust BPE backend (`LA_BUILD_TOKENIZER_FFI=ON`)

The BPE tokenizer is a thin FFI binding to the HuggingFace `tokenizers` Rust
crate, loading `tokenizer.json`. It is **OFF by default** so the module builds
without `cargo` or a checkpoint.

Two integration paths (auto-detected by CMake):

1. **Corrosion** (preferred). If `find_package(Corrosion)` succeeds, the crate
   is imported via `corrosion_import_crate` and linked as `la_tokenizer_ffi`.

2. **Manual cargo** (fallback, no extra CMake deps). CMake invokes
   `cargo build --release --manifest-path rust/Cargo.toml` and links the
   resulting `libla_tokenizer_ffi.a`. Equivalent manual step:

   ```bash
   cd src/tokenizer/rust
   cargo build --release
   # -> target/release/libla_tokenizer_ffi.a
   ```

Enable:

```bash
cmake -S src/tokenizer -B build -DLA_BUILD_TOKENIZER_FFI=ON
cmake --build build
```

When the backend is absent, `BpeTokenizer` methods throw
`std::runtime_error` and `ffi_backend_compiled()` returns `false`.

## C ABI contract

See `include/la/tokenizer/tokenizer_ffi.h`. The Rust crate implements exactly
these symbols; all are panic-safe (`catch_unwind`) and nullptr-tolerant.
Returned buffers are Rust-owned and freed via `la_tok_free_string` /
`la_tok_free_ids`; handles via `la_tok_free`.

## Parity caveats (cannot be validated in this environment)

- No model checkpoint / `tokenizer.json` is available here, so BPE
  encode/decode parity against the Python `Qwen2Tokenizer(Fast)` is NOT
  verified. The FFI path is wired and compiles but is exercised only when a
  real `tokenizer.json` is supplied at runtime.
- The exact spelling/order of special tokens and the default system prompt
  match the documented Qwen2 + InternVL conventions; for a specific checkpoint
  they should be confirmed against its `tokenizer_config.json` and overridden
  via `ChatTemplateConfig` if they differ.
- The `tokenizers` crate version pin (0.20) and feature set may need adjustment
  to match the producing HuggingFace version for guaranteed merge/normalizer
  parity.
