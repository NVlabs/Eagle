// SPDX-License-Identifier: Apache-2.0
//
// la/tokenizer/bpe_tokenizer.hpp
//
// C++ RAII wrapper around the optional Rust FFI binding to the HuggingFace
// `tokenizers` crate (loading tokenizer.json).
//
// This wrapper ALWAYS compiles. The actual encode/decode functionality is only
// available when the library is built with LA_BUILD_TOKENIZER_FFI=ON (which
// links the Rust staticlib `la_tokenizer_ffi`). When the FFI is not compiled
// in, every method that requires a real backend throws std::runtime_error with
// a clear message, and `is_available()` returns false.
//
// Rationale: Phase 0 has no checkpoint / tokenizer.json available in this
// environment, and nvcc/gcc/cargo availability varies. The chat-template
// assembler (chat_template.hpp) is the part that is fully testable now; the BPE
// backend is wired but optional.

#ifndef LA_TOKENIZER_BPE_TOKENIZER_HPP
#define LA_TOKENIZER_BPE_TOKENIZER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace la {
namespace tokenizer {

/// Whether the Rust FFI tokenizer backend was compiled into this build.
/// Reflects the LA_BUILD_TOKENIZER_FFI CMake option at compile time.
bool ffi_backend_compiled();

/// RAII wrapper over the opaque Rust tokenizer handle.
///
/// All methods are no-throw to construct (the object can be created even when
/// the backend is absent, e.g. for type-completeness in tests), but loading a
/// tokenizer or encoding/decoding throws std::runtime_error when the backend is
/// not compiled in or when the underlying Rust call fails.
class BpeTokenizer {
public:
    BpeTokenizer() = default;
    ~BpeTokenizer();

    // Move-only (owns an opaque handle).
    BpeTokenizer(BpeTokenizer&&) noexcept;
    BpeTokenizer& operator=(BpeTokenizer&&) noexcept;
    BpeTokenizer(const BpeTokenizer&) = delete;
    BpeTokenizer& operator=(const BpeTokenizer&) = delete;

    /// Load a tokenizer from a tokenizer.json file path.
    /// Throws std::runtime_error if the FFI backend is absent or load fails.
    static BpeTokenizer from_file(const std::string& tokenizer_json_path);

    /// Load a tokenizer from an in-memory tokenizer.json string.
    /// Throws std::runtime_error if the FFI backend is absent or parse fails.
    static BpeTokenizer from_json(const std::string& tokenizer_json);

    /// True if this instance holds a live backend handle.
    bool is_available() const { return handle_ != nullptr; }

    /// Encode `text` into token ids.
    /// `add_special_tokens` controls whether the tokenizer's configured
    /// special-token post-processor is applied.
    /// Throws std::runtime_error on backend failure.
    std::vector<std::int32_t> encode(const std::string& text,
                                     bool add_special_tokens = false) const;

    /// Decode token ids back into a string.
    /// `skip_special_tokens` drops special tokens from the output.
    /// Throws std::runtime_error on backend failure.
    std::string decode(const std::vector<std::int32_t>& ids,
                       bool skip_special_tokens = false) const;

    /// Register additional special tokens with the tokenizer (e.g.
    /// <IMG_CONTEXT>, <img>, </img>). Returns the number of tokens that were
    /// newly added (0 if all already present).
    /// Throws std::runtime_error on backend failure.
    std::int64_t add_special_tokens(const std::vector<std::string>& tokens);

    /// Look up the id of a single token, or -1 if unknown.
    /// Throws std::runtime_error if the backend is absent.
    std::int32_t token_to_id(const std::string& token) const;

    /// Vocabulary size (including added tokens).
    /// Throws std::runtime_error if the backend is absent.
    std::int64_t vocab_size() const;

private:
    explicit BpeTokenizer(void* handle) : handle_(handle) {}
    void reset();

    void* handle_ = nullptr;  // opaque Rust-owned pointer
};

}  // namespace tokenizer
}  // namespace la

#endif  // LA_TOKENIZER_BPE_TOKENIZER_HPP
