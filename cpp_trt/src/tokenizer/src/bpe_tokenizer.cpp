// SPDX-License-Identifier: Apache-2.0
//
// bpe_tokenizer.cpp - RAII C++ wrapper around the optional Rust FFI backend.
//
// When LA_BUILD_TOKENIZER_FFI is defined (CMake option ON), this translation
// unit links against the C ABI in tokenizer_ffi.h (implemented by the Rust
// staticlib). Otherwise every backend-requiring call throws a clear
// std::runtime_error, and the file still compiles and links standalone.

#include "la/tokenizer/bpe_tokenizer.hpp"

#include <stdexcept>
#include <utility>

#ifdef LA_BUILD_TOKENIZER_FFI
#include "la/tokenizer/tokenizer_ffi.h"
#endif

namespace la {
namespace tokenizer {

bool ffi_backend_compiled() {
#ifdef LA_BUILD_TOKENIZER_FFI
    return true;
#else
    return false;
#endif
}

#ifndef LA_BUILD_TOKENIZER_FFI

// ----------------------------------------------------------------------------
// Stub implementation: no Rust backend compiled in.
// ----------------------------------------------------------------------------

namespace {
[[noreturn]] void no_backend() {
    throw std::runtime_error(
        "la::tokenizer::BpeTokenizer: Rust FFI backend not compiled in "
        "(reconfigure with -DLA_BUILD_TOKENIZER_FFI=ON and a working cargo "
        "toolchain).");
}
}  // namespace

BpeTokenizer::~BpeTokenizer() = default;

BpeTokenizer::BpeTokenizer(BpeTokenizer&& other) noexcept
    : handle_(other.handle_) {
    other.handle_ = nullptr;
}

BpeTokenizer& BpeTokenizer::operator=(BpeTokenizer&& other) noexcept {
    if (this != &other) {
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

void BpeTokenizer::reset() { handle_ = nullptr; }

BpeTokenizer BpeTokenizer::from_file(const std::string&) { no_backend(); }
BpeTokenizer BpeTokenizer::from_json(const std::string&) { no_backend(); }

std::vector<std::int32_t> BpeTokenizer::encode(const std::string&, bool) const {
    no_backend();
}
std::string BpeTokenizer::decode(const std::vector<std::int32_t>&, bool) const {
    no_backend();
}
std::int64_t BpeTokenizer::add_special_tokens(
    const std::vector<std::string>&) {
    no_backend();
}
std::int32_t BpeTokenizer::token_to_id(const std::string&) const {
    no_backend();
}
std::int64_t BpeTokenizer::vocab_size() const { no_backend(); }

#else  // LA_BUILD_TOKENIZER_FFI

// ----------------------------------------------------------------------------
// Real implementation backed by the Rust C ABI.
// ----------------------------------------------------------------------------

namespace {

la_tok_handle* as_handle(void* p) {
    return static_cast<la_tok_handle*>(p);
}
const la_tok_handle* as_handle(const void* p) {
    return static_cast<const la_tok_handle*>(p);
}

[[noreturn]] void throw_status(const char* what, la_tok_status st) {
    throw std::runtime_error(std::string("la::tokenizer::BpeTokenizer: ") +
                             what + " failed (status=" + std::to_string(st) +
                             ")");
}

void require_handle(const void* h) {
    if (h == nullptr) {
        throw std::runtime_error(
            "la::tokenizer::BpeTokenizer: no tokenizer loaded");
    }
}

}  // namespace

BpeTokenizer::~BpeTokenizer() { reset(); }

BpeTokenizer::BpeTokenizer(BpeTokenizer&& other) noexcept
    : handle_(other.handle_) {
    other.handle_ = nullptr;
}

BpeTokenizer& BpeTokenizer::operator=(BpeTokenizer&& other) noexcept {
    if (this != &other) {
        reset();
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

void BpeTokenizer::reset() {
    if (handle_ != nullptr) {
        la_tok_free(as_handle(handle_));
        handle_ = nullptr;
    }
}

BpeTokenizer BpeTokenizer::from_file(const std::string& path) {
    la_tok_status st = LA_TOK_OK;
    la_tok_handle* h = la_tok_from_file(path.c_str(), &st);
    if (h == nullptr) {
        throw_status("from_file", st);
    }
    return BpeTokenizer(static_cast<void*>(h));
}

BpeTokenizer BpeTokenizer::from_json(const std::string& json) {
    la_tok_status st = LA_TOK_OK;
    la_tok_handle* h = la_tok_from_json(json.data(), json.size(), &st);
    if (h == nullptr) {
        throw_status("from_json", st);
    }
    return BpeTokenizer(static_cast<void*>(h));
}

std::vector<std::int32_t> BpeTokenizer::encode(const std::string& text,
                                               bool add_special_tokens) const {
    require_handle(handle_);
    la_tok_ids ids{nullptr, 0};
    la_tok_status st =
        la_tok_encode(as_handle(handle_), text.data(), text.size(),
                      add_special_tokens ? 1 : 0, &ids);
    if (st != LA_TOK_OK) {
        throw_status("encode", st);
    }
    std::vector<std::int32_t> out;
    if (ids.data != nullptr && ids.len > 0) {
        out.assign(ids.data, ids.data + ids.len);
    }
    la_tok_free_ids(ids);
    return out;
}

std::string BpeTokenizer::decode(const std::vector<std::int32_t>& ids,
                                 bool skip_special_tokens) const {
    require_handle(handle_);
    char* out_str = nullptr;
    la_tok_status st = la_tok_decode(
        as_handle(handle_), ids.empty() ? nullptr : ids.data(), ids.size(),
        skip_special_tokens ? 1 : 0, &out_str);
    if (st != LA_TOK_OK) {
        throw_status("decode", st);
    }
    std::string out = (out_str != nullptr) ? std::string(out_str) : std::string();
    la_tok_free_string(out_str);
    return out;
}

std::int64_t BpeTokenizer::add_special_tokens(
    const std::vector<std::string>& tokens) {
    require_handle(handle_);
    std::vector<const char*> cstrs;
    cstrs.reserve(tokens.size());
    for (const auto& t : tokens) {
        cstrs.push_back(t.c_str());
    }
    std::int64_t added = 0;
    la_tok_status st = la_tok_add_special_tokens(
        as_handle(handle_), cstrs.empty() ? nullptr : cstrs.data(),
        cstrs.size(), &added);
    if (st != LA_TOK_OK) {
        throw_status("add_special_tokens", st);
    }
    return added;
}

std::int32_t BpeTokenizer::token_to_id(const std::string& token) const {
    require_handle(handle_);
    std::int32_t id = -1;
    la_tok_status st =
        la_tok_token_to_id(as_handle(handle_), token.c_str(), &id);
    if (st != LA_TOK_OK) {
        throw_status("token_to_id", st);
    }
    return id;
}

std::int64_t BpeTokenizer::vocab_size() const {
    require_handle(handle_);
    std::int64_t n = 0;
    la_tok_status st = la_tok_vocab_size(as_handle(handle_), &n);
    if (st != LA_TOK_OK) {
        throw_status("vocab_size", st);
    }
    return n;
}

#endif  // LA_BUILD_TOKENIZER_FFI

}  // namespace tokenizer
}  // namespace la
