// SPDX-License-Identifier: Apache-2.0
//
// Tests for the BpeTokenizer wrapper.
//
// In Phase 0 (no checkpoint / tokenizer.json, FFI default OFF) we verify the
// degradation contract: without the FFI backend, the wrapper reports
// unavailable and throws on backend operations. When built WITH the FFI
// backend, these throwing tests are skipped (the real path requires a
// tokenizer.json, which is not present in this environment).

#include "la/tokenizer/bpe_tokenizer.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

using la::tokenizer::BpeTokenizer;

TEST(BpeTokenizer, ReportsBackendCompileFlagConsistently) {
    // The free function must agree with the build-time configuration.
#ifdef LA_BUILD_TOKENIZER_FFI
    EXPECT_TRUE(la::tokenizer::ffi_backend_compiled());
#else
    EXPECT_FALSE(la::tokenizer::ffi_backend_compiled());
#endif
}

TEST(BpeTokenizer, DefaultConstructedIsUnavailable) {
    BpeTokenizer tok;
    EXPECT_FALSE(tok.is_available());
}

#ifndef LA_BUILD_TOKENIZER_FFI
TEST(BpeTokenizer, FromFileThrowsWithoutBackend) {
    EXPECT_THROW({ BpeTokenizer::from_file("nonexistent.json"); },
                 std::runtime_error);
}

TEST(BpeTokenizer, FromJsonThrowsWithoutBackend) {
    EXPECT_THROW({ BpeTokenizer::from_json("{}"); }, std::runtime_error);
}

TEST(BpeTokenizer, EncodeThrowsWithoutBackend) {
    BpeTokenizer tok;  // no handle
    EXPECT_THROW({ tok.encode("hello"); }, std::runtime_error);
    EXPECT_THROW({ tok.decode({1, 2, 3}); }, std::runtime_error);
    EXPECT_THROW({ tok.vocab_size(); }, std::runtime_error);
}
#endif  // !LA_BUILD_TOKENIZER_FFI

TEST(BpeTokenizer, MoveLeavesSourceUnavailable) {
    BpeTokenizer a;
    BpeTokenizer b(std::move(a));
    EXPECT_FALSE(a.is_available());
    EXPECT_FALSE(b.is_available());
}
