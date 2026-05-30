/* SPDX-License-Identifier: Apache-2.0
 *
 * la/tokenizer/tokenizer_ffi.h
 *
 * Stable C ABI exposed by the Rust crate `la_tokenizer_ffi`
 * (cpp_trt/src/tokenizer/rust/). This header is the single source of truth for
 * the FFI contract; the Rust lib.rs implements exactly these symbols and the
 * C++ wrapper (bpe_tokenizer.cpp) consumes them when LA_BUILD_TOKENIZER_FFI=ON.
 *
 * Ownership / lifetime conventions:
 *   - la_tok_* handle is created by la_tok_from_file / la_tok_from_json and
 *     MUST be released with la_tok_free.
 *   - Any `char*`/`int32_t*` buffer returned by an FFI call is owned by Rust
 *     and MUST be freed with the matching la_tok_free_string / la_tok_free_ids.
 *   - All functions are nullptr/length tolerant and never unwind across the
 *     boundary (Rust uses catch_unwind internally).
 */

#ifndef LA_TOKENIZER_TOKENIZER_FFI_H
#define LA_TOKENIZER_TOKENIZER_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a loaded tokenizer. */
typedef struct la_tok_handle la_tok_handle;

/* Status codes returned by fallible calls. */
typedef enum la_tok_status {
    LA_TOK_OK = 0,
    LA_TOK_ERR_NULL_ARG = 1,
    LA_TOK_ERR_LOAD = 2,
    LA_TOK_ERR_ENCODE = 3,
    LA_TOK_ERR_DECODE = 4,
    LA_TOK_ERR_PANIC = 5,
    LA_TOK_ERR_UTF8 = 6
} la_tok_status;

/* A heap-allocated int32 buffer owned by Rust. Free with la_tok_free_ids. */
typedef struct la_tok_ids {
    int32_t* data;
    size_t len;
} la_tok_ids;

/* Load a tokenizer from a tokenizer.json file path (NUL-terminated UTF-8).
 * Returns NULL on failure; *out_status (if non-NULL) is set accordingly. */
la_tok_handle* la_tok_from_file(const char* path, la_tok_status* out_status);

/* Load a tokenizer from an in-memory tokenizer.json string of length `len`. */
la_tok_handle* la_tok_from_json(const char* json,
                                size_t len,
                                la_tok_status* out_status);

/* Free a handle. Safe to call with NULL. */
void la_tok_free(la_tok_handle* handle);

/* Encode UTF-8 `text` of byte length `len`.
 * add_special_tokens != 0 applies the tokenizer's special-token processor.
 * On success returns LA_TOK_OK and fills *out (caller frees with
 * la_tok_free_ids). On failure returns a non-zero status and out->data == NULL.
 */
la_tok_status la_tok_encode(const la_tok_handle* handle,
                            const char* text,
                            size_t len,
                            int add_special_tokens,
                            la_tok_ids* out);

/* Decode `len` ids. skip_special_tokens != 0 drops special tokens.
 * On success returns LA_TOK_OK and *out_str points to a NUL-terminated UTF-8
 * string owned by Rust (free with la_tok_free_string). */
la_tok_status la_tok_decode(const la_tok_handle* handle,
                            const int32_t* ids,
                            size_t len,
                            int skip_special_tokens,
                            char** out_str);

/* Add special tokens (array of `count` NUL-terminated UTF-8 strings).
 * Returns the number of newly-added tokens via *out_added (may be 0). */
la_tok_status la_tok_add_special_tokens(la_tok_handle* handle,
                                        const char* const* tokens,
                                        size_t count,
                                        int64_t* out_added);

/* Map a single token to its id, or -1 if unknown. Returns status. */
la_tok_status la_tok_token_to_id(const la_tok_handle* handle,
                                 const char* token,
                                 int32_t* out_id);

/* Vocabulary size (including added tokens). */
la_tok_status la_tok_vocab_size(const la_tok_handle* handle,
                                int64_t* out_size);

/* Free a string previously returned by the FFI. Safe with NULL. */
void la_tok_free_string(char* s);

/* Free an ids buffer previously returned by the FFI. Safe with NULL data. */
void la_tok_free_ids(la_tok_ids ids);

/* Returns a static, NUL-terminated version string of the FFI/crate. */
const char* la_tok_version(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* LA_TOKENIZER_TOKENIZER_FFI_H */
