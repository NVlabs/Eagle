// SPDX-License-Identifier: Apache-2.0
//
// la_tokenizer_ffi - C ABI binding to the HuggingFace `tokenizers` crate.
//
// Implements exactly the contract in
//   cpp_trt/src/tokenizer/include/la/tokenizer/tokenizer_ffi.h
//
// All boundary functions are panic-safe (catch_unwind) and nullptr-tolerant.
// Buffers returned to C are heap-allocated here and MUST be freed via the
// matching la_tok_free_* function.

use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

use tokenizers::tokenizer::Tokenizer;
use tokenizers::AddedToken;

// Keep these numeric values in sync with `la_tok_status` in tokenizer_ffi.h.
const LA_TOK_OK: i32 = 0;
const LA_TOK_ERR_NULL_ARG: i32 = 1;
const LA_TOK_ERR_LOAD: i32 = 2;
const LA_TOK_ERR_ENCODE: i32 = 3;
const LA_TOK_ERR_DECODE: i32 = 4;
const LA_TOK_ERR_PANIC: i32 = 5;
const LA_TOK_ERR_UTF8: i32 = 6;

/// Opaque handle exposed to C. Wraps a real Tokenizer.
pub struct la_tok_handle {
    tok: Tokenizer,
}

/// Mirror of `la_tok_ids` in the C header (must match field order/layout).
#[repr(C)]
pub struct la_tok_ids {
    data: *mut i32,
    len: usize,
}

#[inline]
fn write_status(out_status: *mut i32, st: i32) {
    if !out_status.is_null() {
        unsafe {
            *out_status = st;
        }
    }
}

/// Build a Tokenizer from a parser result, boxing it for C ownership.
fn make_handle(tok: Tokenizer) -> *mut la_tok_handle {
    Box::into_raw(Box::new(la_tok_handle { tok }))
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn la_tok_from_file(
    path: *const c_char,
    out_status: *mut i32,
) -> *mut la_tok_handle {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if path.is_null() {
            write_status(out_status, LA_TOK_ERR_NULL_ARG);
            return ptr::null_mut();
        }
        let c_path = unsafe { std::ffi::CStr::from_ptr(path) };
        let path_str = match c_path.to_str() {
            Ok(s) => s,
            Err(_) => {
                write_status(out_status, LA_TOK_ERR_UTF8);
                return ptr::null_mut();
            }
        };
        match Tokenizer::from_file(path_str) {
            Ok(tok) => {
                write_status(out_status, LA_TOK_OK);
                make_handle(tok)
            }
            Err(_) => {
                write_status(out_status, LA_TOK_ERR_LOAD);
                ptr::null_mut()
            }
        }
    }));
    match result {
        Ok(p) => p,
        Err(_) => {
            write_status(out_status, LA_TOK_ERR_PANIC);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn la_tok_from_json(
    json: *const c_char,
    len: usize,
    out_status: *mut i32,
) -> *mut la_tok_handle {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if json.is_null() {
            write_status(out_status, LA_TOK_ERR_NULL_ARG);
            return ptr::null_mut();
        }
        let bytes = unsafe { slice::from_raw_parts(json as *const u8, len) };
        let s = match std::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => {
                write_status(out_status, LA_TOK_ERR_UTF8);
                return ptr::null_mut();
            }
        };
        match s.parse::<Tokenizer>() {
            Ok(tok) => {
                write_status(out_status, LA_TOK_OK);
                make_handle(tok)
            }
            Err(_) => {
                write_status(out_status, LA_TOK_ERR_LOAD);
                ptr::null_mut()
            }
        }
    }));
    match result {
        Ok(p) => p,
        Err(_) => {
            write_status(out_status, LA_TOK_ERR_PANIC);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn la_tok_free(handle: *mut la_tok_handle) {
    if handle.is_null() {
        return;
    }
    // Reconstruct the Box and drop it. Wrapped in catch_unwind for safety.
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(handle));
    }));
}

// ---------------------------------------------------------------------------
// Encode / decode
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn la_tok_encode(
    handle: *const la_tok_handle,
    text: *const c_char,
    len: usize,
    add_special_tokens: c_int,
    out: *mut la_tok_ids,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() || out.is_null() {
            return LA_TOK_ERR_NULL_ARG;
        }
        // Initialize out to empty so callers can always free safely.
        unsafe {
            (*out).data = ptr::null_mut();
            (*out).len = 0;
        }
        if text.is_null() && len != 0 {
            return LA_TOK_ERR_NULL_ARG;
        }
        let bytes = if len == 0 {
            &[][..]
        } else {
            unsafe { slice::from_raw_parts(text as *const u8, len) }
        };
        let s = match std::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => return LA_TOK_ERR_UTF8,
        };
        let h = unsafe { &*handle };
        match h.tok.encode(s, add_special_tokens != 0) {
            Ok(enc) => {
                let ids: Vec<i32> =
                    enc.get_ids().iter().map(|&u| u as i32).collect();
                let mut boxed = ids.into_boxed_slice();
                let data = boxed.as_mut_ptr();
                let n = boxed.len();
                std::mem::forget(boxed);
                unsafe {
                    (*out).data = data;
                    (*out).len = n;
                }
                LA_TOK_OK
            }
            Err(_) => LA_TOK_ERR_ENCODE,
        }
    }));
    result.unwrap_or(LA_TOK_ERR_PANIC)
}

#[no_mangle]
pub extern "C" fn la_tok_decode(
    handle: *const la_tok_handle,
    ids: *const i32,
    len: usize,
    skip_special_tokens: c_int,
    out_str: *mut *mut c_char,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() || out_str.is_null() {
            return LA_TOK_ERR_NULL_ARG;
        }
        unsafe {
            *out_str = ptr::null_mut();
        }
        if ids.is_null() && len != 0 {
            return LA_TOK_ERR_NULL_ARG;
        }
        let id_slice = if len == 0 {
            &[][..]
        } else {
            unsafe { slice::from_raw_parts(ids, len) }
        };
        let u_ids: Vec<u32> = id_slice.iter().map(|&v| v as u32).collect();
        let h = unsafe { &*handle };
        match h.tok.decode(&u_ids, skip_special_tokens != 0) {
            Ok(text) => match std::ffi::CString::new(text) {
                Ok(c) => {
                    unsafe {
                        *out_str = c.into_raw();
                    }
                    LA_TOK_OK
                }
                Err(_) => LA_TOK_ERR_DECODE, // interior NUL
            },
            Err(_) => LA_TOK_ERR_DECODE,
        }
    }));
    result.unwrap_or(LA_TOK_ERR_PANIC)
}

// ---------------------------------------------------------------------------
// Special tokens / vocab
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn la_tok_add_special_tokens(
    handle: *mut la_tok_handle,
    tokens: *const *const c_char,
    count: usize,
    out_added: *mut i64,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            return LA_TOK_ERR_NULL_ARG;
        }
        if count != 0 && tokens.is_null() {
            return LA_TOK_ERR_NULL_ARG;
        }
        let mut added_tokens: Vec<AddedToken> = Vec::with_capacity(count);
        if count > 0 {
            let arr = unsafe { slice::from_raw_parts(tokens, count) };
            for &p in arr {
                if p.is_null() {
                    return LA_TOK_ERR_NULL_ARG;
                }
                let cs = unsafe { std::ffi::CStr::from_ptr(p) };
                let s = match cs.to_str() {
                    Ok(s) => s,
                    Err(_) => return LA_TOK_ERR_UTF8,
                };
                added_tokens.push(AddedToken::from(s.to_string(), true));
            }
        }
        let h = unsafe { &mut *handle };
        let n = h.tok.add_special_tokens(&added_tokens);
        if !out_added.is_null() {
            unsafe {
                *out_added = n as i64;
            }
        }
        LA_TOK_OK
    }));
    result.unwrap_or(LA_TOK_ERR_PANIC)
}

#[no_mangle]
pub extern "C" fn la_tok_token_to_id(
    handle: *const la_tok_handle,
    token: *const c_char,
    out_id: *mut i32,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() || token.is_null() || out_id.is_null() {
            return LA_TOK_ERR_NULL_ARG;
        }
        let cs = unsafe { std::ffi::CStr::from_ptr(token) };
        let s = match cs.to_str() {
            Ok(s) => s,
            Err(_) => return LA_TOK_ERR_UTF8,
        };
        let h = unsafe { &*handle };
        let id = match h.tok.token_to_id(s) {
            Some(v) => v as i32,
            None => -1,
        };
        unsafe {
            *out_id = id;
        }
        LA_TOK_OK
    }));
    result.unwrap_or(LA_TOK_ERR_PANIC)
}

#[no_mangle]
pub extern "C" fn la_tok_vocab_size(
    handle: *const la_tok_handle,
    out_size: *mut i64,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() || out_size.is_null() {
            return LA_TOK_ERR_NULL_ARG;
        }
        let h = unsafe { &*handle };
        let n = h.tok.get_vocab_size(true) as i64;
        unsafe {
            *out_size = n;
        }
        LA_TOK_OK
    }));
    result.unwrap_or(LA_TOK_ERR_PANIC)
}

// ---------------------------------------------------------------------------
// Buffer freeing / misc
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn la_tok_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(std::ffi::CString::from_raw(s));
    }));
}

#[no_mangle]
pub extern "C" fn la_tok_free_ids(ids: la_tok_ids) {
    if ids.data.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Vec::from_raw_parts(ids.data, ids.len, ids.len));
    }));
}

#[no_mangle]
pub extern "C" fn la_tok_version() -> *const c_char {
    // Static NUL-terminated string with a 'static lifetime.
    concat!("la_tokenizer_ffi ", env!("CARGO_PKG_VERSION"), "\0").as_ptr()
        as *const c_char
}
