// SPDX-License-Identifier: Apache-2.0
//
// chat_template.cpp - pure C++ implementation of the LocateAnything / Qwen2
// chat-template assembler. No external dependencies.
//
// Ported from Embodied/eaglevl/utils/locany/processing_locateanything.py
//   (py_apply_chat_template + <image-k> expansion). Spec Section 6.

#include "la/tokenizer/chat_template.hpp"

#include <stdexcept>

namespace la {
namespace tokenizer {

const std::string& role_name(Role role, const ChatTemplateConfig& cfg) {
    switch (role) {
        case Role::System:
            return cfg.system_name;
        case Role::User:
            return cfg.user_name;
        case Role::Assistant:
            return cfg.assistant_name;
    }
    // Unreachable for valid enum values; keep a stable fallback.
    return cfg.user_name;
}

std::string build_image_block(const ImageGrid& image,
                              const ChatTemplateConfig& cfg) {
    const std::int64_t n = image.num_context_tokens();
    std::string out;
    // Reserve: tags + n copies of the context token.
    out.reserve(cfg.img_start.size() + cfg.img_end.size() +
                static_cast<std::size_t>(n > 0 ? n : 0) * cfg.img_context.size());
    out += cfg.img_start;
    for (std::int64_t i = 0; i < n; ++i) {
        out += cfg.img_context;
    }
    out += cfg.img_end;
    return out;
}

namespace {

// Parse a "<image-K>" marker starting at `content[pos]` where content[pos]=='<'.
// On success returns true, sets `index` to the parsed 1-based K and `end` to
// one past the closing '>'. Only matches the exact literal form "<image-" +
// one-or-more ASCII digits + ">".
bool match_image_marker(const std::string& content, std::size_t pos,
                        std::size_t& index, std::size_t& end) {
    static const std::string kPrefix = "<image-";
    if (content.compare(pos, kPrefix.size(), kPrefix) != 0) {
        return false;
    }
    std::size_t p = pos + kPrefix.size();
    std::size_t digits_start = p;
    std::int64_t value = 0;
    while (p < content.size() && content[p] >= '0' && content[p] <= '9') {
        value = value * 10 + (content[p] - '0');
        ++p;
    }
    if (p == digits_start) {
        return false;  // no digits
    }
    if (p >= content.size() || content[p] != '>') {
        return false;  // not closed
    }
    index = static_cast<std::size_t>(value);
    end = p + 1;  // past '>'
    return true;
}

}  // namespace

std::string expand_image_placeholders(const std::string& content,
                                      const std::vector<ImageGrid>& images,
                                      const ChatTemplateConfig& cfg) {
    std::string out;
    out.reserve(content.size());
    std::size_t i = 0;
    while (i < content.size()) {
        if (content[i] == '<') {
            std::size_t index = 0;
            std::size_t end = 0;
            if (match_image_marker(content, i, index, end)) {
                if (index >= 1 && index <= images.size()) {
                    out += build_image_block(images[index - 1], cfg);
                    i = end;
                    continue;
                }
                // Out-of-range marker: leave it verbatim so callers notice.
            }
        }
        out += content[i];
        ++i;
    }
    return out;
}

namespace {

// Returns true if `content` contains at least one valid "<image-K>" marker.
bool has_image_marker(const std::string& content) {
    for (std::size_t i = 0; i + 1 < content.size(); ++i) {
        if (content[i] == '<') {
            std::size_t index = 0;
            std::size_t end = 0;
            if (match_image_marker(content, i, index, end)) {
                return true;
            }
        }
    }
    return false;
}

// Render the content for one message: expand explicit markers, or, when there
// are images but no markers, prepend each image block on its own line.
std::string render_content(const Message& msg, const ChatTemplateConfig& cfg) {
    if (msg.images.empty()) {
        return msg.content;
    }
    if (has_image_marker(msg.content)) {
        return expand_image_placeholders(msg.content, msg.images, cfg);
    }
    // No markers: lead with image blocks, each followed by a newline, then the
    // textual content. Mirrors the reference where bare images precede the text.
    std::string out;
    for (const auto& img : msg.images) {
        out += build_image_block(img, cfg);
        out += cfg.newline;
    }
    out += msg.content;
    return out;
}

void append_turn(std::string& out, const std::string& role,
                 const std::string& body, const ChatTemplateConfig& cfg) {
    out += cfg.im_start;
    out += role;
    out += cfg.newline;
    out += body;
    out += cfg.im_end;
    out += cfg.newline;
}

}  // namespace

std::string apply_chat_template(const std::vector<Message>& messages,
                                const ChatTemplateConfig& cfg,
                                const ApplyOptions& opts) {
    std::string out;

    const bool has_leading_system =
        !messages.empty() && messages.front().role == Role::System;

    // System injection: only when requested and the caller did not supply one.
    //
    // NOTE: Byte-exact with the reference py_apply_chat_template, the INJECTED
    // default-system block places a newline AFTER the content and BEFORE
    // <|im_end|>:
    //     <|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n
    // whereas a normal turn is:
    //     <|im_start|>{role}\n{content}<|im_end|>\n   (no extra newline)
    if (opts.inject_default_system && !has_leading_system &&
        !cfg.default_system_prompt.empty()) {
        out += cfg.im_start;
        out += cfg.system_name;
        out += cfg.newline;
        out += cfg.default_system_prompt;
        out += cfg.newline;  // reference emits content + "\n" before <|im_end|>
        out += cfg.im_end;
        out += cfg.newline;
    }

    for (const auto& msg : messages) {
        append_turn(out, role_name(msg.role, cfg), render_content(msg, cfg),
                    cfg);
    }

    if (opts.add_generation_prompt) {
        // Open the assistant turn but do NOT close it; the model generates the
        // body and the terminating <|im_end|>.
        out += cfg.im_start;
        out += cfg.assistant_name;
        out += cfg.newline;
    }

    return out;
}

}  // namespace tokenizer
}  // namespace la
