// SPDX-License-Identifier: Apache-2.0
//
// la/tokenizer/chat_template.hpp
//
// Pure-C++ (no external dependencies) chat-template assembler for the
// LocateAnything / Qwen2-style prompt format.
//
// This component is fully testable WITHOUT a model checkpoint, PyTorch, or
// TensorRT. It reproduces the string assembly performed by the reference
// Python `py_apply_chat_template` plus the InternVL-style image placeholder
// expansion:
//
//     <image-k>  ->  <img> + (<IMG_CONTEXT> * ((h*w)//4)) + </img>
//
// Spec Section 6.
//
// Ported (string layout / semantics) from:
//   Embodied/eaglevl/utils/locany/processing_locateanything.py
//     - py_apply_chat_template
//     - the <image-k> -> <img><IMG_CONTEXT>*((h*w)//4)</img> expansion
//   Embodied/eaglevl/model/locany/tokenization_qwen2.py (special tokens)
//
// NOTE: The exact special-token spellings below match the Qwen2 chat format
// (<|im_start|>, <|im_end|>) and the InternVL image markers (<img>, </img>,
// <IMG_CONTEXT>). They are exposed as configurable fields on ChatTemplateConfig
// so a downstream caller that has the real tokenizer_config.json can override
// them to guarantee byte-exact parity once a checkpoint is available.
//
// Reference fidelity (verified against processing_locateanything.py):
//   * py_apply_chat_template emits, per turn, "<|im_start|>{role}\n{content}
//     <|im_end|>\n". For an INJECTED default system turn it instead emits
//     "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n" (note
//     the extra "\n" after the content). apply_chat_template() reproduces both.
//   * py_apply_chat_template itself emits bare "<image-N>" placeholder tokens;
//     the actual "<img>...<IMG_CONTEXT>*((h*w)//merge)...</img>" expansion is
//     performed later in the processor (replace_media_placeholder). This module
//     fuses the two: expand_image_placeholders()/build_image_block() perform
//     the processor-side expansion, with (h*w)//4 matching the default
//     merge_kernel_size [2,2]. The reference processor additionally prepends a
//     human-readable "<image N>" label to each block; that label is processor
//     context-dependent and is intentionally NOT emitted here (the assembler
//     stays faithful to the token-level expansion). Override img_start/img_end/
//     img_context via ChatTemplateConfig for a specific checkpoint.

#ifndef LA_TOKENIZER_CHAT_TEMPLATE_HPP
#define LA_TOKENIZER_CHAT_TEMPLATE_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace la {
namespace tokenizer {

/// A single image attached to a turn. The number of <IMG_CONTEXT> tokens
/// emitted for this image is (grid_h * grid_w) / 4 (integer division), matching
/// the reference 2x2 pixel-unshuffle / merge that quarters the token count.
struct ImageGrid {
    /// Tiled feature-grid height (number of patch rows feeding the LLM).
    std::int64_t grid_h = 0;
    /// Tiled feature-grid width (number of patch columns feeding the LLM).
    std::int64_t grid_w = 0;

    ImageGrid() = default;
    ImageGrid(std::int64_t h, std::int64_t w) : grid_h(h), grid_w(w) {}

    /// Number of <IMG_CONTEXT> tokens this image expands to: (h*w)//4.
    /// Returns 0 if either dimension is non-positive.
    std::int64_t num_context_tokens() const {
        if (grid_h <= 0 || grid_w <= 0) {
            return 0;
        }
        return (grid_h * grid_w) / 4;
    }
};

/// Conversation roles. Mapped to the literal strings used in the <|im_start|>
/// header line ("system", "user", "assistant").
enum class Role {
    System,
    User,
    Assistant,
};

/// One conversation turn.
///
/// `content` is the textual content for the turn. It may contain image
/// placeholder markers of the form "<image-1>", "<image-2>", ... which are
/// expanded against `images` (placeholder "<image-k>" maps to images[k-1]).
///
/// If `content` contains NO "<image-k>" markers but `images` is non-empty, the
/// expanded image blocks are PREPENDED to the content (each on its own line),
/// matching the reference behavior where bare images lead the user turn.
struct Message {
    Role role = Role::User;
    std::string content;
    std::vector<ImageGrid> images;

    Message() = default;
    Message(Role r, std::string c) : role(r), content(std::move(c)) {}
    Message(Role r, std::string c, std::vector<ImageGrid> imgs)
        : role(r), content(std::move(c)), images(std::move(imgs)) {}
};

/// Configurable special tokens / framing strings. Defaults match the Qwen2 +
/// InternVL conventions used by LocateAnything. Override to match a specific
/// tokenizer_config.json for byte-exact parity.
struct ChatTemplateConfig {
    std::string im_start = "<|im_start|>";
    std::string im_end = "<|im_end|>";

    std::string img_start = "<img>";
    std::string img_end = "</img>";
    std::string img_context = "<IMG_CONTEXT>";

    /// Role name strings as they appear after <|im_start|>.
    std::string system_name = "system";
    std::string user_name = "user";
    std::string assistant_name = "assistant";

    /// Default system prompt injected when `add_system` is requested and no
    /// explicit System message is present in the conversation.
    std::string default_system_prompt = "You are a helpful assistant.";

    /// Line separator used inside the template. Qwen2 uses "\n".
    std::string newline = "\n";
};

/// Options controlling assembly.
struct ApplyOptions {
    /// If true, append a trailing generation prompt header
    /// "<|im_start|>assistant\n" so the model continues as the assistant.
    bool add_generation_prompt = true;

    /// If true and the conversation has no leading System message, inject the
    /// configured default_system_prompt as the first turn.
    bool inject_default_system = true;
};

/// Expand a single content string's "<image-k>" markers using `images`.
///
/// "<image-1>" -> "<img>" + N*"<IMG_CONTEXT>" + "</img>" where
/// N = images[0].num_context_tokens(), and so on for k=2,3,...
///
/// Markers referencing an out-of-range index are left untouched (so callers can
/// detect mistakes) ; this never throws.
std::string expand_image_placeholders(const std::string& content,
                                      const std::vector<ImageGrid>& images,
                                      const ChatTemplateConfig& cfg);

/// Build the full <img>...</img> block for one image (no surrounding newlines).
std::string build_image_block(const ImageGrid& image,
                              const ChatTemplateConfig& cfg);

/// Apply the chat template to a list of messages, returning the exact prompt
/// string. Pure function, no I/O.
std::string apply_chat_template(const std::vector<Message>& messages,
                                const ChatTemplateConfig& cfg = {},
                                const ApplyOptions& opts = {});

/// Convenience: map a Role enum to its configured name string.
const std::string& role_name(Role role, const ChatTemplateConfig& cfg);

}  // namespace tokenizer
}  // namespace la

#endif  // LA_TOKENIZER_CHAT_TEMPLATE_HPP
