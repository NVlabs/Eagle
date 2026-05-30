// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the pure-C++ chat-template assembler.
// These run WITHOUT any checkpoint / PyTorch / TRT.

#include "la/tokenizer/chat_template.hpp"

#include <gtest/gtest.h>

#include <string>

using la::tokenizer::ApplyOptions;
using la::tokenizer::ChatTemplateConfig;
using la::tokenizer::ImageGrid;
using la::tokenizer::Message;
using la::tokenizer::Role;

namespace {

std::string repeat(const std::string& s, std::int64_t n) {
    std::string out;
    for (std::int64_t i = 0; i < n; ++i) out += s;
    return out;
}

}  // namespace

TEST(ImageGrid, ContextTokenCountIsQuarterOfGrid) {
    EXPECT_EQ(ImageGrid(2, 2).num_context_tokens(), 1);    // 4/4
    EXPECT_EQ(ImageGrid(4, 4).num_context_tokens(), 4);    // 16/4
    EXPECT_EQ(ImageGrid(32, 32).num_context_tokens(), 256);// 1024/4
    EXPECT_EQ(ImageGrid(6, 10).num_context_tokens(), 15);  // 60/4
}

TEST(ImageGrid, NonPositiveDimsGiveZero) {
    EXPECT_EQ(ImageGrid(0, 4).num_context_tokens(), 0);
    EXPECT_EQ(ImageGrid(4, 0).num_context_tokens(), 0);
    EXPECT_EQ(ImageGrid(-2, 4).num_context_tokens(), 0);
}

TEST(ImageBlock, ExactExpansion) {
    ChatTemplateConfig cfg;
    const std::string blk =
        la::tokenizer::build_image_block(ImageGrid(2, 2), cfg);
    EXPECT_EQ(blk, "<img><IMG_CONTEXT></img>");

    const std::string blk2 =
        la::tokenizer::build_image_block(ImageGrid(4, 4), cfg);
    EXPECT_EQ(blk2, "<img>" + repeat("<IMG_CONTEXT>", 4) + "</img>");
}

TEST(ImageBlock, ZeroContextStillHasTags) {
    ChatTemplateConfig cfg;
    EXPECT_EQ(la::tokenizer::build_image_block(ImageGrid(0, 0), cfg),
              "<img></img>");
}

TEST(ExpandPlaceholders, SingleMarker) {
    ChatTemplateConfig cfg;
    std::vector<ImageGrid> imgs{ImageGrid(4, 4)};
    const std::string out = la::tokenizer::expand_image_placeholders(
        "Look at <image-1> now.", imgs, cfg);
    EXPECT_EQ(out,
              "Look at <img>" + repeat("<IMG_CONTEXT>", 4) + "</img> now.");
}

TEST(ExpandPlaceholders, MultipleMarkersMapByIndex) {
    ChatTemplateConfig cfg;
    std::vector<ImageGrid> imgs{ImageGrid(2, 2), ImageGrid(4, 4)};
    const std::string out = la::tokenizer::expand_image_placeholders(
        "<image-2> vs <image-1>", imgs, cfg);
    const std::string b1 = "<img><IMG_CONTEXT></img>";                  // 2x2
    const std::string b2 = "<img>" + repeat("<IMG_CONTEXT>", 4) + "</img>";  // 4x4
    EXPECT_EQ(out, b2 + " vs " + b1);
}

TEST(ExpandPlaceholders, OutOfRangeLeftVerbatim) {
    ChatTemplateConfig cfg;
    std::vector<ImageGrid> imgs{ImageGrid(2, 2)};
    const std::string out = la::tokenizer::expand_image_placeholders(
        "a <image-3> b", imgs, cfg);
    EXPECT_EQ(out, "a <image-3> b");
}

TEST(ExpandPlaceholders, NonMarkerAnglesUntouched) {
    ChatTemplateConfig cfg;
    std::vector<ImageGrid> imgs{ImageGrid(2, 2)};
    EXPECT_EQ(la::tokenizer::expand_image_placeholders("1 < 2 > 0", imgs, cfg),
              "1 < 2 > 0");
    EXPECT_EQ(
        la::tokenizer::expand_image_placeholders("<image->", imgs, cfg),
        "<image->");  // no digits
}

TEST(ApplyChatTemplate, BasicUserWithDefaultSystemAndGenPrompt) {
    ChatTemplateConfig cfg;
    ApplyOptions opts;  // defaults: inject system + gen prompt
    std::vector<Message> msgs{Message(Role::User, "Hello")};
    const std::string out = la::tokenizer::apply_chat_template(msgs, cfg, opts);
    // Reference: injected default-system has a newline after the content
    // (before <|im_end|>), matching py_apply_chat_template.
    const std::string expected =
        "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(out, expected);
}

TEST(ApplyChatTemplate, ExplicitSystemNotDuplicated) {
    ChatTemplateConfig cfg;
    ApplyOptions opts;
    std::vector<Message> msgs{Message(Role::System, "Custom sys."),
                              Message(Role::User, "Hi")};
    const std::string out = la::tokenizer::apply_chat_template(msgs, cfg, opts);
    const std::string expected =
        "<|im_start|>system\nCustom sys.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(out, expected);
}

TEST(ApplyChatTemplate, NoSystemInjectionWhenDisabled) {
    ChatTemplateConfig cfg;
    ApplyOptions opts;
    opts.inject_default_system = false;
    std::vector<Message> msgs{Message(Role::User, "Hi")};
    const std::string out = la::tokenizer::apply_chat_template(msgs, cfg, opts);
    const std::string expected =
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(out, expected);
}

TEST(ApplyChatTemplate, NoGenerationPromptWhenDisabled) {
    ChatTemplateConfig cfg;
    ApplyOptions opts;
    opts.inject_default_system = false;
    opts.add_generation_prompt = false;
    std::vector<Message> msgs{Message(Role::User, "Hi"),
                              Message(Role::Assistant, "There")};
    const std::string out = la::tokenizer::apply_chat_template(msgs, cfg, opts);
    const std::string expected =
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\nThere<|im_end|>\n";
    EXPECT_EQ(out, expected);
}

TEST(ApplyChatTemplate, ImageMarkerExpandedInTurn) {
    ChatTemplateConfig cfg;
    ApplyOptions opts;
    opts.inject_default_system = false;
    Message m(Role::User, "Describe <image-1>");
    m.images.push_back(ImageGrid(2, 2));  // -> 1 IMG_CONTEXT
    std::vector<Message> msgs{m};
    const std::string out = la::tokenizer::apply_chat_template(msgs, cfg, opts);
    const std::string expected =
        "<|im_start|>user\nDescribe <img><IMG_CONTEXT></img><|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(out, expected);
}

TEST(ApplyChatTemplate, BareImagePrependedWhenNoMarker) {
    ChatTemplateConfig cfg;
    ApplyOptions opts;
    opts.inject_default_system = false;
    Message m(Role::User, "What is this?");
    m.images.push_back(ImageGrid(4, 4));  // -> 4 IMG_CONTEXT
    std::vector<Message> msgs{m};
    const std::string out = la::tokenizer::apply_chat_template(msgs, cfg, opts);
    const std::string expected =
        "<|im_start|>user\n<img>" + repeat("<IMG_CONTEXT>", 4) +
        "</img>\nWhat is this?<|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(out, expected);
}

TEST(ApplyChatTemplate, OverridableSpecialTokens) {
    ChatTemplateConfig cfg;
    cfg.im_start = "[S]";
    cfg.im_end = "[E]";
    cfg.img_context = "<CTX>";
    ApplyOptions opts;
    opts.inject_default_system = false;
    Message m(Role::User, "<image-1>");
    m.images.push_back(ImageGrid(2, 2));
    std::vector<Message> msgs{m};
    const std::string out = la::tokenizer::apply_chat_template(msgs, cfg, opts);
    const std::string expected =
        "[S]user\n<img><CTX></img>[E]\n[S]assistant\n";
    EXPECT_EQ(out, expected);
}
