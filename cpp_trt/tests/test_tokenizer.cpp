// Unit tests for la_tokenizer chat-template assembler.
//
// Expected prompt strings are built INDEPENDENTLY in the test by concatenating
// the documented framing pieces, so the im_start/im_end framing, the injected
// default-system trailing newline, the generation prompt, and the
// <IMG_CONTEXT>-repeated-(h*w)/4 image block are all verified exactly.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "la/tokenizer/chat_template.hpp"

namespace ct = la::tokenizer;

namespace {

// Independent image block: <img> + N*<IMG_CONTEXT> + </img>, N = (h*w)/4.
std::string RefImageBlock(int64_t h, int64_t w, const ct::ChatTemplateConfig& c) {
  std::string s = c.img_start;
  int64_t n = (h <= 0 || w <= 0) ? 0 : (h * w) / 4;
  for (int64_t i = 0; i < n; ++i) s += c.img_context;
  s += c.img_end;
  return s;
}

}  // namespace

TEST(ImageGrid, NumContextTokensIsHWOver4) {
  EXPECT_EQ(ct::ImageGrid(2, 2).num_context_tokens(), 1);
  EXPECT_EQ(ct::ImageGrid(4, 6).num_context_tokens(), 6);
  EXPECT_EQ(ct::ImageGrid(10, 10).num_context_tokens(), 25);
  EXPECT_EQ(ct::ImageGrid(0, 5).num_context_tokens(), 0);
  EXPECT_EQ(ct::ImageGrid(5, -1).num_context_tokens(), 0);
}

TEST(BuildImageBlock, RepeatsImgContext) {
  ct::ChatTemplateConfig cfg;
  ct::ImageGrid g(4, 6);  // (4*6)/4 = 6 context tokens
  std::string got = ct::build_image_block(g, cfg);
  std::string want = RefImageBlock(4, 6, cfg);
  EXPECT_EQ(got, want);
  // sanity: it literally contains 6 copies of <IMG_CONTEXT>.
  size_t count = 0, pos = 0;
  while ((pos = want.find(cfg.img_context, pos)) != std::string::npos) {
    ++count;
    pos += cfg.img_context.size();
  }
  EXPECT_EQ(count, 6u);
}

// Single user turn, no images, with the injected default system block.
TEST(ApplyChatTemplate, UserTurnWithInjectedSystem) {
  ct::ChatTemplateConfig cfg;
  ct::ApplyOptions opts;  // add_generation_prompt=true, inject_default_system=true
  std::vector<ct::Message> msgs = {
      ct::Message(ct::Role::User, "Hello")};
  std::string got = ct::apply_chat_template(msgs, cfg, opts);

  // Independent expected assembly:
  // injected system: <|im_start|>system\n{prompt}\n<|im_end|>\n  (extra \n)
  // user turn:       <|im_start|>user\nHello<|im_end|>\n
  // gen prompt:      <|im_start|>assistant\n
  std::string want;
  want += cfg.im_start + cfg.system_name + cfg.newline +
          cfg.default_system_prompt + cfg.newline + cfg.im_end + cfg.newline;
  want += cfg.im_start + cfg.user_name + cfg.newline + "Hello" + cfg.im_end +
          cfg.newline;
  want += cfg.im_start + cfg.assistant_name + cfg.newline;
  EXPECT_EQ(got, want);
}

// Explicit system message rendered as a NORMAL turn (no extra newline),
// and no injected system because first message is System.
TEST(ApplyChatTemplate, ExplicitSystemIsNormalTurn) {
  ct::ChatTemplateConfig cfg;
  ct::ApplyOptions opts;
  std::vector<ct::Message> msgs = {
      ct::Message(ct::Role::System, "Sys"),
      ct::Message(ct::Role::User, "Hi")};
  std::string got = ct::apply_chat_template(msgs, cfg, opts);

  std::string want;
  // normal system turn: NO extra newline after content
  want += cfg.im_start + cfg.system_name + cfg.newline + "Sys" + cfg.im_end +
          cfg.newline;
  want += cfg.im_start + cfg.user_name + cfg.newline + "Hi" + cfg.im_end +
          cfg.newline;
  want += cfg.im_start + cfg.assistant_name + cfg.newline;
  EXPECT_EQ(got, want);
}

// No generation prompt appended when disabled.
TEST(ApplyChatTemplate, NoGenerationPrompt) {
  ct::ChatTemplateConfig cfg;
  ct::ApplyOptions opts;
  opts.add_generation_prompt = false;
  opts.inject_default_system = false;
  std::vector<ct::Message> msgs = {ct::Message(ct::Role::User, "X")};
  std::string got = ct::apply_chat_template(msgs, cfg, opts);

  std::string want = cfg.im_start + cfg.user_name + cfg.newline + "X" +
                     cfg.im_end + cfg.newline;
  EXPECT_EQ(got, want);
  // must NOT end with an assistant header
  EXPECT_EQ(got.find(cfg.assistant_name), std::string::npos);
}

// Image present, no markers in content: image block prepended on its own line,
// then content. <IMG_CONTEXT> repeated (h*w)/4 times.
TEST(ApplyChatTemplate, ImagePrependedWithContextRepeat) {
  ct::ChatTemplateConfig cfg;
  ct::ApplyOptions opts;
  opts.inject_default_system = false;
  opts.add_generation_prompt = false;

  ct::ImageGrid g(4, 6);  // 6 context tokens
  ct::Message m(ct::Role::User, "describe", {g});
  std::vector<ct::Message> msgs = {m};
  std::string got = ct::apply_chat_template(msgs, cfg, opts);

  // body = image_block + newline + content
  std::string body = RefImageBlock(4, 6, cfg) + cfg.newline + "describe";
  std::string want =
      cfg.im_start + cfg.user_name + cfg.newline + body + cfg.im_end + cfg.newline;
  EXPECT_EQ(got, want);

  // verify exactly 6 <IMG_CONTEXT> in the full string.
  size_t count = 0, pos = 0;
  while ((pos = got.find(cfg.img_context, pos)) != std::string::npos) {
    ++count;
    pos += cfg.img_context.size();
  }
  EXPECT_EQ(count, 6u);
}

// expand_image_placeholders: <image-1> -> image block; out-of-range left verbatim.
TEST(ExpandPlaceholders, ReplacesAndLeavesVerbatim) {
  ct::ChatTemplateConfig cfg;
  std::vector<ct::ImageGrid> imgs = {ct::ImageGrid(2, 2)};  // 1 context token
  std::string in = "before <image-1> after <image-2> end";
  std::string got = ct::expand_image_placeholders(in, imgs, cfg);

  std::string blk = RefImageBlock(2, 2, cfg);
  // <image-2> is out of range (only 1 image) -> left verbatim.
  std::string want = "before " + blk + " after <image-2> end";
  EXPECT_EQ(got, want);
}
