// SPDX-License-Identifier: Apache-2.0
//
// la/config/token_ids.cpp
//
// Implementation of TokenIds, a tiny std-only JSON object parser, and the
// LoadTokenIds loader. No external dependencies (no nlohmann, no torch, no TRT)
// so this builds with a C++17 toolchain alone.
//
// The JSON parser here is intentionally minimal: it recognizes a *flat* object
// of "string": <value> pairs where <value> may be a number, string, boolean,
// null, a nested object, or an array. Only the (key -> integer) pairs are
// retained; nested objects are recursed into (one canonical nesting level is
// honored by name in LoadTokenIds), arrays/strings/etc. are skipped. This is
// sufficient to lift token ids out of a HuggingFace-style config.json without
// pulling in a full JSON library.

#include "la/config/token_ids.hpp"

#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace la::config {

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON scanner.
//
// Produces a flat map of string-key -> integer-value for the top level, and a
// map of string-key -> raw-text for nested object values (so the caller can
// recurse into a chosen nested object by name). This is NOT a general JSON DOM;
// it is just enough to read token ids.
// ---------------------------------------------------------------------------

struct ParseError : std::runtime_error {
  explicit ParseError(const std::string& m) : std::runtime_error(m) {}
};

class JsonScanner {
 public:
  explicit JsonScanner(std::string_view text) : s_(text) {}

  // Parse a single JSON object at the current position. Fills `ints` with
  // every "key": <integer> pair found at this object's top level, and
  // `objects` with every "key": <nested-object> pair (raw text incl. braces).
  // Throws ParseError on malformed input.
  void ParseObject(std::map<std::string, token_id_t>& ints,
                   std::map<std::string, std::string>& objects) {
    SkipWs();
    Expect('{');
    SkipWs();
    if (Peek() == '}') {
      ++pos_;
      return;
    }
    for (;;) {
      SkipWs();
      std::string key = ParseString();
      SkipWs();
      Expect(':');
      SkipWs();
      char c = Peek();
      if (c == '{') {
        std::size_t start = pos_;
        SkipValue();  // skips the whole nested object
        objects.emplace(key, std::string(s_.substr(start, pos_ - start)));
      } else if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
        bool is_int = false;
        token_id_t v = ParseNumber(is_int);
        if (is_int) {
          ints[key] = v;
        }
        // Non-integer numbers are simply not token ids; ignore.
      } else {
        SkipValue();  // string, bool, null, array -> ignore
      }
      SkipWs();
      char d = Next();
      if (d == ',') {
        continue;
      }
      if (d == '}') {
        break;
      }
      throw ParseError("expected ',' or '}' in object");
    }
  }

 private:
  char Peek() const {
    if (pos_ >= s_.size()) throw ParseError("unexpected end of input");
    return s_[pos_];
  }
  char Next() {
    char c = Peek();
    ++pos_;
    return c;
  }
  void Expect(char c) {
    char g = Next();
    if (g != c) {
      throw ParseError(std::string("expected '") + c + "'");
    }
  }
  void SkipWs() {
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else {
        break;
      }
    }
  }

  std::string ParseString() {
    Expect('"');
    std::string out;
    for (;;) {
      if (pos_ >= s_.size()) throw ParseError("unterminated string");
      char c = s_[pos_++];
      if (c == '"') break;
      if (c == '\\') {
        if (pos_ >= s_.size()) throw ParseError("bad escape");
        char e = s_[pos_++];
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            // Decode a \uXXXX escape into UTF-8. We only need this so that
            // token spellings like "<box>" round-trip; surrogate
            // pairs are handled.
            unsigned cp = ParseHex4();
            if (cp >= 0xD800 && cp <= 0xDBFF) {
              if (pos_ + 1 < s_.size() && s_[pos_] == '\\' &&
                  s_[pos_ + 1] == 'u') {
                pos_ += 2;
                unsigned lo = ParseHex4();
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              }
            }
            AppendUtf8(out, cp);
            break;
          }
          default:
            throw ParseError("invalid escape");
        }
      } else {
        out.push_back(c);
      }
    }
    return out;
  }

  unsigned ParseHex4() {
    if (pos_ + 4 > s_.size()) throw ParseError("bad \\u escape");
    unsigned v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = s_[pos_++];
      v <<= 4;
      if (c >= '0' && c <= '9') {
        v |= static_cast<unsigned>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        v |= static_cast<unsigned>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        v |= static_cast<unsigned>(c - 'A' + 10);
      } else {
        throw ParseError("bad hex digit in \\u escape");
      }
    }
    return v;
  }

  static void AppendUtf8(std::string& out, unsigned cp) {
    if (cp <= 0x7F) {
      out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  // Parse a JSON number. Sets `is_int` true iff it had no '.', 'e', or 'E'.
  token_id_t ParseNumber(bool& is_int) {
    std::size_t start = pos_;
    if (Peek() == '-') ++pos_;
    is_int = true;
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if (std::isdigit(static_cast<unsigned char>(c))) {
        ++pos_;
      } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
        is_int = false;
        ++pos_;
      } else {
        break;
      }
    }
    std::string num(s_.substr(start, pos_ - start));
    if (is_int) {
      try {
        return static_cast<token_id_t>(std::stoll(num));
      } catch (...) {
        is_int = false;  // out of range -> treat as non-int, ignore
        return 0;
      }
    }
    return 0;
  }

  // Skip an arbitrary JSON value (object, array, string, number, literal).
  void SkipValue() {
    SkipWs();
    char c = Peek();
    if (c == '{' || c == '[') {
      char open = c;
      char close = (c == '{') ? '}' : ']';
      int depth = 0;
      for (;;) {
        char ch = Next();
        if (ch == '"') {
          --pos_;
          ParseString();  // strings may contain braces/brackets
          continue;
        }
        if (ch == open) {
          ++depth;
        } else if (ch == close) {
          --depth;
          if (depth == 0) break;
        }
      }
    } else if (c == '"') {
      ParseString();
    } else {
      // number / true / false / null
      while (pos_ < s_.size()) {
        char ch = s_[pos_];
        if (ch == ',' || ch == '}' || ch == ']' || ch == ' ' || ch == '\t' ||
            ch == '\n' || ch == '\r') {
          break;
        }
        ++pos_;
      }
    }
  }

  std::string_view s_;
  std::size_t pos_ = 0;
};

// ---------------------------------------------------------------------------
// Field -> accepted JSON key spellings.
//
// The released checkpoint config may spell these differently (canonical name,
// HF special-token literal, etc.). We accept any of the listed spellings and
// bind the first one found. Order within each list = priority (earlier wins).
// ---------------------------------------------------------------------------

struct FieldBinding {
  token_id_t TokenIds::* member;
  std::vector<std::string> keys;
};

const std::vector<FieldBinding>& Bindings() {
  static const std::vector<FieldBinding> b = {
      {&TokenIds::image_token,
       {"image_token_index", "image_token", "image_token_id", "<image>"}},
      {&TokenIds::box_start,
       {"box_start", "box_start_token_id", "bbox_start", "<box>"}},
      {&TokenIds::box_end,
       {"box_end", "box_end_token_id", "bbox_end", "</box>"}},
      {&TokenIds::ref_start, {"ref_start", "ref_start_token_id", "<ref>"}},
      {&TokenIds::ref_end, {"ref_end", "ref_end_token_id", "</ref>"}},
      {&TokenIds::mask, {"mask", "mask_token_id", "seg_token_id", "<mask>"}},
      {&TokenIds::coord_start,
       {"coord_start", "coord_start_token_id", "<coord>"}},
      {&TokenIds::coord_end, {"coord_end", "coord_end_token_id", "</coord>"}},
      {&TokenIds::none, {"none", "none_token_id", "<none>"}},
      {&TokenIds::null_token, {"null", "null_token_id", "<null>"}},
      {&TokenIds::im_end,
       {"im_end", "eos", "eos_token_id", "im_end_token_id", "<|im_end|>"}},
      {&TokenIds::switch_token,
       {"switch", "switch_token", "switch_token_id", "<switch>"}},
  };
  return b;
}

void BindFromMap(TokenIds& t, const std::map<std::string, token_id_t>& ints) {
  for (const auto& fb : Bindings()) {
    if (t.*(fb.member) != kUnsetTokenId) continue;  // already bound
    for (const auto& key : fb.keys) {
      auto it = ints.find(key);
      if (it != ints.end()) {
        t.*(fb.member) = it->second;
        break;
      }
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// TokenIds members
// ---------------------------------------------------------------------------

bool TokenIds::IsComplete() const noexcept {
  for (const auto& fb : Bindings()) {
    if (this->*(fb.member) == kUnsetTokenId) return false;
  }
  return true;
}

void TokenIds::Validate(bool require_all) const {
  if (coord_start == kUnsetTokenId || coord_end == kUnsetTokenId) {
    throw std::runtime_error(
        "TokenIds::Validate: coord_start/coord_end must be set");
  }
  if (coord_start < 0) {
    throw std::runtime_error(
        "TokenIds::Validate: coord_start must be non-negative");
  }
  // Core invariant: the coordinate band is contiguous and spans exactly 1001
  // ids (1000 intervals). This is what makes CoordBin/CoordTokenForBin valid.
  if (coord_end - coord_start != kCoordBandSpan) {
    std::ostringstream os;
    os << "TokenIds::Validate: coord band not contiguous/size-1001: "
       << "coord_end(" << coord_end << ") - coord_start(" << coord_start
       << ") = " << (coord_end - coord_start) << ", expected "
       << kCoordBandSpan;
    throw std::runtime_error(os.str());
  }

  // Sanity: span delimiters must be ordered when both present.
  auto check_order = [](token_id_t s, token_id_t e, const char* what) {
    if (s != kUnsetTokenId && e != kUnsetTokenId && !(s < e)) {
      throw std::runtime_error(std::string("TokenIds::Validate: ") + what +
                               " start id must be < end id");
    }
  };
  check_order(box_start, box_end, "box");
  check_order(ref_start, ref_end, "ref");

  if (require_all && !IsComplete()) {
    throw std::runtime_error(
        "TokenIds::Validate: not all token ids are set (require_all)");
  }
}

TokenIds TokenIds::Fallback() noexcept {
  // Indicative ids from spec Section 5.5. DOCUMENTED FALLBACK ONLY -- these are
  // assigned at add_tokens time for the released checkpoint and must normally
  // be read from its config. Do not treat these as authoritative.
  TokenIds t;
  t.image_token = 151667;
  t.box_start = 151668;
  t.box_end = 151669;
  t.ref_start = 151672;
  t.ref_end = 151673;
  t.mask = 151676;
  t.coord_start = 151677;
  t.coord_end = 152677;  // 151677 + 1000
  t.none = 4064;
  t.null_token = 152678;
  t.im_end = 151645;
  t.switch_token = 152679;
  return t;
}

// ---------------------------------------------------------------------------
// Loaders
// ---------------------------------------------------------------------------

TokenIds LoadTokenIds(std::string_view json_text, bool require_all) {
  std::map<std::string, token_id_t> ints;
  std::map<std::string, std::string> objects;

  JsonScanner scanner(json_text);
  scanner.ParseObject(ints, objects);

  TokenIds t;

  // Prefer a dedicated nested object if the config provides one, then fall
  // back to the top-level integers (so a whole config.json also works).
  for (const char* nested : {"token_ids", "locany_token_ids"}) {
    auto it = objects.find(nested);
    if (it != objects.end()) {
      std::map<std::string, token_id_t> nints;
      std::map<std::string, std::string> nobjs;
      JsonScanner ns(it->second);
      ns.ParseObject(nints, nobjs);
      BindFromMap(t, nints);
    }
  }
  BindFromMap(t, ints);

  if (require_all) {
    t.Validate(/*require_all=*/true);
  }
  return t;
}

TokenIds LoadTokenIdsFromFile(const std::string& path, bool require_all) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("LoadTokenIdsFromFile: cannot open '" + path + "'");
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return LoadTokenIds(ss.str(), require_all);
}

}  // namespace la::config
