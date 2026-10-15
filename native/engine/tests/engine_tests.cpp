// Dependency-free unit tests for the Lokahi engine core.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "decoder.h"
#include "half.h"
#include "json.h"
#include "quant.h"
#include "regex.h"
#include "safetensors.h"
#include "sampling.h"
#include "thread_pool.h"
#include "tokenizer.h"
#include "unicode.h"

using namespace lokahi;

namespace {

int g_failures = 0;

#define EXPECT(cond)                                                    \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::fprintf(stderr, "%s:%d: EXPECT(%s) failed\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                     \
    }                                                                   \
  } while (0)

bool throws(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

void test_json() {
  Json value = Json::parse(R"({"a": [1, 2.5, -3e2], "b": {"c": "xé\n"}, "d": null, "e": true})");
  EXPECT(value.at("a").as_array().size() == 3);
  EXPECT(value.at("a").as_array()[2].as_number() == -300.0);
  EXPECT(value.at("b").at("c").as_string() == "x\xc3\xa9\n");
  EXPECT(value.at("d").is_null());
  EXPECT(value.at("e").as_bool());
  EXPECT(value.int_or("missing", 7) == 7);
  EXPECT(throws([] { Json::parse("{\"a\": 1,}"); }));
  EXPECT(throws([] { Json::parse("[1, 2"); }));
  EXPECT(throws([] { Json::parse("{} extra"); }));
}

void test_half() {
  EXPECT(bf16_to_f32(f32_to_bf16(1.0f)) == 1.0f);
  EXPECT(bf16_to_f32(f32_to_bf16(33.941125f)) == 34.0f);  // sqrt(1152) in BF16
  EXPECT(f16_to_f32(0x3C00) == 1.0f);
  EXPECT(f16_to_f32(0xC000) == -2.0f);
  EXPECT(f16_to_f32(0x0001) > 0.0f && f16_to_f32(0x0001) < 1e-7f);  // subnormal
  EXPECT(std::isinf(f16_to_f32(0x7C00)));
}

struct QuantFixture {
  std::vector<uint32_t> words;
  std::vector<uint16_t> scales, biases;
  std::vector<float> dense;  // reference dequantized weights
  Linear linear;
};

QuantFixture make_quant(int rows, int cols, int bits, int group_size, unsigned seed) {
  QuantFixture f;
  std::mt19937 rng(seed);
  const int per_word = 32 / bits;
  const int groups = cols / group_size;
  f.words.assign(static_cast<size_t>(rows) * cols / per_word, 0);
  f.scales.resize(static_cast<size_t>(rows) * groups);
  f.biases.resize(static_cast<size_t>(rows) * groups);
  f.dense.resize(static_cast<size_t>(rows) * cols);
  std::uniform_int_distribution<uint32_t> code(0, (1u << bits) - 1);
  std::uniform_real_distribution<float> uniform(-0.05f, 0.05f);
  for (int r = 0; r < rows; ++r) {
    for (int g = 0; g < groups; ++g) {
      f.scales[static_cast<size_t>(r) * groups + g] = f32_to_bf16(0.01f + std::fabs(uniform(rng)));
      f.biases[static_cast<size_t>(r) * groups + g] = f32_to_bf16(uniform(rng));
    }
    for (int c = 0; c < cols; ++c) {
      uint32_t q = code(rng);
      f.words[(static_cast<size_t>(r) * cols + c) / per_word] |= q << (bits * (c % per_word));
      int g = c / group_size;
      f.dense[static_cast<size_t>(r) * cols + c] =
          bf16_to_f32(f.scales[static_cast<size_t>(r) * groups + g]) * static_cast<float>(q) +
          bf16_to_f32(f.biases[static_cast<size_t>(r) * groups + g]);
    }
  }
  f.linear.rows = rows;
  f.linear.cols = cols;
  f.linear.quant = {group_size, bits};
  f.linear.words = f.words.data();
  f.linear.scales = f.scales.data();
  f.linear.biases = f.biases.data();
  f.linear.scale_dtype = DType::BF16;
  return f;
}

void test_quant_kernels() {
  ThreadPool pool(3);
  for (int bits : {2, 4, 8}) {
    const int rows = 37, cols = 256, group = 64;
    QuantFixture f = make_quant(rows, cols, bits, group, 1234u + bits);
    std::vector<float> row(cols);
    dequantize_row(f.linear, 5, row.data());
    for (int c = 0; c < cols; ++c) EXPECT(row[c] == f.dense[5 * cols + c]);

    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int n = 3;
    std::vector<float> x(static_cast<size_t>(n) * cols);
    for (float& v : x) v = normal(rng);
    std::vector<float> y(static_cast<size_t>(n) * rows), yv(rows);
    matmul(f.linear, x.data(), n, y.data(), pool);
    matvec(f.linear, x.data() + cols, yv.data(), pool);
    for (int t = 0; t < n; ++t) {
      for (int r = 0; r < rows; ++r) {
        double ref = 0.0;
        for (int c = 0; c < cols; ++c) ref += static_cast<double>(f.dense[r * cols + c]) * x[t * cols + c];
        EXPECT(std::fabs(y[t * rows + r] - ref) < 1e-3 * (1.0 + std::fabs(ref)));
        if (t == 1) EXPECT(std::fabs(yv[r] - ref) < 1e-3 * (1.0 + std::fabs(ref)));
      }
    }
  }
}

void test_quant_config() {
  Json config = Json::parse(R"({"quantization": {"group_size": 64, "bits": 4,
      "model.layers.0.mlp.up_proj": {"bits": 8}, "model.norm": false}})");
  EXPECT(quant_spec_for(config, "model.layers.1.mlp.up_proj").bits == 4);
  EXPECT(quant_spec_for(config, "model.layers.0.mlp.up_proj").bits == 8);
  EXPECT(quant_spec_for(config, "model.layers.0.mlp.up_proj").group_size == 64);
  EXPECT(!quant_spec_for(config, "model.norm").quantized());
  EXPECT(!quant_spec_for(Json::parse("{}"), "x").quantized());
  // GPT-OSS's layout: mxfp4 by default, affine overrides, one without a mode.
  const Json oss = Json::parse(R"({"quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4",
      "q": {"group_size": 64, "bits": 8, "mode": "affine"}, "router": {"group_size": 64, "bits": 8}}})");
  EXPECT(quant_spec_for(oss, "expert").mxfp4() && quant_spec_for(oss, "expert").group_size == 32);
  EXPECT(!quant_spec_for(oss, "q").mxfp4() && quant_spec_for(oss, "q").bits == 8);
  EXPECT(!quant_spec_for(oss, "router").mxfp4() && quant_spec_for(oss, "router").group_size == 64);
  EXPECT(throws([] {
    quant_spec_for(Json::parse(R"({"quantization": {"group_size": 64, "bits": 4, "mode": "mxfp4"}})"), "x");
  }));
  EXPECT(throws([] {
    quant_spec_for(Json::parse(R"({"quantization": {"group_size": 16, "bits": 4, "mode": "nvfp4"}})"), "x");
  }));
}

// MXFP4 rows against hand-computed values, including exponent 0 (2^-127).
void test_mxfp4_kernels() {
  Linear l;
  l.rows = 2;
  l.cols = 64;
  l.quant.group_size = 32;
  l.quant.bits = 4;
  l.quant.mode = QuantMode::Mxfp4;
  std::vector<uint32_t> words(16);
  std::vector<uint8_t> exponents = {127, 125, 0, 130};  // 1, 1/4, 2^-127, 8
  const float lut[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
  std::vector<float> expected(128), x(64);
  std::mt19937 rng(7);
  for (int i = 0; i < 128; ++i) {
    const uint32_t code = rng() & 0xF;
    words[static_cast<size_t>(i / 8)] |= code << (4 * (i % 8));
    const float scale = std::ldexp(1.0f, static_cast<int>(exponents[static_cast<size_t>(i / 32)]) - 127);
    expected[static_cast<size_t>(i)] = lut[code] * scale;
  }
  for (int i = 0; i < 64; ++i) x[static_cast<size_t>(i)] = static_cast<float>(i % 7) - 3.0f;
  l.words = words.data();
  l.exponents = exponents.data();
  std::vector<float> row(64);
  for (int r = 0; r < 2; ++r) {
    dequantize_row(l, r, row.data());
    double dot = 0.0;
    for (int c = 0; c < 64; ++c) {
      EXPECT(row[static_cast<size_t>(c)] == expected[static_cast<size_t>(r * 64 + c)]);
      dot += static_cast<double>(expected[static_cast<size_t>(r * 64 + c)]) * x[static_cast<size_t>(c)];
    }
    ThreadPool pool(1);
    float y[2];
    matvec(l, x.data(), y, pool);
    EXPECT(std::fabs(y[r] - dot) <= 1e-5 * std::max(1.0, std::fabs(dot)));
  }
  EXPECT(l.weight_bytes() == 2 * 8 * 4 + 2 * 2);
}

std::string write_safetensors(const std::string& header, const std::vector<uint8_t>& payload) {
  char path[] = "/tmp/lokahi_test_XXXXXX";
  int fd = mkstemp(path);
  EXPECT(fd >= 0);
  close(fd);
  std::string dir = std::string(path) + ".d";
  std::string cmd = "mkdir -p " + dir;
  EXPECT(std::system(cmd.c_str()) == 0);
  std::ofstream out(dir + "/model.safetensors", std::ios::binary);
  uint64_t len = header.size();
  out.write(reinterpret_cast<const char*>(&len), 8);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
  unlink(path);
  return dir;
}

void test_safetensors() {
  std::vector<uint8_t> payload(8 + 4);
  float values[2] = {1.5f, -2.0f};
  std::memcpy(payload.data(), values, 8);
  uint16_t halfs[2] = {0x3C00, 0x4000};
  std::memcpy(payload.data() + 8, halfs, 4);
  std::string dir = write_safetensors(
      R"({"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
      R"("b":{"dtype":"F16","shape":[1,2],"data_offsets":[8,12]}})",
      payload);
  Checkpoint ck = Checkpoint::open_directory(dir);
  EXPECT(ck.tensors().size() == 2);
  EXPECT(ck.get("a").as<float>()[1] == -2.0f);
  EXPECT(ck.get("b").shape.size() == 2 && ck.get("b").dim(1) == 2);
  float out[2];
  load_vector(ck, "b", 2, out);
  EXPECT(out[0] == 1.0f && out[1] == 2.0f);
  EXPECT(throws([&] { ck.get("missing"); }));

  std::string bad = write_safetensors(R"({"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})", payload);
  EXPECT(throws([&] { Checkpoint::open_directory(bad); }));
}

void test_thread_pool() {
  ThreadPool pool(4);
  std::vector<int> hits(1000, 0);
  for (int round = 0; round < 50; ++round) {
    pool.parallel_for(hits.size(), [&](size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) ++hits[i];
    });
  }
  bool all = true;
  for (int h : hits) all = all && h == 50;
  EXPECT(all);
}

void test_unicode() {
  using namespace unicode;
  auto length = [](std::string_view bytes) { return utf8_sequence_length(bytes, 0); };
  EXPECT(length("A") == 1 && length("\xC3\xA9") == 2 && length("\xE2\x82\xAC") == 3);
  EXPECT(length("\xF0\x9F\x99\x82") == 4 && length("\xF4\x8F\xBF\xBF") == 4);
  EXPECT(length("\xC0\x80") == 0);          // overlong
  EXPECT(length("\xE0\x80\x80") == 0);      // overlong
  EXPECT(length("\xED\xA0\x80") == 0);      // surrogate
  EXPECT(length("\xF4\x90\x80\x80") == 0);  // above U+10FFFF
  EXPECT(length("\xE2\x82") == 0 && length("\x80") == 0 && length("\xFF") == 0);
  EXPECT(throws([] { decode_utf8("a\xED\xBF\xBF" "b"); }));
  EXPECT(decode_utf8("a\xC3\xA9\xF0\x9F\x99\x82") == std::u32string({'a', 0xE9, 0x1F642}));

  EXPECT(category('A') == Lu && category('a') == Ll && category(0x01C5) == Lt && category('7') == Nd);
  EXPECT(category(0x4E00) == Lo && category(0x0301) == Mn && category(0x00A0) == Zs && category(0x0378) == Cn);
  EXPECT(category(0xE000) == Co && category(0x10FFFF) == Cn && category(0x200D) == Cf);
  EXPECT(category(0x13460) == Lo);  // Egyptian Hieroglyphs Extended-A, as the reference's Unicode 16 data says
  EXPECT(is_space('\t') && is_space(0x0B) && is_space(0x85) && is_space(0x3000) && is_space(0x2029));
  EXPECT(!is_space(0x200B) && !is_space(0x180E) && !is_space('x'));

  EXPECT(nfc("e\xCC\x81") == "\xC3\xA9");                            // e + U+0301 -> U+00E9
  EXPECT(nfc("\xE2\x84\xAB") == "\xC3\x85");                        // U+212B ANGSTROM SIGN -> U+00C5
  EXPECT(nfc("\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8") == "\xEA\xB0\x81");  // jamo -> U+AC01
  // a + U+0301 (230) + U+0316 (220): reordered, then the acute composes past
  // the lower-class mark.
  EXPECT(nfc("a\xCC\x81\xCC\x96") == "\xC3\xA1\xCC\x96");
  // Dives Akuru U+11935 U+11930 compose in Unicode 13+, but not in the
  // reference normalizer, so they stay apart.
  EXPECT(nfc("\xF0\x91\xA4\xB5\xF0\x91\xA4\xB0") == "\xF0\x91\xA4\xB5\xF0\x91\xA4\xB0");
  EXPECT(std::string(reference()).rfind("tokenizers ", 0) == 0);
}

// All matches of `pattern` in `text`, as code-point [begin, end) pairs.
std::vector<std::pair<size_t, size_t>> matches(const char* pattern, std::string_view text) {
  const Regex regex = Regex::compile(pattern);
  const std::u32string chars = unicode::decode_utf8(text);
  std::vector<std::pair<size_t, size_t>> out;
  size_t from = 0, begin = 0, end = 0;
  while (from <= chars.size() && regex.search(chars, from, &begin, &end)) {
    out.emplace_back(begin, end);
    from = end > begin ? end : end + 1;
  }
  return out;
}

void test_regex() {
  using M = std::vector<std::pair<size_t, size_t>>;
  EXPECT(matches("a+?b", "aaab") == M({{0, 4}}));
  EXPECT(matches("a*+a", "aaa").empty());  // possessive: never gives back
  EXPECT(matches("(?:ab|a)+b", "abab") == M({{0, 4}}));
  EXPECT(matches("(?:a|ab)(?:c|bcd)", "abcd") == M({{0, 4}}));  // leftmost-first, with backtracking
  EXPECT(matches("x(?=y)", "xyxz") == M({{0, 1}}));
  EXPECT(matches("x(?!y)", "xyxz") == M({{2, 3}}));
  EXPECT(matches("a{2}", "aaaaa") == M({{0, 2}, {2, 4}}));
  EXPECT(matches("a{,2}b", "aaab") == M({{1, 4}}));  // {,n} is {0,n}
  EXPECT(matches("a{x", "a{x") == M({{0, 3}}));      // not an interval: literal
  EXPECT(matches("(?i:'s|'ll)", "'S 'Ll '\xC5\xBF") == M({{0, 2}, {3, 6}, {7, 9}}));  // includes U+017F
  EXPECT(matches("\\s+(?!\\S)|\\s+", "a   b") == M({{1, 3}, {3, 4}}));
  EXPECT(matches("[^\\r\\n\\p{L}\\p{N}]?\\p{L}+", "\"Hello, w\xC3\xB6rld") == M({{0, 6}, {7, 13}}));
  EXPECT(matches("\\p{N}{1,3}", "12345") == M({{0, 3}, {3, 5}}));
  EXPECT(matches("[\\p{Lu}\\p{Lt}]+", "abCD\xC7\x85" "e") == M({{2, 5}}));
  EXPECT(matches("\\p{Uppercase_Letter}", "aB") == M({{1, 2}}));
  EXPECT(matches("\\P{L}+|\\p{^N}", "ab12") == M({{0, 1}, {1, 2}, {2, 4}}));
  EXPECT(matches("[\\]\\-a-c]+", "]-abcd") == M({{0, 5}}));
  EXPECT(matches("^ab|cd$", "ab\nab cd\ncd") == M({{0, 2}, {3, 5}, {6, 8}, {9, 11}}));
  EXPECT(matches("\\x{1F642}|\\u00e9", "\xF0\x9F\x99\x82\xC3\xA9") == M({{0, 1}, {1, 2}}));

  for (const char* bad : {"\\w", "(?<name>a)", "[[a]]", "[a&&b]", "\\p{Han}", "*a", "a)", "(a", "[a", "a{3,2}",
                          "(?i:\\p{L})", "(?i:[a])", "(?i:\\u00e9)", "(?=a)*"}) {
    EXPECT(throws([&] { Regex::compile(bad); }));
  }
  // Group repetition is capped well inside small thread stacks.
  std::string under, over;
  for (int i = 0; i < 200; ++i) under += "ab";
  for (int i = 0; i < 300; ++i) over += "ab";
  EXPECT(matches("(?:ab)+", under) == M({{0, 400}}));
  EXPECT(throws([&] { matches("(?:ab)+", over); }));
}

// For every prefix of random id sequences, decode(ids[0, stable_prefix))
// must be a prefix of every later full decode.
bool streaming_is_prefix_stable(const Tokenizer& tokenizer, const std::vector<int32_t>& pool, unsigned seed) {
  std::mt19937 rng(seed);
  for (int trial = 0; trial < 400; ++trial) {
    std::vector<int32_t> ids;
    for (int i = 0; i < 12; ++i) ids.push_back(pool[rng() % pool.size()]);
    for (bool skip : {true, false}) {
      for (size_t n = 0; n <= ids.size(); ++n) {
        std::vector<int32_t> head(ids.begin(), ids.begin() + static_cast<long>(n));
        const size_t stable = tokenizer.stable_prefix(head, skip);
        const std::string emitted =
            tokenizer.decode(std::vector<int32_t>(head.begin(), head.begin() + static_cast<long>(stable)), skip);
        for (size_t m = n; m <= ids.size(); ++m) {
          const std::string later =
              tokenizer.decode(std::vector<int32_t>(ids.begin(), ids.begin() + static_cast<long>(m)), skip);
          if (later.compare(0, emitted.size(), emitted) != 0) return false;
        }
      }
    }
  }
  return true;
}

// Streaming must never emit text that a later token rewrites. Byte-fallback
// groups are the case that makes this nontrivial for SentencePiece models.
void test_tokenizer_stable_prefix() {
  std::string vocab = R"("<unk>": 0, "<s>": 1, "a": 258, "\u2581": 259, "b": 260)";
  for (int b = 0; b < 256; ++b) {
    char entry[32];
    std::snprintf(entry, sizeof(entry), ", \"<0x%02X>\": %d", b, b + 2);
    vocab += entry;
  }
  Json root = Json::parse(
      R"({"added_tokens": [{"id": 1, "content": "<s>", "special": true}],
          "model": {"type": "BPE", "unk_token": "<unk>", "byte_fallback": true, "merges": [], "vocab": {)" +
      vocab +
      R"(}}, "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"}, {"type": "Fuse"}]}})");
  Tokenizer tokenizer = Tokenizer::from_json(root);
  // Bytes of "é" (C3 A9) and "€" (E2 82 AC) plus stray continuation bytes.
  EXPECT(streaming_is_prefix_stable(
      tokenizer, {258, 259, 260, 1, 0xC3 + 2, 0xA9 + 2, 0xE2 + 2, 0x82 + 2, 0xAC + 2, 0x91 + 2, 'Z' + 2}, 5));
  // Without the hold-back, "Z" then 0x91 would rewrite "Z" as two U+FFFD.
  EXPECT(tokenizer.decode({'Z' + 2}, true) == "Z");
  EXPECT(tokenizer.decode({'Z' + 2, 0x91 + 2}, true) == "\xEF\xBF\xBD\xEF\xBF\xBD");
  EXPECT(tokenizer.stable_prefix({258, 'Z' + 2}, true) == 1);
}

// Byte-level tokens can end inside a UTF-8 sequence; decoding is lossy, so
// a held-back incomplete sequence must not be emitted early.
void test_byte_level_stable_prefix() {
  // GPT-2's byte-to-character alphabet: id b is byte b; 256.. are merges.
  std::string vocab;
  int next = 256;
  std::vector<std::string> chars(256);
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    std::string utf8;
    unicode::append_utf8(utf8, static_cast<char32_t>(printable ? b : next++));
    chars[static_cast<size_t>(b)] = utf8;
  }
  for (int b = 0; b < 256; ++b) {
    if (chars[static_cast<size_t>(b)] == "\"" || chars[static_cast<size_t>(b)] == "\\") {
      vocab += (b ? ", " : "") + std::string("\"\\") + chars[static_cast<size_t>(b)] + "\": " + std::to_string(b);
    } else {
      vocab += (b ? ", " : "") + std::string("\"") + chars[static_cast<size_t>(b)] + "\": " + std::to_string(b);
    }
  }
  const std::string e_acute = chars[0xC3] + chars[0xA9];  // "é" as one token, id 256
  const std::string euro_head = chars[0xE2] + chars[0x82];  // first two bytes of "€", id 257
  Json root = Json::parse(R"({"added_tokens": [{"id": 258, "content": "<s>", "special": true}],
      "model": {"type": "BPE", "merges": [], "vocab": {)" + vocab + ", \"" + e_acute + "\": 256, \"" + euro_head +
                          R"(": 257}}, "decoder": {"type": "ByteLevel"}})");
  Tokenizer tokenizer = Tokenizer::from_json(root);
  EXPECT(tokenizer.decode({256, 'a'}, true) == "\xC3\xA9" "a");
  EXPECT(tokenizer.decode({257, 0xAC}, true) == "\xE2\x82\xAC");
  EXPECT(tokenizer.decode({257, 'a'}, true) == "\xEF\xBF\xBD" "a");  // maximal subpart -> one U+FFFD
  EXPECT(tokenizer.stable_prefix({'a', 257}, true) == 1);
  EXPECT(tokenizer.stable_prefix({'a', 257, 0xAC}, true) == 3);
  EXPECT(tokenizer.stable_prefix({'a', 0xE2, 258}, true) == 1);   // a skipped special does not end the run
  EXPECT(tokenizer.stable_prefix({'a', 0xE2, 258}, false) == 3);  // "<s>" ends it when kept
  EXPECT(tokenizer.stable_prefix({'a', 0x80}, true) == 2);        // a stray continuation is final
  EXPECT(streaming_is_prefix_stable(tokenizer, {'a', 256, 257, 258, 0xC3, 0xA9, 0xE2, 0x82, 0xAC, 0xF0, 0x9F,
                                                0x99, 0x80, 0xED, 0xA0, 0xC0},
                                    9));
}

}  // namespace

void test_sampling() {
  // Known-answer vectors for Philox4x32-10 from Random123 (kat_vectors).
  EXPECT((philox4x32_10({0, 0, 0, 0}, {0, 0}) == PhiloxBlock{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}));
  EXPECT((philox4x32_10({0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}, {0xffffffff, 0xffffffff}) ==
          PhiloxBlock{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}));
  EXPECT((philox4x32_10({0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}, {0xa4093822, 0x299f31d0}) ==
          PhiloxBlock{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}));

  // The key orders floats, including signs, zeros and infinities.
  const float ordered[] = {-INFINITY, -1e30f, -2.5f, -1e-30f, -0.0f, 0.0f, 1e-30f, 2.5f, 1e30f, INFINITY};
  for (size_t i = 1; i < sizeof(ordered) / sizeof(ordered[0]); ++i) {
    EXPECT(order_key(ordered[i - 1]) < order_key(ordered[i]));
  }

  const std::vector<float> logits = {0.5f, 3.0f, 2.9f, -1.0f, 3.0f, 0.0f, 1.0f};
  SamplingParams greedy;
  EXPECT(sample_token(logits.data(), 7, greedy, 0) == 1);  // lowest index on ties
  SamplingParams top1{1.0f, 1.0f, 1, 0.0f, 9};
  EXPECT(sample_token(logits.data(), 7, top1, 3) == 1);
  // Filters bound the support: min_p = 1 keeps only the tied maxima.
  SamplingParams max_only{1.5f, 1.0f, 0, 1.0f, 5};
  SamplingParams top3{2.0f, 1.0f, 3, 0.0f, 5};
  SamplingParams nucleus{1.0f, 0.5f, 0, 0.0f, 5};
  for (int step = 0; step < 200; ++step) {
    const int32_t a = sample_token(logits.data(), 7, max_only, step);
    EXPECT(a == 1 || a == 4);
    const int32_t b = sample_token(logits.data(), 7, top3, step);
    EXPECT(b == 1 || b == 2 || b == 4);
    // The two tied maxima hold about 0.36 each, so the strictly greater mass
    // before token 2 is about 0.72 >= 0.5 and the nucleus is {1, 4}.
    const int32_t c = sample_token(logits.data(), 7, nucleus, step);
    EXPECT(c == 1 || c == 4);
  }
  // Deterministic for a seed and step, and the noise matches its definition.
  SamplingParams warm{0.8f, 0.95f, 0, 0.05f, 1234};
  EXPECT(sample_token(logits.data(), 7, warm, 17) == sample_token(logits.data(), 7, warm, 17));
  EXPECT(gumbel_noise(1234, 17, 5) == gumbel_noise(1234, 17, 5));
  EXPECT(gumbel_noise(1234, 17, 5) != gumbel_noise(1234, 18, 5));
  EXPECT(throws([] { validate_sampling(SamplingParams{-1.0f, 1.0f, 0, 0.0f, 0}); }));
  EXPECT(throws([] { validate_sampling(SamplingParams{1.0f, 0.0f, 0, 0.0f, 0}); }));
  EXPECT(throws([] { validate_sampling(SamplingParams{1.0f, 1.0f, 0, 1.5f, 0}); }));
}

// The text configuration of mlx-community/gemma-4-e2b-it-4bit, as published.
static const char* kGemma4E2BConfig = R"({
  "model_type": "gemma4", "eos_token_id": [1, 106, 50], "tie_word_embeddings": true,
  "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
  "text_config": {
    "attention_bias": false, "attention_k_eq_v": false, "enable_moe_block": false, "eos_token_id": 1,
    "expert_intermediate_size": null, "final_logit_softcapping": 30.0, "global_head_dim": 512, "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 1536, "hidden_size_per_layer_input": 256,
    "intermediate_size": 6144,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
      "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
      "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
      "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
      "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
      "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
      "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
      "full_attention"],
    "max_position_embeddings": 131072, "model_type": "gemma4_text", "num_attention_heads": 8, "num_experts": null,
    "num_global_key_value_heads": null, "num_hidden_layers": 35, "num_key_value_heads": 1,
    "num_kv_shared_layers": 20, "rms_norm_eps": 1e-06,
    "rope_parameters": {
      "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
      "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
    "sliding_window": 512, "tie_word_embeddings": true, "top_k_experts": null,
    "use_bidirectional_attention": null, "use_double_wide_mlp": true, "vocab_size": 262144,
    "vocab_size_per_layer_input": 262144}
})";

void test_gemma4_config() {
  const DecoderConfig c = DecoderConfig::from_json(Json::parse(kGemma4E2BConfig));
  EXPECT(c.architecture == "gemma4" && c.num_layers == 35 && !c.uniform_layers());
  EXPECT(c.eos_token_ids == std::vector<int32_t>({1, 106, 50}));
  EXPECT(c.embed_scale == 39.25f);  // sqrt(1536) rounded to BF16
  EXPECT(c.norm_offset == 0.0f && c.attn_scale == 1.0f && c.v_norm && c.per_layer_input == 256);
  for (int l = 0; l < c.num_layers; ++l) {
    const bool full = l % 5 == 4;
    EXPECT(c.layer_is_sliding[static_cast<size_t>(l)] == !full);
    EXPECT(c.head_dim_of(l) == (full ? 512 : 256) && c.kv_heads_of(l) == 1 && !c.k_eq_v(l));
    // Layers 15-34 attend over the last unshared layer of their kind: 13 or 14.
    EXPECT(c.kv_source(l) == (l < 15 ? l : full ? 14 : 13));
    EXPECT(c.intermediate_of(l) == (l < 15 ? 6144 : 12288));
  }
  const std::vector<double> global = c.rope_inv_freq(false), local = c.rope_inv_freq(true);
  EXPECT(global.size() == 256 && local.size() == 128);
  int rotated = 0;
  for (double f : global) rotated += f != 0.0;
  EXPECT(rotated == 64 && global[0] == 1.0 && global[63] != 0.0 && global[64] == 0.0);
  EXPECT(std::fabs(global[1] / std::pow(1e6, -2.0 / 512) - 1.0) < 1e-12);
  EXPECT(std::fabs(local[1] / std::pow(1e4, -2.0 / 256) - 1.0) < 1e-12 && local[127] != 0.0);

  // The same document with text_config fields replaced.
  auto with = [](std::initializer_list<std::pair<const char*, const char*>> changes) {
    std::string text = kGemma4E2BConfig;
    for (const auto& [key, value] : changes) {
      const std::string field = "\"" + std::string(key) + "\": ";
      const size_t at = text.find(field) + field.size();
      text.replace(at, text.find_first_of(",}", at) - at, value);
    }
    return Json::parse(text);
  };
  EXPECT(throws([&] { DecoderConfig::from_json(with({{"enable_moe_block", "true"}})); }));
  EXPECT(throws([&] { DecoderConfig::from_json(with({{"use_bidirectional_attention", "true"}})); }));
  EXPECT(throws([&] { DecoderConfig::from_json(with({{"num_kv_shared_layers", "35"}})); }));
  // num_global_key_value_heads applies to k_eq_v (full) layers only.
  EXPECT(DecoderConfig::from_json(with({{"num_global_key_value_heads", "4"}})).kv_heads_of(4) == 1);
  const DecoderConfig keqv =
      DecoderConfig::from_json(with({{"attention_k_eq_v", "true"}, {"num_global_key_value_heads", "4"}}));
  EXPECT(keqv.k_eq_v(4) && !keqv.k_eq_v(3) && keqv.kv_heads_of(4) == 4 && keqv.kv_heads_of(3) == 1);
  EXPECT(keqv.kv_heads_of(34) == 4 && keqv.kv_source(34) == 14);
}

// The text fields of openai/gpt-oss-20b's config as mlx-community publishes it.
static const char* kGptOss20BConfig = R"({
  "attention_bias": true, "head_dim": 64, "hidden_act": "silu", "hidden_size": 2880, "intermediate_size": 2880,
  "initial_context_length": 4096, "max_position_embeddings": 131072, "model_type": "gpt_oss",
  "num_attention_heads": 64, "num_experts_per_tok": 4, "experts_per_token": 4, "num_hidden_layers": 24,
  "num_key_value_heads": 8, "num_local_experts": 32, "rms_norm_eps": 1e-05, "rope_theta": 150000,
  "rope_scaling": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 32.0, "original_max_position_embeddings": 4096,
                   "rope_type": "yarn", "truncate": false},
  "sliding_window": 128, "swiglu_limit": 7.0, "tie_word_embeddings": false, "vocab_size": 201088,
  "eos_token_id": 200002,
  "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention",
    "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention",
    "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention",
    "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention",
    "full_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention"]
})";

void test_gpt_oss_config() {
  const DecoderConfig c = DecoderConfig::from_json(Json::parse(kGptOss20BConfig));
  EXPECT(c.architecture == "gpt_oss" && c.num_layers == 24 && c.num_experts == 32 && c.experts_per_token == 4);
  EXPECT(c.moe_intermediate == 2880 && c.sliding_window == 128 && c.qkv_bias && c.o_bias && c.attention_sinks);
  EXPECT(c.router == DecoderConfig::Router::TopKSoftmax && c.router_module == "mlp.router" && c.expert_bias);
  EXPECT(c.activation == Activation::ClampedSwiglu && c.swiglu_limit == 7.0f && !c.uniform_layers());
  for (int l = 0; l < 24; ++l) EXPECT(c.layer_is_sliding[static_cast<size_t>(l)] == (l % 2 == 0) && c.is_moe(l));
  EXPECT(c.eos_token_ids == std::vector<int32_t>({200002}));
  // YaRN's attention factor 0.1 ln 32 + 1, squared into the logit scale.
  const double factor = 0.1 * std::log(32.0) + 1.0;
  EXPECT(std::fabs(c.attn_scale / (factor * factor / 8.0) - 1.0) < 1e-6);
  // Untruncated ramp between dimensions 8.09 and 17.4: pairs below extrapolate,
  // pairs above interpolate (divide by 32), pairs between mix.
  const std::vector<double> inv = c.rope_inv_freq(true);
  auto plain = [](int i) { return std::pow(150000.0, -2.0 * i / 64); };
  EXPECT(inv.size() == 32 && inv[0] == 1.0 && std::fabs(inv[8] / plain(8) - 1.0) < 1e-12);
  EXPECT(std::fabs(inv[18] / (plain(18) / 32) - 1.0) < 1e-12 && std::fabs(inv[31] / (plain(31) / 32) - 1.0) < 1e-12);
  const double pi = 3.14159265358979323846;
  const double low = 64 * std::log(4096 / (32 * 2 * pi)) / (2 * std::log(150000.0));
  const double high = 64 * std::log(4096 / (2 * pi)) / (2 * std::log(150000.0));
  const double ramp = (12 - low) / (high - low);
  EXPECT(std::fabs(inv[12] / (plain(12) / 32 * ramp + plain(12) * (1 - ramp)) - 1.0) < 1e-12);
}

int main() {
  test_json();
  test_half();
  test_quant_kernels();
  test_quant_config();
  test_mxfp4_kernels();
  test_safetensors();
  test_thread_pool();
  test_unicode();
  test_regex();
  test_tokenizer_stable_prefix();
  test_byte_level_stable_prefix();
  test_sampling();
  test_gemma4_config();
  test_gpt_oss_config();
  if (g_failures) {
    std::fprintf(stderr, "%d expectation(s) failed\n", g_failures);
    return 1;
  }
  std::printf("lokahi_engine_tests: all passed\n");
  return 0;
}
