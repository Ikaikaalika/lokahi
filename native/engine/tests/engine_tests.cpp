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
#include "qwen3_5_vision.h"
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
  // f32_to_f16 inverts f16_to_f32 on every finite half, and rounds each
  // midpoint between neighbours to the one with an even code, as Metal does.
  for (uint32_t h = 0; h < 0x10000u; ++h) {
    if ((h & 0x7C00u) == 0x7C00u) continue;  // infinities and NaNs
    EXPECT(f32_to_f16(f16_to_f32(static_cast<uint16_t>(h))) == h);
    if ((h & 0x7FFFu) == 0x7BFFu) continue;  // the largest finite half: its midpoint is the overflow edge
    const double low = f16_to_f32(static_cast<uint16_t>(h)), high = f16_to_f32(static_cast<uint16_t>(h + 1));
    const float midpoint = static_cast<float>((low + high) / 2.0);  // exact in float
    EXPECT(f32_to_f16(midpoint) == ((h & 1u) == 0 ? h : h + 1));
    EXPECT(f32_to_f16(std::nextafter(midpoint, static_cast<float>(high))) == h + 1);
  }
  EXPECT(f32_to_f16(65504.0f) == 0x7BFF && f32_to_f16(65519.0f) == 0x7BFF && f32_to_f16(65520.0f) == 0x7C00);
  EXPECT(f32_to_f16(-1e9f) == 0xFC00 && f32_to_f16(1e-9f) == 0 && f32_to_f16(-0.0f) == 0x8000);
  EXPECT((f32_to_f16(std::nanf("")) & 0x7FFFu) > 0x7C00u);
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
  const int groups = cols / group_size;
  const size_t words_per_row = static_cast<size_t>(cols) * bits / 32;
  f.words.assign(static_cast<size_t>(rows) * words_per_row, 0);
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
      // MLX's layout: each row one little-endian bit stream, so a code can
      // straddle two words (3, 5 and 6 bits).
      const size_t bit = static_cast<size_t>(c) * bits;
      uint32_t* row_words = f.words.data() + static_cast<size_t>(r) * words_per_row;
      row_words[bit / 32] |= q << (bit % 32);
      if (bit % 32 + bits > 32) row_words[bit / 32 + 1] |= q >> (32 - bit % 32);
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
  for (int bits : {2, 3, 4, 5, 6, 8}) {
    for (int group : {32, 64, 128}) {
      const int rows = 37, cols = 256;
      QuantFixture f = make_quant(rows, cols, bits, group, 1234u + bits + group);
      std::vector<float> row(cols);
      for (int r : {0, 5, rows - 1}) {
        dequantize_row(f.linear, r, row.data());
        for (int c = 0; c < cols; ++c) EXPECT(row[c] == f.dense[static_cast<size_t>(r) * cols + c]);
      }

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
  // lmstudio-community's 5- and 6-bit MLX builds, and 3 bits; groups must
  // start on word boundaries (16 six-bit codes fill 3 words, 16 three-bit
  // codes do not), and 7 bits is not an MLX width.
  auto spec = [](const char* quant) { return quant_spec_for(Json::parse(quant), "x"); };
  EXPECT(spec(R"({"quantization": {"group_size": 64, "bits": 5, "mode": "affine"}})").bits == 5);
  EXPECT(spec(R"({"quantization": {"group_size": 64, "bits": 6}})").bits == 6);
  EXPECT(spec(R"({"quantization": {"group_size": 32, "bits": 3}})").bits == 3);
  EXPECT(spec(R"({"quantization": {"group_size": 16, "bits": 6}})").group_size == 16);
  EXPECT(throws([&] { spec(R"({"quantization": {"group_size": 16, "bits": 3}})"); }));
  EXPECT(throws([&] { spec(R"({"quantization": {"group_size": 64, "bits": 7}})"); }));
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

// The text configuration of google/gemma-4-12B-it (gemma4_unified), as
// published with its QAT checkpoints; the vision and audio configs omitted.
static const char* kGemma4_12BConfig = R"({
  "architectures": ["Gemma4UnifiedForConditionalGeneration"], "model_type": "gemma4_unified",
  "eos_token_id": [1, 106], "tie_word_embeddings": true,
  "text_config": {
    "attention_bias": false, "attention_dropout": 0.0, "attention_k_eq_v": true, "bos_token_id": 2,
    "dtype": "bfloat16", "enable_moe_block": false, "eos_token_id": 1, "final_logit_softcapping": 30.0,
    "global_head_dim": 512, "head_dim": 256, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 3840,
    "hidden_size_per_layer_input": 0, "initializer_range": 0.02, "intermediate_size": 15360,
    "layer_types": [
      "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
      "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
      "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention"],
    "max_position_embeddings": 262144, "model_type": "gemma4_unified_text", "moe_intermediate_size": null,
    "num_attention_heads": 16, "num_experts": null, "num_global_key_value_heads": 1, "num_hidden_layers": 48,
    "num_key_value_heads": 8, "num_kv_shared_layers": 0, "pad_token_id": 0, "rms_norm_eps": 1e-06,
    "rope_parameters": {
      "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
      "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
    "sliding_window": 1024, "tie_word_embeddings": true, "top_k_experts": null,
    "use_bidirectional_attention": "vision", "use_cache": true, "use_double_wide_mlp": false,
    "vocab_size": 262144, "vocab_size_per_layer_input": 0}
})";

// mlx-community/gemma-3-4b-it-4bit's config: the text fields not listed take
// Transformers' Gemma3TextConfig defaults.
static const char* kGemma3_4BConfig = R"({
  "architectures": ["Gemma3ForConditionalGeneration"], "boi_token_index": 255999, "eoi_token_index": 256000,
  "eos_token_id": [1, 106], "image_token_index": 262144, "initializer_range": 0.02, "mm_tokens_per_image": 256,
  "model_type": "gemma3", "quantization": {"group_size": 64, "bits": 4},
  "text_config": {"hidden_size": 2560, "intermediate_size": 10240, "model_type": "gemma3_text",
                  "num_hidden_layers": 34, "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
                  "sliding_window": 1024},
  "torch_dtype": "bfloat16", "transformers_version": "4.50.0.dev0",
  "vision_config": {"hidden_size": 1152, "image_size": 896, "intermediate_size": 4304,
                    "model_type": "siglip_vision_model", "num_attention_heads": 16, "num_hidden_layers": 27,
                    "patch_size": 14, "vision_use_head": false, "skip_vision": true}
})";

void test_gemma3_config() {
  const DecoderConfig c = DecoderConfig::from_json(Json::parse(kGemma3_4BConfig));
  EXPECT(c.architecture == "gemma3" && c.num_layers == 34 && c.hidden_size == 2560);
  EXPECT(c.vocab_size == 262208 && c.num_heads == 8 && c.num_kv_heads == 4 && c.head_dim == 256);
  EXPECT(c.attn_scale == 0.0625f && c.sliding_window == 1024 && c.eos_token_ids == std::vector<int32_t>({1, 106}));
  EXPECT(c.embed_scale == 50.5f);  // sqrt(2560) rounded to BF16
  EXPECT(c.rope_scaling.kind == RopeScaling::Kind::Linear && c.rope_scaling.factor == 8.0);
  for (int l = 0; l < c.num_layers; ++l) EXPECT(c.layer_is_sliding[static_cast<size_t>(l)] == (l % 6 != 5));
}

void test_gemma4_config() {
  const DecoderConfig c = DecoderConfig::from_json(Json::parse(kGemma4E2BConfig));
  EXPECT(c.architecture == "gemma4" && c.num_layers == 35);
  EXPECT(c.per_layer_embed_scale == 16.0f && c.per_layer_input_scale == 0.70703125f);  // BF16-rounded 2^-0.5
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
  // Without global_head_dim, full layers are 512 wide, as Transformers and
  // MLX-LM default it.
  std::string no_global = kGemma4E2BConfig;
  const std::string global_field = "\"global_head_dim\": 512, ";
  no_global.erase(no_global.find(global_field), global_field.size());
  EXPECT(DecoderConfig::from_json(Json::parse(no_global)).head_dim_of(4) == 512);
  EXPECT(throws([&] { DecoderConfig::from_json(with({{"use_bidirectional_attention", "\"all\""}})); }));

  // Gemma 4 12B: 48 layers, every sixth full; 16 query heads over 8 KV heads
  // of 256 on sliding layers, and over one 512-wide head whose values are its
  // keys on full layers; no per-layer inputs, KV sharing or experts.
  const DecoderConfig g12 = DecoderConfig::from_json(Json::parse(kGemma4_12BConfig));
  EXPECT(g12.architecture == "gemma4_unified" && g12.num_layers == 48 && g12.hidden_size == 3840);
  EXPECT(g12.num_heads == 16 && g12.per_layer_input == 0 && g12.num_experts == 0 && g12.sliding_window == 1024);
  EXPECT(g12.final_logit_softcapping == 30.0f && g12.eos_token_ids == std::vector<int32_t>({1, 106}));
  EXPECT(g12.image_blocks == DecoderConfig::ImageBlocks::SlidingLayers);
  for (int l = 0; l < g12.num_layers; ++l) {
    const bool full = l % 6 == 5;
    EXPECT(g12.layer_is_sliding[static_cast<size_t>(l)] == !full && g12.kv_source(l) == l);
    EXPECT(g12.head_dim_of(l) == (full ? 512 : 256) && g12.kv_heads_of(l) == (full ? 1 : 8));
    EXPECT(g12.k_eq_v(l) == full && g12.intermediate_of(l) == 15360);
  }
  // A text_config without its own model_type takes the document's.
  std::string untyped = kGemma4_12BConfig;
  const std::string text_type = "\"model_type\": \"gemma4_unified_text\", ";
  untyped.erase(untyped.find(text_type), text_type.size());
  EXPECT(DecoderConfig::from_json(Json::parse(untyped)).architecture == "gemma4_unified");
  // Its defaults are Transformers' Gemma4UnifiedTextConfig's: a 1024-token
  // window, "vision" attention and every sixth layer full.
  std::string bare = kGemma4_12BConfig;
  for (const char* field : {"\"sliding_window\": 1024, ", "\"use_bidirectional_attention\": \"vision\", "}) {
    bare.erase(bare.find(field), std::string(field).size());
  }
  const size_t types = bare.find("\"layer_types\": [");
  bare.erase(types, bare.find("],", types) + 2 - types);
  const DecoderConfig defaults = DecoderConfig::from_json(Json::parse(bare));
  EXPECT(defaults.sliding_window == 1024 && defaults.image_blocks == DecoderConfig::ImageBlocks::SlidingLayers);
  for (int l = 0; l < defaults.num_layers; ++l) {
    EXPECT(defaults.layer_is_sliding[static_cast<size_t>(l)] == (l % 6 != 5));
  }
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
  EXPECT(c.activation == Activation::ClampedSwiglu && c.swiglu_limit == 7.0f);
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

// mlx-community/GLM-4.5-Air-4bit's config.json, as published.
static const char* kGlm45AirConfig = R"({
  "architectures": ["Glm4MoeForCausalLM"], "attention_bias": true, "attention_dropout": 0.0,
  "eos_token_id": [151329, 151336, 151338], "first_k_dense_replace": 1, "head_dim": 128, "hidden_act": "silu",
  "hidden_size": 4096, "initializer_range": 0.02, "intermediate_size": 10944, "max_position_embeddings": 131072,
  "model_type": "glm4_moe", "moe_intermediate_size": 1408, "n_group": 1, "n_routed_experts": 128,
  "n_shared_experts": 1, "norm_topk_prob": true, "num_attention_heads": 96, "num_experts_per_tok": 8,
  "num_hidden_layers": 46, "num_key_value_heads": 8, "num_nextn_predict_layers": 1, "pad_token_id": 151329,
  "partial_rotary_factor": 0.5, "quantization": {"group_size": 64, "bits": 4},
  "quantization_config": {"group_size": 64, "bits": 4}, "rms_norm_eps": 1e-05, "rope_scaling": null,
  "rope_theta": 1000000, "routed_scaling_factor": 1.0, "tie_word_embeddings": false, "topk_group": 1,
  "torch_dtype": "bfloat16", "transformers_version": "4.54.0", "use_cache": true, "use_qk_norm": false,
  "vocab_size": 151552
})";

void test_glm_config() {
  const DecoderConfig c = DecoderConfig::from_json(Json::parse(kGlm45AirConfig));
  EXPECT(c.architecture == "glm4_moe" && c.num_layers == 46 && c.num_experts == 128 && c.experts_per_token == 8);
  EXPECT(c.router == DecoderConfig::Router::SigmoidGrouped && c.n_group == 1 && c.routed_scaling == 1.0f);
  EXPECT(c.shared_intermediate == 1408 && c.moe_intermediate == 1408 && c.norm_topk_prob);
  EXPECT(c.qkv_bias && !c.o_bias && !c.qk_norm && c.rotary_dim == 64);
  EXPECT(!c.is_moe(0) && c.is_moe(1) && c.is_moe(45) && c.intermediate_size == 10944);
  EXPECT(c.eos_token_ids == std::vector<int32_t>({151329, 151336, 151338}));
  // RoPE over the first 64 of 128 dimensions, exponents over 64.
  const std::vector<double> inv = c.rope_inv_freq(false);
  EXPECT(inv.size() == 32 && inv[0] == 1.0 && std::fabs(inv[1] / std::pow(1e6, -2.0 / 64) - 1.0) < 1e-12);
}

// deepseek-ai/DeepSeek-V3's config.json, as published (FP8 weights; the
// config alone parses).
static const char* kDeepseekV3Config = R"({
  "architectures": ["DeepseekV3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 0,
  "eos_token_id": 1, "ep_size": 1, "first_k_dense_replace": 3, "hidden_act": "silu", "hidden_size": 7168,
  "initializer_range": 0.02, "intermediate_size": 18432, "kv_lora_rank": 512, "max_position_embeddings": 163840,
  "model_type": "deepseek_v3", "moe_intermediate_size": 2048, "moe_layer_freq": 1, "n_group": 8,
  "n_routed_experts": 256, "n_shared_experts": 1, "norm_topk_prob": true, "num_attention_heads": 128,
  "num_experts_per_tok": 8, "num_hidden_layers": 61, "num_key_value_heads": 128, "num_nextn_predict_layers": 1,
  "q_lora_rank": 1536, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
  "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                          "weight_block_size": [128, 128]},
  "rms_norm_eps": 1e-06, "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 40, "mscale": 1.0,
  "mscale_all_dim": 1.0, "original_max_position_embeddings": 4096, "type": "yarn"}, "rope_theta": 10000,
  "routed_scaling_factor": 2.5, "scoring_func": "sigmoid", "tie_word_embeddings": false, "topk_group": 4,
  "topk_method": "noaux_tc", "torch_dtype": "bfloat16", "transformers_version": "4.33.1", "use_cache": true,
  "v_head_dim": 128, "vocab_size": 129280
})";

// mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx's config.json, as published.
static const char* kDeepseekV2LiteConfig = R"({
  "architectures": ["DeepseekV2ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
  "aux_loss_alpha": 0.001, "bos_token_id": 100000, "eos_token_id": 100001, "first_k_dense_replace": 1,
  "hidden_act": "silu", "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 10944,
  "kv_lora_rank": 512, "max_position_embeddings": 163840, "model_type": "deepseek_v2",
  "moe_intermediate_size": 1408, "moe_layer_freq": 1, "n_group": 1, "n_routed_experts": 64, "n_shared_experts": 2,
  "norm_topk_prob": false, "num_attention_heads": 16, "num_experts_per_tok": 6, "num_hidden_layers": 27,
  "num_key_value_heads": 16, "pretraining_tp": 1, "q_lora_rank": null, "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64, "quantization": {"group_size": 64, "bits": 4}, "rms_norm_eps": 1e-06,
  "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 40, "mscale": 0.707, "mscale_all_dim": 0.707,
                   "original_max_position_embeddings": 4096, "type": "yarn"},
  "rope_theta": 10000, "routed_scaling_factor": 1.0, "scoring_func": "softmax", "seq_aux": true,
  "tie_word_embeddings": false, "topk_group": 1, "topk_method": "greedy", "torch_dtype": "bfloat16",
  "transformers_version": "4.33.1", "use_cache": true, "v_head_dim": 128, "vocab_size": 102400
})";

// moonshotai/Moonlight-16B-A3B-Instruct's config.json, as published.
static const char* kMoonlightConfig = R"({
  "architectures": ["DeepseekV3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
  "aux_loss_alpha": 0.001, "bos_token_id": 163584, "eos_token_id": 163586, "ep_size": 1, "first_k_dense_replace": 1,
  "hidden_act": "silu", "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 11264,
  "kv_lora_rank": 512, "max_position_embeddings": 8192, "model_type": "deepseek_v3", "moe_intermediate_size": 1408,
  "moe_layer_freq": 1, "n_group": 1, "n_routed_experts": 64, "n_shared_experts": 2, "norm_topk_prob": true,
  "num_attention_heads": 16, "num_experts_per_tok": 6, "num_hidden_layers": 27, "num_key_value_heads": 16,
  "num_nextn_predict_layers": 0, "pretraining_tp": 1, "q_lora_rank": null, "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64, "rms_norm_eps": 1e-05, "rope_theta": 50000.0, "routed_scaling_factor": 2.446,
  "scoring_func": "sigmoid", "seq_aux": true, "tie_word_embeddings": false, "topk_group": 1,
  "topk_method": "noaux_tc", "torch_dtype": "bfloat16", "transformers_version": "4.46.3", "use_cache": true,
  "v_head_dim": 128, "vocab_size": 163840
})";

// moonshotai/Kimi-K2-Instruct's config.json, as published.
static const char* kKimiK2Config = R"({
  "architectures": ["DeepseekV3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
  "aux_loss_alpha": 0.001, "bos_token_id": 163584, "eos_token_id": 163585, "first_k_dense_replace": 1,
  "hidden_act": "silu", "hidden_size": 7168, "initializer_range": 0.02, "intermediate_size": 18432,
  "kv_lora_rank": 512, "max_position_embeddings": 131072, "model_type": "kimi_k2", "moe_intermediate_size": 2048,
  "moe_layer_freq": 1, "n_group": 1, "n_routed_experts": 384, "n_shared_experts": 1, "norm_topk_prob": true,
  "num_attention_heads": 64, "num_experts_per_tok": 8, "num_hidden_layers": 61, "num_key_value_heads": 64,
  "num_nextn_predict_layers": 0, "pretraining_tp": 1, "q_lora_rank": 1536, "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64, "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3",
  "quant_method": "fp8", "weight_block_size": [128, 128]}, "rms_norm_eps": 1e-06, "rope_theta": 50000.0,
  "routed_scaling_factor": 2.827, "rope_scaling": {"beta_fast": 1.0, "beta_slow": 1.0, "factor": 32.0,
  "mscale": 1.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 4096, "type": "yarn"},
  "scoring_func": "sigmoid", "seq_aux": true, "tie_word_embeddings": false, "topk_group": 1,
  "topk_method": "noaux_tc", "torch_dtype": "bfloat16", "transformers_version": "4.48.3", "use_cache": true,
  "v_head_dim": 128, "vocab_size": 163840
})";

// zai-org/GLM-4.7-Flash's config.json, as published.
static const char* kGlm47FlashConfig = R"({
  "architectures": ["Glm4MoeLiteForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
  "pad_token_id": 154820, "eos_token_id": [154820, 154827, 154829], "hidden_act": "silu", "hidden_size": 2048,
  "intermediate_size": 10240, "max_position_embeddings": 202752, "model_type": "glm4_moe_lite",
  "moe_intermediate_size": 1536, "topk_method": "noaux_tc", "norm_topk_prob": true, "num_attention_heads": 20,
  "n_group": 1, "topk_group": 1, "n_routed_experts": 64, "n_shared_experts": 1, "routed_scaling_factor": 1.8,
  "num_experts_per_tok": 4, "first_k_dense_replace": 1, "num_hidden_layers": 47, "num_key_value_heads": 20,
  "num_nextn_predict_layers": 1, "partial_rotary_factor": 1.0, "rms_norm_eps": 1e-05, "rope_scaling": null,
  "rope_theta": 1000000, "tie_word_embeddings": false, "dtype": "bfloat16", "transformers_version": "5.0.0rc0",
  "q_lora_rank": 768, "kv_lora_rank": 512, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
  "vocab_size": 154880
})";

void test_deepseek_configs() {
  const double pi = 3.14159265358979323846;
  auto mscale = [](double factor, double m) { return 0.1 * m * std::log(factor) + 1.0; };
  auto plain = [](double base, int i) { return std::pow(base, -2.0 * i / 64); };

  const DecoderConfig v3 = DecoderConfig::from_json(Json::parse(kDeepseekV3Config));
  EXPECT(v3.architecture == "deepseek_v3" && v3.mla() && v3.num_layers == 61 && v3.num_heads == 128);
  EXPECT(v3.kv_lora_rank == 512 && v3.q_lora_rank == 1536 && v3.latent_width() == 576 && v3.head_dim == 192);
  EXPECT(v3.qk_nope_dim == 128 && v3.qk_rope_dim == 64 && v3.v_head_dim == 128 && v3.rotary_dim == 64);
  EXPECT(v3.rope_pairs && v3.rope_factor == 1.0f);
  // The softmax scale takes mscale(40, 1) squared; the attention factor is 1.
  EXPECT(std::fabs(v3.attn_scale / (std::pow(mscale(40, 1.0), 2) / std::sqrt(192.0)) - 1.0) < 1e-6);
  EXPECT(v3.router == DecoderConfig::Router::SigmoidGrouped && v3.exclude_dropped_groups);
  EXPECT(v3.num_experts == 256 && v3.experts_per_token == 8 && v3.n_group == 8 && v3.topk_group == 4);
  EXPECT(v3.moe_intermediate == 2048 && v3.shared_intermediate == 2048 && v3.routed_scaling == 2.5f && v3.norm_topk_prob);
  EXPECT(!v3.is_moe(2) && v3.is_moe(3) && v3.is_moe(60));
  // YaRN over the 64 rope dimensions: the ramp runs from floor(10.5) to
  // ceil(22.5); below it frequencies extrapolate, above it they interpolate.
  const std::vector<double> inv = v3.rope_inv_freq(false);
  EXPECT(inv.size() == 32 && inv[0] == 1.0 && std::fabs(inv[9] / plain(1e4, 9) - 1.0) < 1e-12);
  EXPECT(std::fabs(inv[23] / (plain(1e4, 23) / 40) - 1.0) < 1e-12 && std::fabs(inv[31] / (plain(1e4, 31) / 40) - 1.0) < 1e-12);
  const double low = std::floor(64 * std::log(4096 / (32 * 2 * pi)) / (2 * std::log(1e4)));
  const double high = std::ceil(64 * std::log(4096 / (2 * pi)) / (2 * std::log(1e4)));
  EXPECT(low == 10.0 && high == 23.0);
  const double ramp = (16 - low) / (high - low);
  EXPECT(std::fabs(inv[16] / (plain(1e4, 16) / 40 * ramp + plain(1e4, 16) * (1 - ramp)) - 1.0) < 1e-12);

  const DecoderConfig v2 = DecoderConfig::from_json(Json::parse(kDeepseekV2LiteConfig));
  EXPECT(v2.architecture == "deepseek_v2" && v2.mla() && v2.q_lora_rank == 0 && v2.num_heads == 16 && v2.rope_pairs);
  EXPECT(v2.router == DecoderConfig::Router::Softmax && v2.n_group == 1 && !v2.exclude_dropped_groups);
  EXPECT(v2.num_experts == 64 && v2.experts_per_token == 6 && v2.shared_intermediate == 2816 && !v2.norm_topk_prob);
  EXPECT(v2.routed_scaling == 1.0f && v2.rope_factor == 1.0f && !v2.is_moe(0) && v2.is_moe(1));
  EXPECT(std::fabs(v2.attn_scale / (std::pow(mscale(40, 0.707), 2) / std::sqrt(192.0)) - 1.0) < 1e-6);

  const DecoderConfig moon = DecoderConfig::from_json(Json::parse(kMoonlightConfig));
  EXPECT(moon.architecture == "deepseek_v3" && moon.q_lora_rank == 0 && moon.rms_norm_eps == 1e-5f);
  EXPECT(moon.rope_scaling.kind == RopeScaling::Kind::None && std::fabs(moon.attn_scale * std::sqrt(192.0) - 1.0) < 1e-6);
  EXPECT(moon.routed_scaling == 2.446f && moon.shared_intermediate == 2816 && moon.n_group == 1);
  EXPECT(moon.eos_token_ids == std::vector<int32_t>({163586}));

  const DecoderConfig kimi = DecoderConfig::from_json(Json::parse(kKimiK2Config));
  EXPECT(kimi.architecture == "deepseek_v3" && kimi.num_experts == 384 && kimi.num_heads == 64);
  EXPECT(std::fabs(kimi.attn_scale / (std::pow(mscale(32, 1.0), 2) / std::sqrt(192.0)) - 1.0) < 1e-6);
  // beta_fast = beta_slow = 1: the ramp spans the one dimension around 19.2.
  const std::vector<double> k_inv = kimi.rope_inv_freq(false);
  EXPECT(std::fabs(k_inv[18] / plain(5e4, 18) - 1.0) < 1e-12 && std::fabs(k_inv[20] / (plain(5e4, 20) / 32) - 1.0) < 1e-12);

  // GLM-4.7-Flash: V3's latent attention and routing, values wider than the
  // unrotated keys, the latent norms at 1e-6 under rms_norm_eps 1e-5.
  const DecoderConfig glm = DecoderConfig::from_json(Json::parse(kGlm47FlashConfig));
  EXPECT(glm.architecture == "glm4_moe_lite" && glm.mla() && glm.num_heads == 20 && glm.q_lora_rank == 768);
  EXPECT(glm.head_dim == 256 && glm.v_head_dim == 256 && glm.latent_width() == 576 && glm.rope_pairs);
  EXPECT(glm.rms_norm_eps == 1e-5f && glm.latent_norm_eps == 1e-6f && glm.rope_factor == 1.0f);
  EXPECT(std::fabs(glm.attn_scale * std::sqrt(256.0) - 1.0) < 1e-6);
  EXPECT(glm.router == DecoderConfig::Router::SigmoidGrouped && glm.exclude_dropped_groups && glm.n_group == 1);
  EXPECT(glm.num_experts == 64 && glm.experts_per_token == 4 && glm.shared_intermediate == 1536);
  EXPECT(glm.routed_scaling == 1.8f && glm.norm_topk_prob && !glm.is_moe(0) && glm.is_moe(1) && glm.is_moe(46));
  EXPECT(glm.eos_token_ids == std::vector<int32_t>({154820, 154827, 154829}));
  const std::vector<double> g_inv = glm.rope_inv_freq(false);
  EXPECT(g_inv.size() == 32 && std::fabs(g_inv[1] / std::pow(1e6, -2.0 / 64) - 1.0) < 1e-12);
}

// The text configuration of Qwen/Qwen3.5-0.8B (Transformers' BF16 layout), as
// published; vision_config omitted.
static const char* kQwen35_08BConfig = R"({
  "architectures": ["Qwen3_5ForConditionalGeneration"], "image_token_id": 248056, "model_type": "qwen3_5",
  "text_config": {
    "attention_bias": false, "attention_dropout": 0.0, "attn_output_gate": true, "dtype": "bfloat16",
    "eos_token_id": 248044, "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu",
    "hidden_size": 1024, "initializer_range": 0.02, "intermediate_size": 3584,
    "layer_types": [
      "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention",
      "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention",
      "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention",
      "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    "linear_num_value_heads": 16, "linear_value_head_dim": 128, "max_position_embeddings": 262144,
    "mlp_only_layers": [], "model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false, "num_attention_heads": 8, "num_hidden_layers": 24,
    "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "tie_word_embeddings": true, "use_cache": true,
    "vocab_size": 248320, "mamba_ssm_dtype": "float32",
    "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "default",
                        "rope_theta": 10000000, "partial_rotary_factor": 0.25}},
  "tie_word_embeddings": true, "transformers_version": "4.57.0.dev0", "video_token_id": 248057,
  "vision_end_token_id": 248054, "vision_start_token_id": 248053
})";

// The text configuration of mlx-community/Qwen3.5-35B-A3B-4bit, as
// published; vision_config omitted.
static const char* kQwen35_35BA3BConfig = R"({
  "architectures": ["Qwen3_5MoeForConditionalGeneration"], "eos_token_id": [248046, 248044],
  "image_token_id": 248056, "model_type": "qwen3_5_moe",
  "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
  "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine"},
  "text_config": {
    "attention_bias": false, "attention_dropout": 0.0, "attn_output_gate": true, "dtype": "bfloat16",
    "eos_token_id": 248044, "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu",
    "hidden_size": 2048, "initializer_range": 0.02,
    "layer_types": [
      "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention",
      "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention",
      "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention",
      "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention",
      "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention",
      "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention",
      "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    "linear_num_value_heads": 32, "linear_value_head_dim": 128, "max_position_embeddings": 262144,
    "mlp_only_layers": [], "model_type": "qwen3_5_moe_text", "moe_intermediate_size": 512,
    "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false, "num_attention_heads": 16,
    "num_experts": 256, "num_experts_per_tok": 8, "num_hidden_layers": 40, "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06, "router_aux_loss_coef": 0.001, "shared_expert_intermediate_size": 512,
    "use_cache": true, "vocab_size": 248320, "mamba_ssm_dtype": "float32",
    "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "default",
                        "rope_theta": 10000000, "partial_rotary_factor": 0.25}},
  "tie_word_embeddings": false, "transformers_version": "4.57.0.dev0", "video_token_id": 248057,
  "vision_end_token_id": 248054, "vision_start_token_id": 248053
})";

// Qwen/Qwen3.8-27B's config, as published (Transformers 5's layout, as
// lmstudio-community's and mlx-community's MLX conversions keep it).
static const char* kQwen38_27BConfig = R"({
  "architectures": ["Qwen3_5ForConditionalGeneration"], "image_token_id": 248056, "language_model_only": false,
  "model_type": "qwen3_5",
  "text_config": {
    "attention_bias": false, "attention_dropout": 0.0, "attn_output_gate": true, "bos_token_id": 248044,
    "dtype": "bfloat16", "eos_token_id": 248044, "full_attention_interval": 4, "head_dim": 256,
    "hidden_act": "silu", "hidden_size": 5120, "initializer_range": 0.02, "intermediate_size": 17408,
    "layer_types": [
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    "linear_num_value_heads": 48, "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 262144, "model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false, "num_attention_heads": 24, "num_hidden_layers": 64,
    "num_key_value_heads": 4, "output_gate_type": "swish", "pad_token_id": null, "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
                        "rope_theta": 10000000, "rope_type": "default"},
    "tie_word_embeddings": false, "use_cache": true, "vocab_size": 248320},
  "tie_word_embeddings": false, "transformers_version": "5.8.0.dev0", "video_token_id": 248057,
  "vision_config": {"deepstack_visual_indexes": [], "depth": 27, "hidden_act": "gelu_pytorch_tanh",
                    "hidden_size": 1152, "in_channels": 3, "initializer_range": 0.02, "intermediate_size": 4304,
                    "model_type": "qwen3_5", "num_heads": 16, "num_position_embeddings": 2304,
                    "out_hidden_size": 5120, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2},
  "vision_end_token_id": 248054, "vision_start_token_id": 248053
})";

// Qwen/Qwen3-Next-80B-A3B-Instruct's config, as published.
static const char* kQwen3Next80BConfig = R"({
  "architectures": ["Qwen3NextForCausalLM"], "attention_dropout": 0.0, "bos_token_id": 151643,
  "decoder_sparse_step": 1, "eos_token_id": 151645, "full_attention_interval": 4, "head_dim": 256,
  "hidden_act": "silu", "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 5120,
  "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
  "linear_num_value_heads": 32, "linear_value_head_dim": 128, "max_position_embeddings": 262144,
  "mlp_only_layers": [], "model_type": "qwen3_next", "moe_intermediate_size": 512, "norm_topk_prob": true,
  "num_attention_heads": 16, "num_experts": 512, "num_experts_per_tok": 10, "num_hidden_layers": 48,
  "num_key_value_heads": 2, "output_router_logits": false, "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-06,
  "rope_scaling": null, "rope_theta": 10000000, "router_aux_loss_coef": 0.001,
  "shared_expert_intermediate_size": 512, "tie_word_embeddings": false, "torch_dtype": "bfloat16",
  "transformers_version": "4.57.0.dev0", "use_cache": true, "use_sliding_window": false, "vocab_size": 151936
})";

void test_qwen3_5_configs() {
  auto linear_every_fourth = [](const DecoderConfig& c) {
    for (int l = 0; l < c.num_layers; ++l) {
      if (c.is_linear(l) != ((l + 1) % 4 != 0)) return false;
    }
    return true;
  };
  // Partial RoPE: rotate-half within the first 64 of 256 dimensions, exponents over 64.
  auto rope_64_of_256 = [](const DecoderConfig& c) {
    const std::vector<double> inv = c.rope_inv_freq(false);
    return c.rotary_dim == 64 && inv.size() == 32 && std::fabs(inv[1] / std::pow(1e7, -2.0 / 64) - 1.0) < 1e-12;
  };

  const DecoderConfig small = DecoderConfig::from_json(Json::parse(kQwen35_08BConfig));
  EXPECT(small.architecture == "qwen3_5" && small.num_layers == 24 && small.hidden_size == 1024);
  EXPECT(small.num_heads == 8 && small.num_kv_heads == 2 && small.head_dim == 256 && small.qk_norm);
  EXPECT(small.attn_output_gate && !small.shared_expert_gate && small.intermediate_size == 3584);
  EXPECT(small.linear_k_heads == 16 && small.linear_v_heads == 16 && small.linear_k_dim == 128);
  EXPECT(small.linear_v_dim == 128 && small.linear_conv == 4 && small.linear_channels() == 6144);
  EXPECT(linear_every_fourth(small) && small.num_experts == 0 && !small.is_moe(0) && rope_64_of_256(small));
  EXPECT(std::fabs(small.attn_scale * 16.0f - 1.0f) < 1e-6f && small.eos_token_ids == std::vector<int32_t>({248044}));

  const DecoderConfig moe = DecoderConfig::from_json(Json::parse(kQwen35_35BA3BConfig));
  EXPECT(moe.architecture == "qwen3_5_moe" && moe.num_layers == 40 && moe.num_heads == 16);
  EXPECT(moe.linear_k_heads == 16 && moe.linear_v_heads == 32 && moe.linear_channels() == 8192);
  EXPECT(moe.num_experts == 256 && moe.experts_per_token == 8 && moe.moe_intermediate == 512);
  EXPECT(moe.shared_intermediate == 512 && moe.shared_expert_gate && moe.norm_topk_prob);
  EXPECT(moe.router == DecoderConfig::Router::Softmax && moe.routed_scaling == 1.0f);
  EXPECT(moe.is_moe(0) && moe.is_moe(3) && moe.is_moe(39) && linear_every_fourth(moe) && rope_64_of_256(moe));
  EXPECT(moe.eos_token_ids == std::vector<int32_t>({248046, 248044}));

  // Qwen3.8-27B is a Qwen3.5 model: 16 groups of three Gated DeltaNet layers
  // and one gated full-attention layer, 48 value heads over 16 key heads.
  const DecoderConfig q38 = DecoderConfig::from_json(Json::parse(kQwen38_27BConfig));
  EXPECT(q38.architecture == "qwen3_5" && q38.num_layers == 64 && q38.hidden_size == 5120);
  EXPECT(q38.num_heads == 24 && q38.num_kv_heads == 4 && q38.head_dim == 256 && q38.qk_norm);
  EXPECT(q38.attn_output_gate && q38.intermediate_size == 17408 && q38.num_experts == 0);
  EXPECT(q38.linear_k_heads == 16 && q38.linear_v_heads == 48 && q38.linear_k_dim == 128);
  EXPECT(q38.linear_v_dim == 128 && q38.linear_channels() == 10240 && q38.vocab_size == 248320);
  EXPECT(linear_every_fourth(q38) && rope_64_of_256(q38) && q38.eos_token_ids == std::vector<int32_t>({248044}));
  // output_gate_type names the Gated DeltaNet norm's gate: SiLU ("swish"),
  // as in Qwen3.5. Another activation fails closed.
  std::string silu_gate = kQwen38_27BConfig, sigmoid_gate = kQwen38_27BConfig;
  silu_gate.replace(silu_gate.find("\"swish\""), 7, "\"silu\"");
  sigmoid_gate.replace(sigmoid_gate.find("\"swish\""), 7, "\"sigmoid\"");
  EXPECT(DecoderConfig::from_json(Json::parse(silu_gate)).linear_channels() == 10240);
  EXPECT(throws([&] { DecoderConfig::from_json(Json::parse(sigmoid_gate)); }));

  const DecoderConfig next = DecoderConfig::from_json(Json::parse(kQwen3Next80BConfig));
  EXPECT(next.architecture == "qwen3_next" && next.num_layers == 48 && next.intermediate_size == 5120);
  EXPECT(next.num_experts == 512 && next.experts_per_token == 10 && next.norm_topk_prob && next.shared_expert_gate);
  EXPECT(next.is_moe(0) && next.is_moe(47) && linear_every_fourth(next) && rope_64_of_256(next));
  EXPECT(next.attn_output_gate && next.linear_v_heads == 32 && next.eos_token_ids == std::vector<int32_t>({151645}));

  // Interleaved M-RoPE over the 32 rotary frequencies, sections [11, 11, 10]:
  // rows at 1, 4, ..., 31, columns at 2, 5, ..., 29, time elsewhere.
  auto interleaved = [](const DecoderConfig& c) {
    if (c.rope_axes.size() != 32) return false;
    for (int i = 0; i < 32; ++i) {
      const int expected = i % 3 == 1 && i < 33 ? 1 : i % 3 == 2 && i < 30 ? 2 : 0;
      if (c.rope_axes[static_cast<size_t>(i)] != expected) return false;
    }
    return true;
  };
  EXPECT(interleaved(small) && interleaved(moe) && interleaved(q38) && next.rope_axes.empty());
  // Without mrope_section, Transformers and MLX-VLM use [11, 11, 10].
  std::string unsectioned = kQwen35_08BConfig;
  const std::string section = "\"mrope_section\": [11, 11, 10], ";
  unsectioned.erase(unsectioned.find(section), section.size());
  EXPECT(interleaved(DecoderConfig::from_json(Json::parse(unsectioned))));
  // Uneven sections [16, 6, 3]: rows at 1, 4, ..., 16, columns at 2, 5, 8.
  std::string uneven = kQwen35_08BConfig;
  uneven.replace(uneven.find("[11, 11, 10]"), 12, "[16, 6, 3]");
  const DecoderConfig custom = DecoderConfig::from_json(Json::parse(uneven));
  bool sections = custom.rope_axes.size() == 32;
  for (int i = 0; sections && i < 32; ++i) {
    const int expected = i % 3 == 1 && i < 18 ? 1 : i % 3 == 2 && i < 9 ? 2 : 0;
    sections = custom.rope_axes[static_cast<size_t>(i)] == expected;
  }
  EXPECT(sections);
  EXPECT(small.injection_rules(7).mrope && !small.injection_rules(7).bidirectional_blocks &&
         small.injection_rules(7).rope_offset == 7);
}

// Qwen/Qwen3-VL-2B-Instruct's config.json, as published.
static const char* kQwen3VL2BConfig = R"({
  "architectures": ["Qwen3VLForConditionalGeneration"], "image_token_id": 151655, "model_type": "qwen3_vl",
  "text_config": {
    "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 151643, "dtype": "bfloat16",
    "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu", "hidden_size": 2048, "initializer_range": 0.02,
    "intermediate_size": 6144, "max_position_embeddings": 262144, "model_type": "qwen3_vl_text",
    "num_attention_heads": 16, "num_hidden_layers": 28, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
    "rope_scaling": {"mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default"},
    "rope_theta": 5000000, "tie_word_embeddings": true, "use_cache": true, "vocab_size": 151936},
  "tie_word_embeddings": true, "transformers_version": "4.57.0.dev0", "video_token_id": 151656,
  "vision_config": {
    "deepstack_visual_indexes": [5, 11, 17], "depth": 24, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1024,
    "in_channels": 3, "initializer_range": 0.02, "intermediate_size": 4096, "model_type": "qwen3_vl",
    "num_heads": 16, "num_position_embeddings": 2304, "out_hidden_size": 2048, "patch_size": 16,
    "spatial_merge_size": 2, "temporal_patch_size": 2},
  "vision_end_token_id": 151653, "vision_start_token_id": 151652
})";

void test_qwen3_vl_config() {
  // Sections [temporal 24, rows 20, columns 20] over 64 frequencies: rows at
  // 1, 4, ..., 58, columns at 2, 5, ..., 59, the rest temporal.
  auto sections = [](const DecoderConfig& c) {
    if (c.rope_axes.size() != 64) return false;
    for (int i = 0; i < 64; ++i) {
      const int expected = i % 3 == 1 && i < 60 ? 1 : i % 3 == 2 && i < 60 ? 2 : 0;
      if (c.rope_axes[static_cast<size_t>(i)] != expected) return false;
    }
    return true;
  };
  const DecoderConfig c = DecoderConfig::from_json(Json::parse(kQwen3VL2BConfig));
  EXPECT(c.architecture == "qwen3_vl" && c.num_layers == 28 && c.hidden_size == 2048 && c.head_dim == 128);
  EXPECT(c.qk_norm && c.num_kv_heads == 8 && c.rotary_dim == 0 && sections(c) && c.deepstack);
  EXPECT(std::fabs(c.rope_theta - 5e6f) < 1.0f && c.eos_token_ids == std::vector<int32_t>({151645}));
  EXPECT(c.injection_rules(0).deepstack_layers == 28 && c.injection_rules(0).mrope);
  // Transformers 5 writes rope_parameters, with rope_theta inside, which
  // wins over the rope_theta beside it.
  std::string v5 = kQwen3VL2BConfig;
  const std::string scaling =
      R"("rope_scaling": {"mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default"},)";
  const std::string parameters =
      R"("rope_parameters": {"mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default", "rope_theta": 4000000},)";
  v5.replace(v5.find(scaling), scaling.size(), parameters);
  const DecoderConfig five = DecoderConfig::from_json(Json::parse(v5));
  EXPECT(sections(five) && std::fabs(five.rope_theta - 4e6f) < 1.0f);
  // Beside rope_scaling, Transformers ignores rope_parameters entirely: the
  // sections and theta come from rope_scaling and the rope_theta beside it.
  std::string both = kQwen3VL2BConfig;
  std::string other = parameters;
  other.replace(other.find("[24, 20, 20]"), 12, "[32, 16, 16]");
  both.insert(both.find(scaling), other);
  const DecoderConfig legacy = DecoderConfig::from_json(Json::parse(both));
  EXPECT(sections(legacy) && std::fabs(legacy.rope_theta - 5e6f) < 1.0f);
  // Without any rope_theta, Transformers' Qwen3-VL default is 500000.
  std::string untheta = kQwen3VL2BConfig;
  const std::string theta = R"("rope_theta": 5000000, )";
  untheta.erase(untheta.find(theta), theta.size());
  EXPECT(std::fabs(DecoderConfig::from_json(Json::parse(untheta)).rope_theta - 5e5f) < 1.0f);
  // Without mrope_section, Transformers and MLX-VLM use [24, 20, 20].
  std::string unsectioned = kQwen3VL2BConfig;
  const std::string section = R"("mrope_section": [24, 20, 20], )";
  unsectioned.erase(unsectioned.find(section), section.size());
  EXPECT(sections(DecoderConfig::from_json(Json::parse(unsectioned))));
  // Text-only families take no deepstack rows.
  EXPECT(DecoderConfig::from_json(Json::parse(kQwen35_08BConfig)).injection_rules(0).deepstack_layers == 0);

  const Json processor = Json::parse(R"({"size": {"longest_edge": 16777216, "shortest_edge": 65536}})");
  const Qwen35VisionConfig vision = Qwen35VisionConfig::from_json(Json::parse(kQwen3VL2BConfig), &processor);
  EXPECT(vision.deepstack == std::vector<int>({5, 11, 17}) && vision.depth == 24 && vision.text_hidden_size == 2048);
}

// Qwen/Qwen2.5-VL-3B-Instruct's config.json, as published (Transformers 4:
// the text fields at the top level).
static const char* kQwen25VL3BConfig = R"({
  "architectures": ["Qwen2_5_VLForConditionalGeneration"], "attention_dropout": 0.0, "bos_token_id": 151643,
  "eos_token_id": 151645, "vision_start_token_id": 151652, "vision_end_token_id": 151653,
  "vision_token_id": 151654, "image_token_id": 151655, "video_token_id": 151656, "hidden_act": "silu",
  "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 11008, "max_position_embeddings": 128000,
  "max_window_layers": 70, "model_type": "qwen2_5_vl", "num_attention_heads": 16, "num_hidden_layers": 36,
  "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "sliding_window": 32768,
  "tie_word_embeddings": true, "torch_dtype": "bfloat16", "transformers_version": "4.41.2", "use_cache": true,
  "use_sliding_window": false,
  "vision_config": {
    "depth": 32, "hidden_act": "silu", "hidden_size": 1280, "intermediate_size": 3420, "num_heads": 16,
    "in_chans": 3, "out_hidden_size": 2048, "patch_size": 14, "spatial_merge_size": 2, "spatial_patch_size": 14,
    "window_size": 112, "fullatt_block_indexes": [7, 15, 23, 31], "tokens_per_second": 2,
    "temporal_patch_size": 2},
  "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
  "vocab_size": 151936
})";

void test_qwen2_5_vl_config() {
  // Chunked sections [temporal 16, rows 24, columns 24] over 64 frequencies.
  auto chunked = [](const DecoderConfig& c, int temporal, int rows) {
    if (c.rope_axes.size() != 64) return false;
    for (int i = 0; i < 64; ++i) {
      if (c.rope_axes[static_cast<size_t>(i)] != (i < temporal ? 0 : i < temporal + rows ? 1 : 2)) return false;
    }
    return true;
  };
  const DecoderConfig c = DecoderConfig::from_json(Json::parse(kQwen25VL3BConfig));
  EXPECT(c.architecture == "qwen2_5_vl" && c.num_layers == 36 && c.hidden_size == 2048 && c.head_dim == 128);
  EXPECT(c.qkv_bias && !c.qk_norm && c.num_kv_heads == 2 && c.rotary_dim == 0 && chunked(c, 16, 24));
  EXPECT(std::fabs(c.rope_theta - 1e6f) < 1.0f && c.eos_token_ids == std::vector<int32_t>({151645}));
  EXPECT(c.injection_rules(3).mrope && c.injection_rules(3).deepstack_layers == 0 &&
         !c.injection_rules(3).bidirectional_blocks && c.injection_rules(3).rope_offset == 3);
  // Uneven sections, and Transformers' default [16, 24, 24] without any.
  std::string uneven = kQwen25VL3BConfig;
  uneven.replace(uneven.find("[16, 24, 24]"), 12, "[10, 30, 24]");
  EXPECT(chunked(DecoderConfig::from_json(Json::parse(uneven)), 10, 30));
  std::string unsectioned = kQwen25VL3BConfig;
  const std::string section = R"(, "mrope_section": [16, 24, 24])";
  unsectioned.erase(unsectioned.find(section), section.size());
  EXPECT(chunked(DecoderConfig::from_json(Json::parse(unsectioned)), 16, 24));
  // Outside Qwen2.5-VL, {"type": "mrope"} is no scaling the Llama family knows.
  std::string plain = kQwen25VL3BConfig;
  const std::string type = R"("model_type": "qwen2_5_vl")";
  plain.replace(plain.find(type), type.size(), R"("model_type": "qwen2")");
  EXPECT(throws([&] { DecoderConfig::from_json(Json::parse(plain)); }));
  // Transformers 5 saves text_config with rope_parameters (rope_theta inside)
  // and layer_types; without any rope_theta its default is 1000000.
  const std::string v5 = R"({"model_type": "qwen2_5_vl", "image_token_id": 151655, "text_config": {
    "model_type": "qwen2_5_vl_text", "hidden_size": 2048, "intermediate_size": 11008, "num_hidden_layers": 2,
    "num_attention_heads": 16, "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "vocab_size": 151936,
    "use_sliding_window": false, "sliding_window": null, "layer_types": ["full_attention", "full_attention"],
    "rope_parameters": {"type": "mrope", "mrope_section": [8, 28, 28], "rope_theta": 250000.0, "rope_type": "default"}},
    "vision_config": {}})";
  const DecoderConfig five = DecoderConfig::from_json(Json::parse(v5));
  EXPECT(five.architecture == "qwen2_5_vl" && chunked(five, 8, 28) && std::fabs(five.rope_theta - 2.5e5f) < 1.0f);
  std::string untheta = v5;
  const std::string theta = R"(, "rope_theta": 250000.0)";
  untheta.erase(untheta.find(theta), theta.size());
  EXPECT(std::fabs(DecoderConfig::from_json(Json::parse(untheta)).rope_theta - 1e6f) < 1.0f);

  const Json processor = Json::parse(R"({"min_pixels": 3136, "max_pixels": 12845056, "patch_size": 14,
    "temporal_patch_size": 2, "merge_size": 2})");
  const Qwen35VisionConfig vision = Qwen35VisionConfig::from_json(Json::parse(kQwen25VL3BConfig), &processor);
  EXPECT(vision.qwen2_5() && vision.depth == 32 && vision.hidden_size == 1280 && vision.patch_size == 14);
  EXPECT(vision.window == 4 && vision.full_attention == std::vector<int>({7, 15, 23, 31}) && vision.deepstack.empty());
  EXPECT(vision.position_side == 0 && vision.text_hidden_size == 2048 && vision.factor() == 28);
  EXPECT(vision.min_pixels == 3136 && vision.max_pixels == 12845056 && std::fabs(vision.mean[0] - 0.48145466f) < 1e-7f);
  // A window_size that is not a multiple of the merged patch floors, as in
  // Transformers (112 // 2 // 14 == 4, 120 // 2 // 14 == 4).
  std::string wider = kQwen25VL3BConfig;
  wider.replace(wider.find("\"window_size\": 112"), 18, "\"window_size\": 120");
  EXPECT(Qwen35VisionConfig::from_json(Json::parse(wider), &processor).window == 4);
}

// Qwen/Qwen2-VL-2B-Instruct's config.json, as published.
static const char* kQwen2VL2BConfig = R"({
  "architectures": ["Qwen2VLForConditionalGeneration"], "attention_dropout": 0.0, "bos_token_id": 151643,
  "eos_token_id": 151645, "vision_start_token_id": 151652, "vision_end_token_id": 151653,
  "vision_token_id": 151654, "image_token_id": 151655, "video_token_id": 151656, "hidden_act": "silu",
  "hidden_size": 1536, "initializer_range": 0.02, "intermediate_size": 8960, "max_position_embeddings": 32768,
  "max_window_layers": 28, "model_type": "qwen2_vl", "num_attention_heads": 12, "num_hidden_layers": 28,
  "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "sliding_window": 32768,
  "tie_word_embeddings": true, "torch_dtype": "bfloat16", "transformers_version": "4.41.2", "use_cache": true,
  "use_sliding_window": false,
  "vision_config": {
    "depth": 32, "embed_dim": 1280, "mlp_ratio": 4, "num_heads": 16, "in_chans": 3, "hidden_size": 1536,
    "patch_size": 14, "spatial_merge_size": 2, "spatial_patch_size": 14, "temporal_patch_size": 2},
  "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
  "vocab_size": 151936
})";

void test_qwen2_vl_config() {
  const DecoderConfig c = DecoderConfig::from_json(Json::parse(kQwen2VL2BConfig));
  EXPECT(c.architecture == "qwen2_vl" && c.num_layers == 28 && c.hidden_size == 1536 && c.head_dim == 128);
  EXPECT(c.qkv_bias && c.num_kv_heads == 2 && c.rope_axes.size() == 64 && c.rope_axes[15] == 0 &&
         c.rope_axes[16] == 1 && c.rope_axes[39] == 1 && c.rope_axes[40] == 2 && c.rope_axes[63] == 2);
  EXPECT(std::fabs(c.rope_theta - 1e6f) < 1.0f && c.injection_rules(0).mrope &&
         c.injection_rules(0).deepstack_layers == 0);
  const Json processor = Json::parse(R"({"min_pixels": 3136, "max_pixels": 12845056, "patch_size": 14,
    "temporal_patch_size": 2, "merge_size": 2})");
  const Qwen35VisionConfig vision = Qwen35VisionConfig::from_json(Json::parse(kQwen2VL2BConfig), &processor);
  // embed_dim is the tower's width, mlp_ratio sizes its MLP, hidden_size is the text width.
  EXPECT(vision.tower == Qwen35VisionConfig::Tower::Qwen2 && vision.hidden_size == 1280 &&
         vision.intermediate_size == 5120 && vision.text_hidden_size == 1536);
  EXPECT(!vision.qwen2_5() && !vision.qwen3_5() && vision.window == 0 && vision.full_attention.empty() &&
         vision.position_side == 0 && vision.factor() == 28 && vision.head_dim() == 80);
}

// Qwen35VisionEncoder's geometry reads only the config; this encoder runs nothing.
class GeometryEncoder final : public Qwen35VisionEncoder {
 public:
  explicit GeometryEncoder(Qwen35VisionConfig c) : Qwen35VisionEncoder(weights(std::move(c))) {}
  const char* backend() const override { return "test"; }
  ImageFeatures encode_features(const Image&) override { return {}; }
  using Qwen35VisionEncoder::run_order;

 private:
  static std::shared_ptr<const Qwen35VisionWeights> weights(Qwen35VisionConfig c) {
    auto w = std::make_shared<Qwen35VisionWeights>();
    w->config = std::move(c);
    return w;
  }
};

void test_qwen2_5_vl_window_order() {
  // A 6 x 10 patch grid (3 x 5 merged tokens) in windows of 2 x 2 tokens, as
  // Transformers' get_vision_window_index orders it: merged tokens 0 1 5 6 |
  // 2 3 7 8 | 4 9 | 10 11 | 12 13 | 14, each token's four patches together.
  Qwen35VisionConfig c;
  c.tower = Qwen35VisionConfig::Tower::Qwen2_5;
  c.merge = 2;
  c.window = 2;
  std::vector<int32_t> bounds;
  const std::vector<int> order = GeometryEncoder(c).run_order(6, 10, &bounds);
  std::vector<int> expected;
  for (int token : {0, 1, 5, 6, 2, 3, 7, 8, 4, 9, 10, 11, 12, 13, 14}) {
    for (int i = 0; i < 4; ++i) expected.push_back(token * 4 + i);
  }
  EXPECT(order == expected);
  std::vector<int32_t> windows;
  int begin = 0;
  for (int length : {16, 16, 8, 8, 8, 4}) {
    for (int i = 0; i < length; ++i) windows.insert(windows.end(), {begin, begin + length});
    begin += length;
  }
  EXPECT(bounds == windows);
  // Outside Qwen2.5-VL, patches keep merge-block order and see the whole image.
  c.tower = Qwen35VisionConfig::Tower::Qwen3_5;
  const std::vector<int> identity = GeometryEncoder(c).run_order(2, 4, &bounds);
  EXPECT(identity == std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7}) && bounds.size() == 16 && bounds[6] == 0 &&
         bounds[7] == 8);
}

void test_mrope_injection_plan() {
  // Two images in a 20-token prefill at position 10, after earlier images
  // moved RoPE positions 5 ahead: a 2x3 grid at 12 and a 4x2 grid at 20.
  PromptInjection injection;
  injection.blocks = {{12, 18}, {20, 28}};
  injection.grids = {{2, 3}, {4, 2}};
  for (int pos = 12; pos < 28; ++pos) {
    if (pos >= 18 && pos < 20) continue;
    injection.positions.push_back(pos);
    injection.rows.push_back(static_cast<float>(pos));
  }
  const InjectionRules mrope{false, true, 5};
  const InjectionPlan plan = plan_injection(injection, 10, 20, 1, 4, mrope);
  std::vector<int32_t> expected = {15, 15, 15, 16, 16, 16};  // text: index + 5
  for (int j = 0; j < 6; ++j) expected.insert(expected.end(), {17, 17 + j / 3, 17 + j % 3});
  expected.insert(expected.end(), {20, 20, 20, 21, 21, 21});  // 5 + max(2, 3) - 6 = 2 ahead
  for (int j = 0; j < 8; ++j) expected.insert(expected.end(), {22, 22 + j / 2, 22 + j % 2});
  expected.insert(expected.end(), {26, 26, 26, 27, 27, 27});  // 2 + max(4, 2) - 8 = -2
  EXPECT(plan.rope == expected && plan.rope_offset == -2);
  // Causal images: passes of 4 may split them, so the plan keeps no blocks.
  EXPECT(plan.blocks.empty() && plan.reach == 0 && plan.pass_length(0, 4) == 4 && plan.last_key[3] == 13);

  // A text-only prefill carries the offset through.
  const InjectionPlan text = plan_injection(PromptInjection{}, 30, 3, 1, 4, InjectionRules{false, true, -2});
  EXPECT(text.rope.empty() && text.rope_offset == -2);

  auto fails = [&](PromptInjection bad, InjectionRules rules) {
    return throws([&] { plan_injection(bad, 10, 20, 1, 4, rules); });
  };
  PromptInjection no_grids = injection;
  no_grids.grids.clear();
  PromptInjection short_grid = injection;
  short_grid.grids[1] = {3, 2};
  PromptInjection outside = injection;
  outside.positions.push_back(18);
  outside.rows.push_back(18.0f);
  EXPECT(!fails(injection, mrope));
  EXPECT(fails(no_grids, mrope) && fails(short_grid, mrope) && fails(outside, mrope));
  EXPECT(fails(injection, InjectionRules{false, false, 0}));  // grids for a model without M-RoPE
  EXPECT(fails(no_grids, InjectionRules{true, false, 0}));    // a bidirectional block must fit a chunk of 4
  EXPECT(!throws([&] { plan_injection(no_grids, 10, 20, 1, 8, InjectionRules{true, false, 0}); }));

  // Deepstack rows: as many as the injected rows, for no more layers than allowed.
  PromptInjection deep = injection;
  deep.layer_rows.assign(2, injection.rows);
  EXPECT(!fails(deep, InjectionRules{false, true, 5, 2}));
  EXPECT(fails(deep, InjectionRules{false, true, 5, 1}) && fails(deep, mrope));
  deep.layer_rows[1].pop_back();
  EXPECT(fails(deep, InjectionRules{false, true, 5, 2}));
}

void test_qwen3_5_vision_config() {
  // Qwen/Qwen3.5-0.8B's vision fields and preprocessor_config.json.
  const Json config = Json::parse(R"({
    "model_type": "qwen3_5", "image_token_id": 248056, "text_config": {"hidden_size": 1024},
    "vision_config": {"deepstack_visual_indexes": [], "depth": 12, "hidden_act": "gelu_pytorch_tanh",
                      "hidden_size": 768, "in_channels": 3, "intermediate_size": 3072, "model_type": "qwen3_5",
                      "num_heads": 12, "num_position_embeddings": 2304, "out_hidden_size": 1024,
                      "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2}})");
  const Json processor = Json::parse(R"({"size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2, "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5], "image_processor_type": "Qwen2VLImageProcessorFast"})");
  const Qwen35VisionConfig c = Qwen35VisionConfig::from_json(config, &processor);
  EXPECT(c.depth == 12 && c.hidden_size == 768 && c.head_dim() == 64 && c.position_side == 48);
  EXPECT(c.factor() == 32 && c.min_pixels == 65536 && c.max_pixels == 16777216 && c.text_hidden_size == 1024);
  EXPECT(c.mean[0] == 0.5f && c.std[2] == 0.5f && std::fabs(c.rescale * 255.0f - 1.0f) < 1e-6f);
  // min_pixels and max_pixels override the size's edges; CLIP's mean and std are the default.
  const Json legacy = Json::parse(R"({"min_pixels": 3136, "max_pixels": 12845056})");
  const Qwen35VisionConfig old = Qwen35VisionConfig::from_json(config, &legacy);
  EXPECT(old.min_pixels == 3136 && old.max_pixels == 12845056 && std::fabs(old.mean[0] - 0.48145466f) < 1e-7f);
  EXPECT(throws([&] { Qwen35VisionConfig::from_json(config, nullptr); }));

  // Qwen/Qwen3.8-27B's tower: 27 blocks 1152 wide (16 heads of 72), without
  // deepstack, and its preprocessor_config.json.
  const Json q38_processor = Json::parse(R"({"size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2, "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5], "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast"})");
  const Qwen35VisionConfig q38 = Qwen35VisionConfig::from_json(Json::parse(kQwen38_27BConfig), &q38_processor);
  EXPECT(q38.tower == Qwen35VisionConfig::Tower::Qwen3_5 && q38.depth == 27 && q38.hidden_size == 1152);
  EXPECT(q38.intermediate_size == 4304 && q38.num_heads == 16 && q38.head_dim() == 72);
  EXPECT(q38.position_side == 48 && q38.factor() == 32 && q38.text_hidden_size == 5120);
  EXPECT(q38.deepstack.empty() && q38.min_pixels == 65536 && q38.max_pixels == 16777216);
}

void test_kv_cache_option() {
  // Metal stores standard attention's keys and values as FP16 unless asked
  // for FP32; the CPU keeps FP32 unless asked; latent attention keeps FP32.
  const DecoderConfig dense = DecoderConfig::from_json(Json::parse(kQwen38_27BConfig));
  DecoderConfig latent = dense;
  latent.kv_lora_rank = 512;
  LoadOptions o;
  EXPECT(kv_cache_f16(o, dense, true) && !kv_cache_f16(o, dense, false) && !kv_cache_f16(o, latent, true));
  o.kv_cache = "f32";
  EXPECT(!kv_cache_f16(o, dense, true));
  o.kv_cache = "f16";
  EXPECT(kv_cache_f16(o, dense, false) && throws([&] { kv_cache_f16(o, latent, true); }));
  o.kv_cache = "bf16";
  EXPECT(throws([&] { kv_cache_f16(o, dense, true); }));
  // The profiler's attention traffic: q in and out in float32, and every
  // visible key and value at the cache's width. Qwen3.8's layer 3 (24 heads,
  // 4 KV heads of 256) decoding at position 99 sees 100 keys.
  const double f32 = traffic::attention(dense, 3, 99, 1), f16 = traffic::attention(dense, 3, 99, 1, 2.0);
  EXPECT(f32 == 256.0 * (4.0 * 2 * 24 + 4.0 * 2 * 100 * 4) && f16 == 256.0 * (4.0 * 2 * 24 + 2.0 * 2 * 100 * 4));
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
  test_gemma3_config();
  test_gemma4_config();
  test_gpt_oss_config();
  test_glm_config();
  test_deepseek_configs();
  test_qwen3_5_configs();
  test_kv_cache_option();
  test_mrope_injection_plan();
  test_qwen3_vl_config();
  test_qwen2_5_vl_config();
  test_qwen2_vl_config();
  test_qwen2_5_vl_window_order();
  test_qwen3_5_vision_config();
  if (g_failures) {
    std::fprintf(stderr, "%d expectation(s) failed\n", g_failures);
    return 1;
  }
  std::printf("lokahi_engine_tests: all passed\n");
  return 0;
}
