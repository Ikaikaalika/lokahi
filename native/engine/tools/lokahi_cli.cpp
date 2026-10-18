// lokahi: command-line driver for the native engine.
//
//   lokahi info       --model DIR [--backend B]
//   lokahi logits     --model DIR --tokens 2,5,9 [--prefill N] [--images A.rgb,B.rgb] --out FILE
//   lokahi generate   --model DIR (--tokens 2,5,9 | --prompt TEXT) --max-new N [--no-eos-stop]
//                     [--images A.rgb,B.rgb]
//                     [--temperature T] [--top-p P] [--top-k K] [--min-p M] [--seed S]
//                     [--speculate N [--draft-reference FILE] [--drafter lookup|mtp|both]]
//                                       # greedy: verify up to N drafts a pass
//                     [--mtp-probe K [--mtp-logits FILE]]   # the MTP head's next K drafts
//   (every command)   [--draft-model DIR]   # a qwen3_5_mtp drafter for --drafter mtp
//   lokahi image-features --model DIR --images A.ppm[,B.ppm] --out FILE [--backend B]
//   lokahi image-geometry --model DIR --width W --height H   # the processor's resize target
//   lokahi sample     --logits-file F.f32 --vocab V [sampling flags] [--steps N] [--first-step S]
//                     [--backend cpu|metal]
//   lokahi bench      --model DIR (--tokens ... | --tokens-file F) --max-new N
//                     [--warmups W] [--repetitions R] [--images A.ppm,B.ppm]
//                     [--speculate N [--draft-reference FILE] [--drafter lookup|mtp|both]]
//   lokahi profile    --model DIR (--tokens ... | --tokens-file F | --prompt TEXT)
//                     [--max-new N] [--warmups W]
//   lokahi tokenize   --model DIR (--text TEXT | --texts-file F.json) [--no-special]
//   lokahi detokenize --model DIR (--tokens 1,2 | --ids-file F.json) [--keep-special]
//                     [--stream [--stops-file S.json]]
//   lokahi pretokenize --model DIR (--text TEXT | --texts-file F.json)
//   lokahi run        --model DIR --prompt TEXT [--max-new N] [sampling flags]
//   lokahi serve      --config SERVICE.json      # Lōkahi wire v1 as NDJSON on stdin/stdout
//   lokahi chat-template (--model DIR | --template-file F [--config-file C]) --messages-file M.json [...] (raw text)
//   lokahi wire-check [--roundtrip | --encode | --stream REQUEST_ID] (FILE... | --stdin)
//
// `run` streams generated text to stdout and `serve` writes one wire
// document per line; every other command prints one JSON object on stdout.
//
// --images takes binary PPM (P6) files already resized to a size the model
// takes (image-geometry reports the processor's choice), or for models with
// one square size, raw RGB8 bytes of it; image i fills the prompt's i-th run
// of image tokens. --vision-attention-rows N bounds the Qwen towers' Metal
// attention scores to N query rows at a time (by default, 128 MiB of them).
#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "chat_template.h"
#include "common.h"
#include "model.h"
#include "json.h"
#include "safetensors.h"
#include "sampling.h"
#include "service.h"
#include "text_stream.h"
#include "tokenizer.h"
#include "vision.h"
#include "wire.h"

#if defined(LOKAHI_WITH_METAL)
#include "metal/metal_decoder.h"
#endif

using namespace lokahi;

namespace {

struct Args {
  std::string command;
  std::map<std::string, std::string> values;
  bool has(const std::string& key) const { return values.count(key) != 0; }
  std::string get(const std::string& key, const std::string& fallback = "") const {
    auto it = values.find(key);
    return it == values.end() ? fallback : it->second;
  }
  int get_int(const std::string& key, int fallback) const {
    return has(key) ? std::atoi(get(key).c_str()) : fallback;
  }
};

Args parse_args(int argc, char** argv) {
  Args args;
  LK_CHECK(argc >= 2,
           "usage: lokahi <info|logits|generate|bench|profile|tokenize|detokenize|pretokenize|run> --model DIR ...");
  args.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string key = argv[i];
    LK_CHECK(key.rfind("--", 0) == 0, "unexpected argument " + key);
    key = key.substr(2);
    if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
      args.values[key] = argv[++i];
    } else {
      args.values[key] = "1";
    }
  }
  return args;
}

std::vector<int32_t> parse_tokens(const std::string& text) {
  std::vector<int32_t> tokens;
  std::string item;
  for (char c : text + ",") {
    if (c == ',' || c == ' ' || c == '\n' || c == '\t' || c == '[' || c == ']') {
      if (!item.empty()) tokens.push_back(static_cast<int32_t>(std::stol(item)));
      item.clear();
    } else {
      item.push_back(c);
    }
  }
  return tokens;
}

Tokenizer load_tokenizer(const Args& args);

std::vector<int32_t> prompt_tokens(const Args& args) {
  if (args.has("prompt")) return load_tokenizer(args).encode(args.get("prompt"), true);
  if (args.has("tokens-file")) return parse_tokens(read_text_file(args.get("tokens-file")));
  LK_CHECK(args.has("tokens"), "--tokens or --tokens-file is required");
  return parse_tokens(args.get("tokens"));
}

LoadOptions load_options(const Args& args) {
  LoadOptions options;
  options.backend = args.get("backend", "auto");
  options.max_context = args.get_int("max-context", 4096);
  options.prefill_chunk = args.get_int("prefill-chunk", 512);
  options.cpu_threads = static_cast<unsigned>(args.get_int("threads", 0));
  options.kv_cache = args.get("kv-cache", "auto");
  options.draft_model = args.get("draft-model", "");
  return options;
}

// --speculate N drafts up to N tokens per verify pass from the prompt and
// the output so far, and from --draft-reference's tokens when given.
// --drafter mtp drafts with the model's multi-token prediction head (in the
// checkpoint, or --draft-model's qwen3_5_mtp drafter); both prefers a lookup
// draft when there is one.
SpeculationOptions speculation_options(const Args& args) {
  SpeculationOptions speculation;
  speculation.max_draft = args.get_int("speculate", 0);
  LK_CHECK(speculation.max_draft >= 0, "--speculate must not be negative");
  if (args.has("draft-reference")) speculation.reference = parse_tokens(read_text_file(args.get("draft-reference")));
  speculation.drafter = parse_drafter(args.get("drafter", "lookup"));
  return speculation;
}

SamplingParams sampling_options(const Args& args) {
  SamplingParams params;
  params.temperature = std::stof(args.get("temperature", "0"));
  params.top_p = std::stof(args.get("top-p", "1"));
  params.top_k = args.get_int("top-k", 0);
  params.min_p = std::stof(args.get("min-p", "0"));
  params.seed = std::stoull(args.get("seed", "0"));
  validate_sampling(params);
  return params;
}

std::string json_string(const std::string& text) {
  std::string out = "\"";
  for (unsigned char c : text) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
          out += buffer;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  return out + "\"";
}

Tokenizer load_tokenizer(const Args& args) {
  return Tokenizer::load(args.get("model") + "/tokenizer.json");
}

std::string json_tokens(const std::vector<int32_t>& tokens) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < tokens.size(); ++i) out << (i ? "," : "") << tokens[i];
  out << "]";
  return out.str();
}

double peak_rss_bytes() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
  return static_cast<double>(usage.ru_maxrss);  // bytes
#else
  return static_cast<double>(usage.ru_maxrss) * 1024.0;  // kilobytes
#endif
}

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  size_t mid = values.size() / 2;
  return values.size() % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

std::vector<std::string> split_list(const std::string& text) {
  std::vector<std::string> items;
  std::stringstream in(text);
  for (std::string item; std::getline(in, item, ',');) {
    if (!item.empty()) items.push_back(item);
  }
  return items;
}

// Each --images file through the vision tower, in order, with the token
// grids the decoder's positions follow (models with M-RoPE only).
struct EncodedImages {
  std::vector<ImageFeatures> features;
  std::vector<std::pair<int, int>> grids;
};

std::vector<Image> read_images(const Args& args, const VisionEncoder& encoder) {
  std::vector<Image> images;
  for (const std::string& path : split_list(args.get("images"))) images.push_back(read_image(path, encoder.fixed_size()));
  LK_CHECK(!images.empty(), "--images names no files");
  return images;
}

EncodedImages encode_images(const std::vector<Image>& images, VisionEncoder& encoder) {
  EncodedImages out;
  for (const Image& image : images) {
    out.features.push_back(encoder.encode_features(image));
    const auto grid = encoder.token_grid(image.width, image.height);
    if (grid.first > 0) out.grids.push_back(grid);
  }
  return out;
}

EncodedImages encode_images(const Args& args, VisionEncoder& encoder) {
  return encode_images(read_images(args, encoder), encoder);
}

// The vision encoder on the same backend as the decoder (--backend), with at
// most --vision-attention-rows query rows per attention pass (0: a budget).
std::unique_ptr<VisionEncoder> load_vision(const Args& args) {
  auto encoder = VisionEncoder::load(args.get("model"), args.get("backend", "auto"),
                                     static_cast<unsigned>(args.get_int("threads", 0)));
  LK_CHECK(encoder != nullptr, "this model has no vision tower");
  const int rows = args.get_int("vision-attention-rows", 0);
  LK_CHECK(rows >= 0, "--vision-attention-rows must not be negative");
  encoder->set_attention_rows(rows);
  return encoder;
}

// With --images, hands the next prefill of `tokens` their features.
void inject_images(const Args& args, Model& model, const std::vector<int32_t>& tokens) {
  if (!args.has("images")) return;
  auto encoder = load_vision(args);
  const EncodedImages images = encode_images(args, *encoder);
  model.inject(place_images(tokens, model.position(), encoder->image_token_id(), encoder->text_hidden_size(),
                            images.features, images.grids));
}

// Writes every image's rows, then (Qwen3-VL) each deepstack layer's rows for
// every image, as float32.
int cmd_image_features(const Args& args) {
  LK_CHECK(args.has("images") && args.has("out"), "image-features needs --images and --out");
  auto encoder = load_vision(args);
  const auto features = encode_images(args, *encoder).features;
  std::ofstream out(args.get("out"), std::ios::binary);
  LK_CHECK(out.good(), "cannot write " + args.get("out"));
  auto write = [&](const std::vector<float>& rows) {
    out.write(reinterpret_cast<const char*>(rows.data()), static_cast<std::streamsize>(sizeof(float) * rows.size()));
  };
  std::string tokens;
  for (const ImageFeatures& image : features) {
    write(image.rows);
    tokens += (tokens.empty() ? "" : ",") +
              std::to_string(image.rows.size() / static_cast<size_t>(encoder->text_hidden_size()));
  }
  const size_t layers = features.front().deepstack.size();
  for (size_t l = 0; l < layers; ++l) {
    for (const ImageFeatures& image : features) write(image.deepstack[l]);
  }
  const std::string deepstack = layers > 0 ? ",\"deepstack\":" + std::to_string(layers) : "";
  std::printf("{\"backend\":\"%s\",\"tokens\":[%s],\"width\":%d%s}\n", encoder->backend(), tokens.c_str(),
              encoder->text_hidden_size(), deepstack.c_str());
  return 0;
}

// The size the model's processor resizes a --width x --height image to, and
// the image tokens it then takes.
int cmd_image_geometry(const Args& args) {
  LK_CHECK(args.has("width") && args.has("height"), "image-geometry needs --width and --height");
  auto encoder = VisionEncoder::load(args.get("model"), "cpu");
  LK_CHECK(encoder != nullptr, "this model has no vision tower");
  const auto [width, height] = encoder->target_size(args.get_int("width", 0), args.get_int("height", 0));
  std::printf("{\"width\":%d,\"height\":%d,\"tokens\":%d,\"image_token_id\":%d}\n", width, height,
              encoder->tokens_for(width, height), encoder->image_token_id());
  return 0;
}

int cmd_info(const Args& args) {
  auto model = load_model(args.get("model"), load_options(args));
  std::string vision;
  const Json config = Json::parse(read_text_file(args.get("model") + "/config.json"));
  if (has_vision(config) && !vision_supported(config)) {
    // The text model runs; asking for images fails closed.
    vision = ",\"vision\":{\"supported\":false,\"model_type\":\"" +
             config.at("vision_config").string_or("model_type", "") + "\"}";
  } else if (auto encoder = VisionEncoder::load(args.get("model"), "cpu")) {
    // image_size 0: images take any accepted size (image-geometry).
    vision = ",\"vision\":{\"image_size\":" + std::to_string(encoder->fixed_size()) +
             ",\"image_token_id\":" + std::to_string(encoder->image_token_id()) + "}";
  }
  std::printf(
      "{\"architecture\":\"%s\",\"backend\":\"%s\",\"device\":%s,\"vocab_size\":%d,\"max_context\":%d,"
      "\"kv_cache\":\"%s\",\"weight_bytes\":%zu,\"mtp\":%s%s}\n",
      model->architecture(), model->backend(), json_string(model->device()).c_str(), model->vocab_size(),
      model->max_context(), model->kv_cache(), model->weight_bytes(), model->has_mtp() ? "true" : "false",
      vision.c_str());
  return 0;
}

int cmd_logits(const Args& args) {
  auto model = load_model(args.get("model"), load_options(args));
  std::vector<int32_t> tokens = prompt_tokens(args);
  int prefill = args.get_int("prefill", static_cast<int>(tokens.size()));
  LK_CHECK(prefill >= 1 && prefill <= static_cast<int>(tokens.size()), "--prefill out of range");
  LK_CHECK(args.has("out"), "--out is required");
  std::ofstream out(args.get("out"), std::ios::binary);
  LK_CHECK(out.good(), "cannot write " + args.get("out"));
  const int vocab = model->vocab_size();
  int rows = 0;
  inject_images(args, *model, std::vector<int32_t>(tokens.begin(), tokens.begin() + prefill));
  const float* logits = model->prefill(tokens.data(), prefill);
  out.write(reinterpret_cast<const char*>(logits), sizeof(float) * vocab);
  ++rows;
  for (size_t i = static_cast<size_t>(prefill); i < tokens.size(); ++i) {
    logits = model->decode(tokens[i]);
    out.write(reinterpret_cast<const char*>(logits), sizeof(float) * vocab);
    ++rows;
  }
  std::printf("{\"backend\":\"%s\",\"rows\":%d,\"vocab_size\":%d,\"prefill\":%d}\n", model->backend(),
              rows, vocab, prefill);
  return 0;
}

int cmd_generate(const Args& args) {
  auto model = load_model(args.get("model"), load_options(args));
  std::vector<int32_t> tokens = prompt_tokens(args);
  GenerationTiming timing;
  inject_images(args, *model, tokens);
  model->set_speculation(speculation_options(args));
  auto generated = model->generate(tokens, args.get_int("max-new", 16), sampling_options(args),
                                   !args.has("no-eos-stop"), &timing);
  std::string text;
  if (args.has("prompt")) text = ",\"text\":" + json_string(load_tokenizer(args).decode(generated, true));
  // --continue N decodes N more tokens greedily after the generation, from its
  // last token: after a stop, the model must resume from the tokens it kept
  // (Metal discards the decode steps it launched past the stop).
  std::string continued;
  if (args.has("continue") && !generated.empty()) {
    std::vector<int32_t> more;
    int32_t token = generated.back();
    for (int i = 0; i < args.get_int("continue", 0); ++i) {
      token = argmax(model->decode(token), model->vocab_size());
      more.push_back(token);
    }
    continued = ",\"continued\":" + json_tokens(more);
  }
  // --mtp-probe K reports the K drafts the MTP head makes after the last
  // token, and --mtp-logits FILE writes their [K, vocab] logits.
  std::string probe;
  if (args.has("mtp-probe") && !generated.empty()) {
    const int32_t last = continued.empty() ? generated.back() : -1;
    LK_CHECK(last >= 0, "--mtp-probe does not combine with --continue");
    std::vector<float> logits;
    const std::vector<int32_t> drafts = model->mtp_draft(last, args.get_int("mtp-probe", 0), &logits);
    if (args.has("mtp-logits")) {
      std::ofstream out(args.get("mtp-logits"), std::ios::binary);
      LK_CHECK(out.good(), "cannot write " + args.get("mtp-logits"));
      out.write(reinterpret_cast<const char*>(logits.data()),
                static_cast<std::streamsize>(sizeof(float) * logits.size()));
    }
    probe = ",\"mtp_drafts\":" + json_tokens(drafts);
  }
  const SpeculationStats& stats = model->speculation_stats();
  const std::string speculated = ",\"speculation\":{\"passes\":" + std::to_string(stats.passes) +
                                 ",\"drafted\":" + std::to_string(stats.drafted) +
                                 ",\"accepted\":" + std::to_string(stats.accepted) + "}";
  std::printf("{\"backend\":\"%s\",\"tokens\":%s%s%s%s%s,\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f}\n",
              model->backend(), json_tokens(generated).c_str(), text.c_str(), continued.c_str(), probe.c_str(),
              speculated.c_str(), timing.prefill_seconds, timing.decode_seconds);
  return 0;
}

// The host sampler over rows of float32 logits: tokens[row][step] for steps
// [first-step, first-step + steps).
int cmd_sample(const Args& args) {
  const int vocab = args.get_int("vocab", 0);
  LK_CHECK(vocab > 0, "sample needs --vocab");
  std::ifstream in(args.get("logits-file"), std::ios::binary | std::ios::ate);
  LK_CHECK(in.good(), "cannot read --logits-file");
  const auto bytes = static_cast<size_t>(in.tellg());
  LK_CHECK(bytes > 0 && bytes % (sizeof(float) * static_cast<size_t>(vocab)) == 0,
           "--logits-file size is not a whole number of rows");
  std::vector<float> logits(bytes / sizeof(float));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(logits.data()), static_cast<std::streamsize>(bytes));
  const SamplingParams params = sampling_options(args);
  const int steps = args.get_int("steps", 1);
  const int first = args.get_int("first-step", 0);
  const std::string backend = args.get("backend", "cpu");
  std::vector<int32_t> drawn;
#if defined(LOKAHI_WITH_METAL)
  if (backend == "metal") drawn = metal_sample_rows(logits, vocab, params, first, steps);
#endif
  if (drawn.empty()) {
    LK_CHECK(backend == "cpu", "sample: backend '" + backend + "' is not available");
    for (size_t row = 0; row * static_cast<size_t>(vocab) < logits.size(); ++row) {
      for (int step = first; step < first + steps; ++step) {
        drawn.push_back(sample_token(logits.data() + row * static_cast<size_t>(vocab), vocab, params, step));
      }
    }
  }
  std::string rows;
  for (size_t begin = 0; begin < drawn.size(); begin += static_cast<size_t>(steps)) {
    std::vector<int32_t> tokens(drawn.begin() + static_cast<std::ptrdiff_t>(begin),
                                drawn.begin() + static_cast<std::ptrdiff_t>(begin) + steps);
    rows += (begin ? "," : "") + json_tokens(tokens);
  }
  std::printf("{\"tokens\":[%s]}\n", rows.c_str());
  return 0;
}

int cmd_tokenize(const Args& args) {
  Tokenizer tokenizer = load_tokenizer(args);
  const bool special = !args.has("no-special");
  if (args.has("texts-file")) {
    Json texts = Json::parse(read_text_file(args.get("texts-file")));
    std::string out = "{\"ids\":[";
    bool first = true;
    for (const Json& text : texts.as_array()) {
      out += (first ? "" : ",") + json_tokens(tokenizer.encode(text.as_string(), special));
      first = false;
    }
    std::printf("%s]}\n", out.c_str());
    return 0;
  }
  LK_CHECK(args.has("text"), "--text or --texts-file is required");
  std::printf("{\"ids\":%s}\n", json_tokens(tokenizer.encode(args.get("text"), special)).c_str());
  return 0;
}

// Each id list through a TextStream: the text every push returned, the
// finish text last, and the token after which a stop string ended it.
std::string stream_texts(const Tokenizer& tokenizer, const Json& lists, const Args& args) {
  std::vector<std::string> stops;
  if (args.has("stops-file")) {
    const Json list = Json::parse(read_text_file(args.get("stops-file")));
    for (const Json& stop : list.as_array()) stops.push_back(stop.as_string());
  }
  std::string out = "{\"streams\":[";
  for (size_t i = 0; i < lists.as_array().size(); ++i) {
    TextStream stream(tokenizer, stops, !args.has("keep-special"));
    std::string deltas;
    long stopped_after = -1;
    const auto& list = lists.as_array()[i].as_array();
    for (size_t k = 0; k < list.size(); ++k) {
      deltas += (k ? "," : "") + json_string(stream.push(static_cast<int32_t>(list[k].as_int())));
      if (stream.stopped() && stopped_after < 0) stopped_after = static_cast<long>(k);
    }
    deltas += (list.empty() ? "" : ",") + json_string(stream.finish());
    out += (i ? "," : "") + std::string("{\"deltas\":[") + deltas + "],\"stopped_after\":" +
           std::to_string(stopped_after) + ",\"text\":" + json_string(stream.text()) + "}";
  }
  return out + "]}";
}

int cmd_detokenize(const Args& args) {
  Tokenizer tokenizer = load_tokenizer(args);
  const bool skip = !args.has("keep-special");
  if (args.has("stream")) {
    LK_CHECK(args.has("ids-file"), "--stream needs --ids-file");
    std::printf("%s\n", stream_texts(tokenizer, Json::parse(read_text_file(args.get("ids-file"))), args).c_str());
    return 0;
  }
  if (args.has("ids-file")) {
    Json lists = Json::parse(read_text_file(args.get("ids-file")));
    std::string out = "{\"texts\":[";
    bool first = true;
    for (const Json& list : lists.as_array()) {
      std::vector<int32_t> ids;
      for (const Json& id : list.as_array()) ids.push_back(static_cast<int32_t>(id.as_int()));
      out += (first ? "" : ",") + json_string(tokenizer.decode(ids, skip));
      first = false;
    }
    std::printf("%s]}\n", out.c_str());
    return 0;
  }
  std::printf("{\"text\":%s}\n", json_string(tokenizer.decode(parse_tokens(args.get("tokens")), skip)).c_str());
  return 0;
}

// The words BPE runs on (normalizer + pre-tokenizer), for diagnostics.
int cmd_pretokenize(const Args& args) {
  Tokenizer tokenizer = load_tokenizer(args);
  auto words_json = [&](const std::string& text) {
    std::string out = "[";
    for (const std::string& word : tokenizer.words(text)) out += (out.size() > 1 ? "," : "") + json_string(word);
    return out + "]";
  };
  if (args.has("texts-file")) {
    Json texts = Json::parse(read_text_file(args.get("texts-file")));
    std::string out = "{\"words\":[";
    bool first = true;
    for (const Json& text : texts.as_array()) {
      out += (first ? "" : ",") + words_json(text.as_string());
      first = false;
    }
    std::printf("%s]}\n", out.c_str());
    return 0;
  }
  LK_CHECK(args.has("text"), "--text or --texts-file is required");
  std::printf("{\"words\":%s}\n", words_json(args.get("text")).c_str());
  return 0;
}

// Streams decoded text while generating. Only text that later tokens cannot
// change is written, so the stream always equals the final decode.
int cmd_run(const Args& args) {
  LK_CHECK(args.has("prompt"), "--prompt is required");
  Tokenizer tokenizer = load_tokenizer(args);
  auto model = load_model(args.get("model"), load_options(args));
  std::vector<int32_t> prompt = tokenizer.encode(args.get("prompt"), true);
  std::vector<int32_t> generated;
  TextStream stream(tokenizer, {});
  auto emit = [](const std::string& text) {
    std::fwrite(text.data(), 1, text.size(), stdout);
    std::fflush(stdout);
  };
  auto on_token = [&](int32_t token) {
    generated.push_back(token);
    emit(stream.push(token));
    return true;
  };
  GenerationTiming timing;
  model->generate(prompt, args.get_int("max-new", 256), sampling_options(args), true, &timing, on_token);
  emit(stream.finish());
  std::printf("\n");
  const double decode_rate = generated.size() > 1 ? (generated.size() - 1) / timing.decode_seconds : 0.0;
  std::fprintf(stderr, "[%s] prompt %zu tokens in %.1f ms; %zu tokens at %.1f tok/s\n", model->backend(),
               prompt.size(), 1000.0 * timing.prefill_seconds, generated.size(), decode_rate);
  return 0;
}

// Lōkahi wire v1 over stdin/stdout: each input line is a start or control
// document, and each event or receipt is written as one line. Invalid input
// is reported on stderr as {"error": ...} and answered with nothing, like a
// failed XPC call. Exits once stdin closes and every admitted request has
// its receipt.
int cmd_serve(const Args& args) {
  Service service(ServiceOptions::from_json(Json::parse(read_text_file(args.get("config")))));
  std::mutex out;
  auto sink = [&out](const std::string& envelope) {
    std::lock_guard<std::mutex> lock(out);
    std::fwrite(envelope.data(), 1, envelope.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
  };
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    try {
      const wire::MessageType type = wire::validate(line);
      if (type == wire::MessageType::Start) {
        service.start(line, sink);
      } else if (type == wire::MessageType::Control) {
        service.control(line);
      } else {
        fail("lokahi: serve accepts only start and control documents");
      }
    } catch (const std::exception& error) {
      std::lock_guard<std::mutex> lock(out);
      std::fprintf(stderr, "{\"error\":%s}\n", json_string(error.what()).c_str());
      std::fflush(stderr);
    }
  }
  service.wait_idle();
  return 0;
}

// With --speculate, an untimed plain greedy run first gives the tokens every
// repetition must reproduce: speculation may only change the speed.
int cmd_bench(const Args& args) {
  LoadOptions options = load_options(args);
  const double load_start = now_seconds();
  auto model = load_model(args.get("model"), options);
  const double load_seconds = now_seconds() - load_start;
  std::vector<int32_t> prompt = prompt_tokens(args);
  const int max_new = args.get_int("max-new", 128);
  const int warmups = args.get_int("warmups", 2);
  const int repetitions = args.get_int("repetitions", 5);
  LK_CHECK(max_new >= 2, "--max-new must be at least 2");
  SpeculationOptions speculation = speculation_options(args);
  const bool speculating = speculation.max_draft > 0;
  LK_CHECK(!speculating || model->can_verify(), std::string(model->backend()) + ": cannot verify drafts");
  std::vector<int32_t> plain;
  if (speculating) {
    model->reset();
    plain = model->generate_greedy(prompt, max_new, false, nullptr);
  }
  const std::string drafter = args.get("drafter", "lookup");
  const int max_draft = speculation.max_draft;
  model->set_speculation(std::move(speculation));
  // --images: every run encodes them through the vision tower and hands them
  // to its prefill; ttft_ms then counts the encoding (vision_ms) too.
  std::unique_ptr<VisionEncoder> encoder;
  std::vector<Image> images;
  if (args.has("images")) {
    LK_CHECK(!speculating, "bench takes --images or --speculate, not both");
    encoder = load_vision(args);
    images = read_images(args, *encoder);
  }

  std::ostringstream reps;
  reps << "[";
  bool invariant = true;
  bool matches_plain = true;
  std::vector<int32_t> reference;
  for (int run = 0; run < warmups + repetitions; ++run) {
    model->reset();
    double vision_seconds = 0.0;
    if (encoder) {
      const double start = now_seconds();
      const EncodedImages encoded = encode_images(images, *encoder);
      model->inject(place_images(prompt, model->position(), encoder->image_token_id(), encoder->text_hidden_size(),
                                 encoded.features, encoded.grids));
      vision_seconds = now_seconds() - start;
    }
    GenerationTiming timing;
    auto tokens = model->generate_greedy(prompt, max_new, false, &timing);
    LK_CHECK(static_cast<int>(tokens.size()) == max_new, "generation stopped early");
    if (run < warmups) continue;
    if (reference.empty()) reference = tokens;
    invariant = invariant && tokens == reference;
    matches_plain = matches_plain && (!speculating || tokens == plain);
    std::vector<double> intervals;
    for (size_t i = 1; i < timing.token_times.size(); ++i) {
      intervals.push_back(1000.0 * (timing.token_times[i] - timing.token_times[i - 1]));
    }
    // A verify pass emits its tokens together, so most intervals are zero;
    // the mean is the time per token there.
    const double inter_token_ms = speculating ? 1000.0 * timing.decode_seconds / (max_new - 1) : median(intervals);
    const double ttft_ms = 1000.0 * (vision_seconds + timing.prefill_seconds);
    const double total = vision_seconds + timing.prefill_seconds + timing.decode_seconds;
    const SpeculationStats& stats = model->speculation_stats();
    reps << (run > warmups ? "," : "") << "{\"ttft_ms\":" << ttft_ms
         << ",\"inter_token_latency_ms\":" << inter_token_ms
         << ",\"prompt_tokens_per_second\":" << prompt.size() / timing.prefill_seconds
         << ",\"decode_tokens_per_second\":" << (max_new - 1) / timing.decode_seconds
         << ",\"aggregate_tokens_per_second_including_prefill\":" << max_new / total
         << ",\"peak_memory_gb\":" << peak_rss_bytes() / 1e9;
    if (encoder) reps << ",\"vision_ms\":" << 1000.0 * vision_seconds;
    if (speculating) {
      reps << ",\"speculation\":{\"passes\":" << stats.passes << ",\"drafted\":" << stats.drafted
           << ",\"accepted\":" << stats.accepted << "}";
    }
    reps << ",\"token_ids\":" << json_tokens(tokens) << "}";
  }
  reps << "]";
  const std::string speculated =
      speculating ? "{\"max_draft\":" + std::to_string(max_draft) + ",\"drafter\":" + json_string(drafter) +
                        ",\"inter_token_latency\":\"mean\",\"matches_plain_greedy\":" +
                        (matches_plain ? "true" : "false") + "}"
                  : "null";
  std::printf(
      "{\"runner\":\"lokahi-native-%s\",\"backend\":\"%s\",\"device\":%s,\"architecture\":\"%s\","
      "\"load_seconds\":%.6f,\"weight_bytes\":%zu,\"prompt_tokens\":%zu,\"output_tokens\":%d,"
      "\"warmups\":%d,\"repetition_token_invariance\":%s,\"speculation\":%s,\"peak_memory_semantics\":"
      "\"process peak resident set size\",\"repetitions\":%s}\n",
      model->backend(), model->backend(), json_string(model->device()).c_str(), model->architecture(), load_seconds,
      model->weight_bytes(),
      prompt.size(), max_new, warmups, invariant ? "true" : "false", speculated.c_str(), reps.str().c_str());
  return invariant && matches_plain ? 0 : 1;
}

std::string json_sections(const Profiler& profiler) {
  std::ostringstream out;
  out << "[";
  bool first = true;
  for (const ProfileSection& section : profiler.sections()) {
    out << (first ? "" : ",") << "{\"name\":" << json_string(section.name) << ",\"seconds\":" << section.seconds
        << ",\"calls\":" << section.calls << ",\"bytes\":" << section.bytes << "}";
    first = false;
  }
  out << "]";
  return out.str();
}

// Attribution for the improvement loop: the time and memory traffic of every
// section of one prefill and of the decode steps after it. Sections run one
// command buffer at a time on Metal, so wall times here are not fast-path
// latencies; benchmark those with `bench`.
int cmd_profile(const Args& args) {
  auto model = load_model(args.get("model"), load_options(args));
  const std::vector<int32_t> prompt = prompt_tokens(args);
  const int steps = args.get_int("max-new", 32);
  const int warmups = args.get_int("warmups", 1);
  LK_CHECK(!prompt.empty() && steps >= 1 && warmups >= 0, "profile needs a prompt, --max-new >= 1, --warmups >= 0");
  for (int run = 0; run < warmups; ++run) {
    model->reset();
    model->generate_greedy(prompt, steps + 1, false, nullptr);
  }
  model->reset();

  Profiler prefill, decode;
  model->set_profiler(&prefill);
  double start = now_seconds();
  int32_t token = argmax(model->prefill(prompt.data(), static_cast<int>(prompt.size())), model->vocab_size());
  const double prefill_seconds = now_seconds() - start;
  std::vector<int32_t> tokens = {token};
  model->set_profiler(&decode);
  start = now_seconds();
  for (int i = 0; i < steps; ++i) {
    token = argmax(model->decode(token), model->vocab_size());
    tokens.push_back(token);
  }
  const double decode_seconds = now_seconds() - start;
  model->set_profiler(nullptr);

  std::ostringstream out;
  out.precision(9);
  out << "{\"backend\":\"" << model->backend() << "\",\"architecture\":\"" << model->architecture()
      << "\",\"weight_bytes\":" << model->weight_bytes() << ",\"prompt_tokens\":" << prompt.size()
      << ",\"decode_steps\":" << steps
      << ",\"timing_boundary\":\"section seconds are GPU time per command buffer on Metal and host time on the "
         "CPU; wall seconds include per-section synchronization\""
      << ",\"prefill\":{\"wall_seconds\":" << prefill_seconds << ",\"sections\":" << json_sections(prefill) << "}"
      << ",\"decode\":{\"wall_seconds\":" << decode_seconds << ",\"sections\":" << json_sections(decode) << "}"
      << ",\"tokens\":" << json_tokens(tokens) << "}";
  std::printf("%s\n", out.str().c_str());
  return 0;
}

// Checks wire v1 documents, one per FILE or per --stdin line, and prints one
// JSON verdict per document: {"file" or "line", "valid", "type", "error"}.
//   --roundtrip   decodes and re-encodes; a single FILE prints the encoding
//                 itself, --stdin adds it to each verdict as "encoded".
//   --stream ID   feeds the documents in order to one StreamValidator and adds
//                 "terminated" and "next_sequence".
//   --encode      reads each event or receipt into the structs without
//                 validation, so tests can show the encoders refuse bad values.
int cmd_wire_check(int argc, char** argv) {
  const std::string usage =
      "usage: lokahi wire-check [--roundtrip | --encode | --stream REQUEST_ID] (FILE... | --stdin)";
  enum class Mode { Check, Roundtrip, Encode, Stream } mode = Mode::Check;
  std::optional<wire::StreamValidator> stream;
  bool from_stdin = false;
  std::vector<std::string> files;
  for (int i = 0; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--stdin") {
      from_stdin = true;
    } else if (arg == "--roundtrip") {
      mode = Mode::Roundtrip;
    } else if (arg == "--encode") {
      mode = Mode::Encode;
    } else if (arg == "--stream" && i + 1 < argc) {
      mode = Mode::Stream;
      stream.emplace(argv[++i]);
    } else {
      LK_CHECK(arg.rfind("--", 0) != 0, usage);
      files.push_back(arg);
    }
  }
  LK_CHECK(from_stdin == files.empty(), usage);

  const auto reencode = [](const std::string& document) {
    const wire::MessageType type = wire::validate(document);
    if (type == wire::MessageType::Start) return wire::encode(wire::decode_start(document));
    if (type == wire::MessageType::Control) return wire::encode(wire::decode_control(document));
    if (type == wire::MessageType::Event) return wire::encode(wire::decode_event(document));
    return wire::encode(wire::decode_receipt(document));
  };
  if (mode == Mode::Roundtrip && !from_stdin) {
    LK_CHECK(files.size() == 1, "usage: lokahi wire-check --roundtrip FILE");
    std::printf("%s\n", reencode(read_text_file(files[0])).c_str());
    return 0;
  }

  const auto build_and_encode = [](const std::string& document) {
    const Json j = Json::parse(document);
    const auto text = [](const Json& o, const char* key) { return o.string_or(key, ""); };
    const auto maybe_text = [](const Json& o, const char* key) -> std::optional<std::string> {
      const Json* v = o.find(key);
      return v ? std::optional<std::string>(v->as_string()) : std::nullopt;
    };
    const auto maybe_int = [](const Json& o, const char* key) -> std::optional<int64_t> {
      const Json* v = o.find(key);
      return v ? std::optional<int64_t>(v->as_int()) : std::nullopt;
    };
    const auto maybe_number = [](const Json& o, const char* key) -> std::optional<double> {
      const Json* v = o.find(key);
      return v && !v->is_null() ? std::optional<double>(v->as_number()) : std::nullopt;
    };
    if (text(j, "message_type") == "event") {
      wire::Event e;
      e.request_id = text(j, "request_id");
      e.sequence = j.int_or("sequence", 0);
      e.kind = text(j, "kind");
      e.emitted_at_unix_ms = j.int_or("emitted_at_unix_ms", 0);
      e.progress = maybe_number(j, "progress");
      e.text_delta = maybe_text(j, "text_delta");
      e.generated_tokens_delta = maybe_int(j, "generated_tokens_delta");
      e.queue_position = maybe_int(j, "queue_position");
      e.queue_depth = maybe_int(j, "queue_depth");
      e.message = maybe_text(j, "message");
      return wire::encode(e);
    }
    wire::Receipt r;
    r.request_id = text(j, "request_id");
    r.terminal_sequence = j.int_or("terminal_sequence", 0);
    r.status = text(j, "status");
    r.finish_reason = text(j, "finish_reason");
    r.completed_at_unix_ms = j.int_or("completed_at_unix_ms", 0);
    r.output_text = maybe_text(j, "output_text");
    const Json& usage = j.at("usage");
    r.usage = {usage.int_or("prompt_tokens", 0), usage.int_or("completion_tokens", 0),
               usage.int_or("total_tokens", 0)};
    const Json& metrics = j.at("metrics");
    r.metrics.queue_ms = metrics.number_or("queue_ms", 0);
    r.metrics.ttft_ms = maybe_number(metrics, "ttft_ms");
    r.metrics.inter_token_latency_ms = maybe_number(metrics, "inter_token_latency_ms");
    r.metrics.decode_tokens_per_second = maybe_number(metrics, "decode_tokens_per_second");
    r.metrics.peak_memory_bytes = metrics.int_or("peak_memory_bytes", 0);
    r.metrics.maximum_batch_size = metrics.int_or("maximum_batch_size", 1);
    const Json& execution = j.at("execution");
    r.execution.backend = text(execution, "backend");
    r.execution.route = text(execution, "route");
    r.execution.plan_id = text(execution, "plan_id");
    for (const Json& id : execution.at("evidence_ids").as_array()) r.execution.evidence_ids.push_back(id.as_string());
    r.execution.fallback_used = execution.at("fallback_used").as_bool();
    r.execution.fallback_reason = maybe_text(execution, "fallback_reason");
    if (const Json* f = j.find("failure")) {
      r.failure = wire::Failure{text(*f, "code"), text(*f, "stage"), f->at("retryable").as_bool(), text(*f, "message")};
    }
    return wire::encode(r);
  };

  bool all_valid = true;
  const auto report = [&](const std::string& label, const std::string& document) {
    std::string type = "null", error = "null", encoded = "null";
    try {
      if (mode == Mode::Encode) {
        const std::string out = build_and_encode(document);
        type = json_string(std::string(wire::name(wire::validate(out))));
        encoded = json_string(out);
      } else {
        const wire::MessageType kind = wire::validate(document);
        type = json_string(std::string(wire::name(kind)));
        if (mode == Mode::Roundtrip) encoded = json_string(reencode(document));
        if (mode == Mode::Stream && kind == wire::MessageType::Event) stream->accept(wire::decode_event(document));
        if (mode == Mode::Stream && kind == wire::MessageType::Receipt) stream->accept(wire::decode_receipt(document));
        if (mode == Mode::Stream && kind != wire::MessageType::Event && kind != wire::MessageType::Receipt) {
          throw wire::ContractError("wire: a stream carries only events and receipts");
        }
      }
    } catch (const wire::ContractError& e) {
      error = json_string(e.what());
      all_valid = false;
    }
    std::string extra;
    if (mode == Mode::Roundtrip || mode == Mode::Encode) extra = ",\"encoded\":" + encoded;
    if (mode == Mode::Stream) {
      extra = std::string(",\"terminated\":") + (stream->terminated() ? "true" : "false") +
              ",\"next_sequence\":" + std::to_string(stream->next_sequence());
    }
    std::printf("{%s,\"valid\":%s,\"type\":%s,\"error\":%s%s}\n", label.c_str(), error == "null" ? "true" : "false",
                type.c_str(), error.c_str(), extra.c_str());
  };
  if (from_stdin) {
    std::ios::sync_with_stdio(false);
    std::string line;
    for (int64_t n = 1; std::getline(std::cin, line); ++n) report("\"line\":" + std::to_string(n), line);
  } else {
    for (const std::string& file : files) report("\"file\":" + json_string(file), read_text_file(file));
  }
  return all_valid ? 0 : 1;
}

// Renders a chat template as transformers' apply_chat_template(tokenize=False)
// does and writes the text unchanged. Extra variables come from --set k=v and
// --flag k=true|false (comma-separated lists, since each option is given once)
// and from --vars-file, a JSON object of strings and booleans for values that
// contain commas.
int cmd_chat_template(const Args& args) {
  ChatTemplateContext context;
  auto token = [](const Json& config, const char* key) {
    const Json* value = config.find(key);
    if (value && value->is_object()) value = value->find("content");
    return value && value->is_string() ? value->as_string() : std::string();
  };
  auto load = [&]() {
    if (args.has("model")) return load_chat_template(args.get("model"), &context);
    LK_CHECK(args.has("template-file"), "--model or --template-file is required");
    if (args.has("config-file")) {
      const Json config = Json::parse(read_text_file(args.get("config-file")));
      context.bos_token = token(config, "bos_token");
      context.eos_token = token(config, "eos_token");
    }
    return ChatTemplate::parse(read_text_file(args.get("template-file")));
  };
  const ChatTemplate chat = load();
  LK_CHECK(args.has("messages-file"), "--messages-file is required");
  std::vector<ChatMessage> messages;
  const Json list = Json::parse(read_text_file(args.get("messages-file")));
  for (const Json& m : list.as_array()) {
    messages.push_back({m.at("role").as_string(), m.at("content").as_string()});
  }
  context.add_generation_prompt = !args.has("no-generation-prompt");
  auto each = [](const std::string& spec, auto&& apply) {
    std::stringstream items(spec);
    for (std::string item; std::getline(items, item, ',');) {
      const size_t eq = item.find('=');
      LK_CHECK(eq != std::string::npos, "expected key=value, got " + item);
      apply(item.substr(0, eq), item.substr(eq + 1));
    }
  };
  if (args.has("set")) {
    each(args.get("set"), [&](const std::string& k, const std::string& v) { context.strings[k] = v; });
  }
  if (args.has("flag")) {
    each(args.get("flag"), [&](const std::string& k, const std::string& v) {
      LK_CHECK(v == "true" || v == "false", "--flag values are true or false, got " + v);
      context.flags[k] = v == "true";
    });
  }
  if (args.has("vars-file")) {
    const Json vars = Json::parse(read_text_file(args.get("vars-file")));
    for (const auto& [key, value] : vars.as_object()) {
      LK_CHECK(value.is_string() || value.is_bool(), "--vars-file values must be strings or booleans: " + key);
      if (value.is_string()) context.strings[key] = value.as_string();
      if (value.is_bool()) context.flags[key] = value.as_bool();
    }
  }
  const std::string text = chat.render(messages, context);
  std::fwrite(text.data(), 1, text.size(), stdout);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::strcmp(argv[1], "wire-check") == 0) return cmd_wire_check(argc - 2, argv + 2);
    Args args = parse_args(argc, argv);
    if (args.command == "chat-template") return cmd_chat_template(args);
    LK_CHECK(args.has("model") || args.command == "sample" || args.command == "serve", "--model is required");
    if (args.command == "tokenize") return cmd_tokenize(args);
    if (args.command == "detokenize") return cmd_detokenize(args);
    if (args.command == "pretokenize") return cmd_pretokenize(args);
    if (args.command == "run") return cmd_run(args);
    if (args.command == "info") return cmd_info(args);
    if (args.command == "image-features") return cmd_image_features(args);
    if (args.command == "image-geometry") return cmd_image_geometry(args);
    if (args.command == "logits") return cmd_logits(args);
    if (args.command == "generate") return cmd_generate(args);
    if (args.command == "sample") return cmd_sample(args);
    if (args.command == "serve") return cmd_serve(args);
    if (args.command == "bench") return cmd_bench(args);
    if (args.command == "profile") return cmd_profile(args);
    fail("unknown command " + args.command);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 2;
  }
}
