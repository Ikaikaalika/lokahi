// lokahi: command-line driver for the native engine.
//
//   lokahi info       --model DIR [--backend B]
//   lokahi logits     --model DIR --tokens 2,5,9 [--prefill N] --out FILE
//   lokahi generate   --model DIR (--tokens 2,5,9 | --prompt TEXT) --max-new N [--no-eos-stop]
//                     [--temperature T] [--top-p P] [--top-k K] [--min-p M] [--seed S]
//   lokahi sample     --logits-file F.f32 --vocab V [sampling flags] [--steps N] [--first-step S]
//                     [--backend cpu|metal]
//   lokahi bench      --model DIR (--tokens ... | --tokens-file F) --max-new N
//                     [--warmups W] [--repetitions R]
//   lokahi profile    --model DIR (--tokens ... | --tokens-file F | --prompt TEXT)
//                     [--max-new N] [--warmups W]
//   lokahi tokenize   --model DIR (--text TEXT | --texts-file F.json) [--no-special]
//   lokahi detokenize --model DIR (--tokens 1,2 | --ids-file F.json) [--keep-special]
//                     [--stream [--stops-file S.json]]
//   lokahi pretokenize --model DIR (--text TEXT | --texts-file F.json)
//   lokahi run        --model DIR --prompt TEXT [--max-new N]
//
// `run` streams generated text to stdout; every other command prints one
// JSON object on stdout.
#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "model.h"
#include "json.h"
#include "safetensors.h"
#include "sampling.h"
#include "text_stream.h"
#include "tokenizer.h"

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
  return options;
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

int cmd_info(const Args& args) {
  auto model = load_model(args.get("model"), load_options(args));
  std::printf(
      "{\"architecture\":\"%s\",\"backend\":\"%s\",\"vocab_size\":%d,\"max_context\":%d,"
      "\"weight_bytes\":%zu}\n",
      model->architecture(), model->backend(), model->vocab_size(), model->max_context(),
      model->weight_bytes());
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
  auto generated = model->generate(tokens, args.get_int("max-new", 16), sampling_options(args),
                                   !args.has("no-eos-stop"), &timing);
  std::string text;
  if (args.has("prompt")) text = ",\"text\":" + json_string(load_tokenizer(args).decode(generated, true));
  std::printf("{\"backend\":\"%s\",\"tokens\":%s%s,\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f}\n",
              model->backend(), json_tokens(generated).c_str(), text.c_str(), timing.prefill_seconds,
              timing.decode_seconds);
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
  size_t emitted = 0;
  auto emit = [&](const std::string& text) {
    LK_CHECK(text.size() >= emitted, "streaming decode went backwards");
    std::fwrite(text.data() + emitted, 1, text.size() - emitted, stdout);
    std::fflush(stdout);
    emitted = text.size();
  };
  auto on_token = [&](int32_t token) {
    generated.push_back(token);
    const size_t stable = tokenizer.stable_prefix(generated, true);
    emit(tokenizer.decode(std::vector<int32_t>(generated.begin(), generated.begin() + static_cast<long>(stable)), true));
    return true;
  };
  GenerationTiming timing;
  model->generate_greedy(prompt, args.get_int("max-new", 256), true, &timing, on_token);
  emit(tokenizer.decode(generated, true));
  std::printf("\n");
  const double decode_rate = generated.size() > 1 ? (generated.size() - 1) / timing.decode_seconds : 0.0;
  std::fprintf(stderr, "[%s] prompt %zu tokens in %.1f ms; %zu tokens at %.1f tok/s\n", model->backend(),
               prompt.size(), 1000.0 * timing.prefill_seconds, generated.size(), decode_rate);
  return 0;
}

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

  std::ostringstream reps;
  reps << "[";
  bool invariant = true;
  std::vector<int32_t> reference;
  for (int run = 0; run < warmups + repetitions; ++run) {
    model->reset();
    GenerationTiming timing;
    auto tokens = model->generate_greedy(prompt, max_new, false, &timing);
    LK_CHECK(static_cast<int>(tokens.size()) == max_new, "generation stopped early");
    if (run < warmups) continue;
    if (reference.empty()) reference = tokens;
    invariant = invariant && tokens == reference;
    std::vector<double> intervals;
    for (size_t i = 1; i < timing.token_times.size(); ++i) {
      intervals.push_back(1000.0 * (timing.token_times[i] - timing.token_times[i - 1]));
    }
    const double ttft_ms = 1000.0 * timing.prefill_seconds;
    const double total = timing.prefill_seconds + timing.decode_seconds;
    reps << (run > warmups ? "," : "") << "{\"ttft_ms\":" << ttft_ms
         << ",\"inter_token_latency_ms\":" << median(intervals)
         << ",\"prompt_tokens_per_second\":" << prompt.size() / timing.prefill_seconds
         << ",\"decode_tokens_per_second\":" << (max_new - 1) / timing.decode_seconds
         << ",\"aggregate_tokens_per_second_including_prefill\":" << max_new / total
         << ",\"peak_memory_gb\":" << peak_rss_bytes() / 1e9
         << ",\"token_ids\":" << json_tokens(tokens) << "}";
  }
  reps << "]";
  std::printf(
      "{\"runner\":\"lokahi-native-%s\",\"backend\":\"%s\",\"architecture\":\"%s\","
      "\"load_seconds\":%.6f,\"weight_bytes\":%zu,\"prompt_tokens\":%zu,\"output_tokens\":%d,"
      "\"warmups\":%d,\"repetition_token_invariance\":%s,\"peak_memory_semantics\":"
      "\"process peak resident set size\",\"repetitions\":%s}\n",
      model->backend(), model->backend(), model->architecture(), load_seconds, model->weight_bytes(),
      prompt.size(), max_new, warmups, invariant ? "true" : "false", reps.str().c_str());
  return invariant ? 0 : 1;
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

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    LK_CHECK(args.has("model") || args.command == "sample", "--model is required");
    if (args.command == "tokenize") return cmd_tokenize(args);
    if (args.command == "detokenize") return cmd_detokenize(args);
    if (args.command == "pretokenize") return cmd_pretokenize(args);
    if (args.command == "run") return cmd_run(args);
    if (args.command == "info") return cmd_info(args);
    if (args.command == "logits") return cmd_logits(args);
    if (args.command == "generate") return cmd_generate(args);
    if (args.command == "sample") return cmd_sample(args);
    if (args.command == "bench") return cmd_bench(args);
    if (args.command == "profile") return cmd_profile(args);
    fail("unknown command " + args.command);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 2;
  }
}
