#include "service.h"

#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <utility>

#include "chat_template.h"
#include "common.h"
#include "json.h"
#include "model.h"
#include "text_stream.h"
#include "tokenizer.h"
#include "wire.h"

namespace lokahi {

namespace {

constexpr size_t kMaxDeltaBytes = 16384;  // well under the 65536 code points an event may carry
constexpr size_t kMaxMessageChars = 1000;

int64_t unix_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

int64_t peak_rss_bytes() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
  return static_cast<int64_t>(usage.ru_maxrss);  // bytes
#else
  return static_cast<int64_t>(usage.ru_maxrss) * 1024;  // kilobytes
#endif
}

// Failure messages are bounded by the contract; keep whole UTF-8 characters.
std::string bounded(std::string message) {
  if (message.empty()) message = "unspecified error";
  if (message.size() <= kMaxMessageChars) return message;
  size_t cut = kMaxMessageChars;
  while (cut > 0 && (static_cast<unsigned char>(message[cut]) & 0xC0) == 0x80) --cut;
  return message.substr(0, cut);
}

struct LoadedModel {
  LoadedModel(const ServiceModel& id, const LoadOptions& options)
      : identity(id),
        model(load_model(id.directory, options)),
        tokenizer(Tokenizer::load(id.directory + "/tokenizer.json")),
        chat_template(load_chat_template(id.directory, &context)) {}

  ServiceModel identity;
  std::unique_ptr<Model> model;
  Tokenizer tokenizer;
  ChatTemplateContext context;  // filled by load_chat_template, so declared before it
  ChatTemplate chat_template;
  std::vector<bool> busy;  // the batched worker's sequences in use (worker thread only)
};

struct Request {
  Request(wire::Start s, EnvelopeSink k) : start(std::move(s)), sink(std::move(k)), stream(start.request_id) {}

  wire::Start start;
  EnvelopeSink sink;
  wire::StreamValidator stream;
  LoadedModel* model = nullptr;
  std::vector<int32_t> prompt;
  int64_t next_sequence = 0;
  double admitted = 0.0;  // now_seconds()
  std::atomic<bool> cancelled{false};
  std::string sent_text;  // text delivered in events
};

struct Outcome {
  std::string status, finish_reason;
  std::optional<wire::Failure> failure;
  std::optional<std::string> output_text;
  int64_t completion_tokens = 0;
  wire::Metrics metrics;
  std::string backend, plan_id;
};

wire::Failure failure(std::string code, std::string stage, bool retryable, std::string message) {
  return wire::Failure{std::move(code), std::move(stage), retryable, bounded(std::move(message))};
}

}  // namespace

ServiceOptions ServiceOptions::from_json(const Json& config) {
  ServiceOptions options;
  LK_CHECK(config.is_object(), "service config must be an object");
  for (const Json& entry : config.at("models").as_array()) {
    ServiceModel model;
    model.id = entry.at("id").as_string();
    model.revision = entry.at("revision").as_string();
    model.artifact_digest = entry.at("artifact_digest").as_string();
    model.adapter_id = entry.string_or("adapter_id", "");
    model.directory = entry.at("directory").as_string();
    options.models.push_back(std::move(model));
  }
  LK_CHECK(!options.models.empty(), "service config lists no models");
  options.backend = config.string_or("backend", options.backend);
  options.max_context = static_cast<int>(config.int_or("max_context", options.max_context));
  options.kv_cache = config.string_or("kv_cache", options.kv_cache);
  options.queue_limit = static_cast<int>(config.int_or("queue_limit", options.queue_limit));
  options.max_sequences = static_cast<int>(config.int_or("max_sequences", options.max_sequences));
  options.text_flush_ms = config.number_or("text_flush_ms", options.text_flush_ms);
  options.runtime_revision = config.string_or("runtime_revision", "");
  LK_CHECK(options.queue_limit >= 0 && options.text_flush_ms >= 0.0, "service config has a negative limit");
  LK_CHECK(options.max_sequences >= 1, "service config max_sequences must be at least 1");
  return options;
}

struct Service::Impl {
  explicit Impl(ServiceOptions o) : options(std::move(o)) {
    LoadOptions load;
    load.backend = options.backend;
    load.max_context = options.max_context;
    load.kv_cache = options.kv_cache;
    load.max_sequences = options.max_sequences;
    for (const ServiceModel& m : options.models) {
      models.push_back(std::make_unique<LoadedModel>(m, load));
      models.back()->busy.assign(static_cast<size_t>(options.max_sequences), false);
    }
    worker = std::thread([this] { options.max_sequences > 1 ? work_batched() : work(); });
  }

  ~Impl() {
    std::deque<std::shared_ptr<Request>> queued;
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
      for (auto& r : running) r->cancelled = true;
      queued.swap(queue);
    }
    cv.notify_all();
    for (auto& r : queued) {
      finish(*r, cancelled_outcome(*r, "not_started", "the service is shutting down"));
      forget(*r);
    }
    worker.join();
  }

  // ---- admission ------------------------------------------------------------

  void start(std::string_view document, EnvelopeSink sink) {
    LK_CHECK(sink != nullptr, "service: start needs an event sink");
    auto r = std::make_shared<Request>(wire::decode_start(document), std::move(sink));
    r->admitted = now_seconds();
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!active.emplace(r->start.request_id, r).second) {
        throw wire::ContractError("lokahi: request_id " + r->start.request_id + " is already in flight");
      }
    }
    if (std::optional<wire::Failure> refusal = admit(*r)) {
      finish(*r, rejected_outcome(std::move(*refusal)));
      forget(*r);
      return;
    }
    // Events go out before the request is queued: once the worker can see
    // it, only the worker writes to its stream.
    wire::Event accepted;
    accepted.kind = "accepted";
    send(*r, std::move(accepted));
    int64_t ahead = 0;
    {
      // Requests ahead of this one for a sequence: none while one is free.
      std::lock_guard<std::mutex> lock(mutex);
      const int64_t waiting = static_cast<int64_t>(queue.size());
      ahead = waiting >= free_sequences() ? waiting - free_sequences() + 1 : 0;
    }
    if (ahead > 0) {
      wire::Event state;
      state.kind = "queue_state";
      state.queue_position = ahead;
      state.queue_depth = ahead + 1;
      send(*r, std::move(state));
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      queue.push_back(r);
    }
    cv.notify_all();
  }

  // Returns why the request cannot run, or nothing when it is admitted.
  std::optional<wire::Failure> admit(Request& r) {
    const wire::Start& s = r.start;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (draining || stopping) {
        return failure("service-draining", "not_started", true, "the runtime is draining and admits no new requests");
      }
      if (static_cast<int64_t>(queue.size()) >= options.queue_limit + free_sequences()) {
        return failure("overloaded", "not_started", true, "the request queue is full");
      }
    }
    if (!options.runtime_revision.empty() && s.runtime_revision != options.runtime_revision) {
      return failure("runtime-revision-mismatch", "not_started", true,
                     "this runtime is revision " + options.runtime_revision);
    }
    for (const auto& m : models) {
      const ServiceModel& id = m->identity;
      if (id.id == s.model.id && id.revision == s.model.revision && id.artifact_digest == s.model.artifact_digest &&
          (id.adapter_id.empty() || id.adapter_id == s.model.adapter_id)) {
        r.model = m.get();
        break;
      }
    }
    if (r.model == nullptr) {
      return failure("unsupported-model-revision", "not_started", true,
                     "model " + s.model.id + " at revision " + s.model.revision + " is not installed in this runtime");
    }
    if (unix_ms() >= s.deadline_unix_ms) {
      return failure("deadline-exceeded", "not_started", false, "the deadline passed before admission");
    }
    const size_t weights = r.model->model->weight_bytes();
    const int64_t ceiling = s.memory.max_resident_weight_bytes.value_or(s.memory.max_runtime_memory_bytes);
    if (static_cast<int64_t>(weights) > ceiling) {
      return failure("memory-limit", "not_started", true,
                     "the model's " + std::to_string(weights) + " weight bytes exceed the request's memory ceiling");
    }
    std::vector<ChatMessage> messages;
    for (const wire::Message& m : s.messages) messages.push_back({m.role, m.content});
    try {
      const std::string prompt = r.model->chat_template.render(messages, r.model->context);
      // The template already carries the BOS token, as apply_chat_template assumes.
      r.prompt = r.model->tokenizer.encode(prompt, false);
    } catch (const std::exception& error) {
      return failure("template-error", "not_started", false, error.what());
    }
    const int64_t limit = std::min<int64_t>(s.objective.context_tokens, r.model->model->max_context());
    const int64_t needed = static_cast<int64_t>(r.prompt.size()) + s.objective.max_output_tokens;
    if (r.prompt.empty() || needed > limit) {
      return failure("context-exceeded", "not_started", false,
                     std::to_string(r.prompt.size()) + " prompt tokens plus " +
                         std::to_string(s.objective.max_output_tokens) + " output tokens exceed the " +
                         std::to_string(limit) + "-token context");
    }
    return std::nullopt;
  }

  void control(std::string_view document) {
    const wire::Control c = wire::decode_control(document);
    if (c.action == "drain") {
      std::lock_guard<std::mutex> lock(mutex);
      draining = true;
      return;
    }
    std::shared_ptr<Request> queued;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto it = active.find(*c.target_request_id);
      if (it == active.end()) return;
      std::shared_ptr<Request> r = it->second;
      r->cancelled = true;
      auto at = std::find(queue.begin(), queue.end(), r);
      if (at != queue.end()) {
        queued = r;
        queue.erase(at);
      }
    }
    if (queued) {
      finish(*queued, cancelled_outcome(*queued, "not_started", "cancelled before it started"));
      forget(*queued);
    }
  }

  void wait_idle() {
    std::unique_lock<std::mutex> lock(mutex);
    idle_cv.wait(lock, [&] { return queue.empty() && running.empty() && active.empty(); });
  }

  // Sequences no running request holds (with mutex held): requests beyond
  // them wait in the queue.
  int64_t free_sequences() const {
    return std::max<int64_t>(0, options.max_sequences - static_cast<int64_t>(running.size()));
  }

  // ---- execution ------------------------------------------------------------

  void work() {
    for (;;) {
      std::shared_ptr<Request> r;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return stopping || !queue.empty(); });
        if (queue.empty()) return;
        r = queue.front();
        queue.pop_front();
        running.push_back(r);
      }
      finish(*r, run(*r));
      release(r);
    }
  }

  // Leaves the running set and the requests in flight.
  void release(const std::shared_ptr<Request>& r) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      running.erase(std::remove(running.begin(), running.end(), r), running.end());
    }
    forget(*r);
  }

  // One request's generation as its tokens arrive: text through its stop
  // strings into coalesced text_delta events, cancellation and the deadline,
  // and its outcome at the end. Model::generate feeds it (one request at a
  // time) or the batched worker does.
  struct Generation {
    Generation(Impl& service, Request& request, double started)
        : self(service),
          r(request),
          queue_ms((started - request.admitted) * 1000.0),
          text(request.model->tokenizer, request.start.sampling.stop),
          last_flush(started) {
      const wire::Start& s = r.start;
      sampling.temperature = static_cast<float>(s.sampling.temperature);
      sampling.top_p = static_cast<float>(s.sampling.top_p);
      sampling.seed = s.sampling.seed ? static_cast<uint64_t>(*s.sampling.seed)
                                      : (uint64_t{std::random_device{}()} << 32) ^ std::random_device{}();
    }

    void flush() {
      if (pending.empty()) return;
      wire::Event delta;
      delta.kind = "text_delta";
      delta.text_delta = pending;
      delta.generated_tokens_delta = pending_tokens;
      self.send(r, std::move(delta));
      r.sent_text += pending;
      pending.clear();
      pending_tokens = 0;
      last_flush = now_seconds();
    }

    // Takes a generated token; false once generation must stop after it.
    bool on_token(int32_t token) {
      pending += text.push(token);
      ++pending_tokens;
      if (r.cancelled) return false;
      if (unix_ms() >= r.start.deadline_unix_ms) {
        deadline_hit = true;
        return false;
      }
      if (text.stopped()) return false;
      if ((now_seconds() - last_flush) * 1000.0 >= self.options.text_flush_ms || pending.size() >= kMaxDeltaBytes) {
        flush();
      }
      return true;
    }

    // The outcome once generation has ended with `tokens`.
    Outcome complete(const std::vector<int32_t>& tokens, const GenerationTiming& timing) {
      const int64_t generated = static_cast<int64_t>(tokens.size());
      if (r.cancelled) {
        Outcome out = self.cancelled_outcome(r, r.sent_text.empty() ? "running" : "partial_output",
                                             "cancelled while running", queue_ms);
        out.completion_tokens = generated;
        out.metrics = self.metrics(r, queue_ms, timing);
        return out;
      }
      if (deadline_hit) {
        const char* stage = r.sent_text.empty() ? "running" : "partial_output";
        return self.failed_outcome(r, failure("deadline-exceeded", stage, false, "the deadline passed while generating"),
                                   queue_ms, generated, timing);
      }
      pending += text.finish();
      flush();
      const auto& eos = r.model->model->eos_token_ids();
      const bool at_eos = !tokens.empty() && std::find(eos.begin(), eos.end(), tokens.back()) != eos.end();
      Outcome out = self.base_outcome(r);
      out.status = "completed";
      out.finish_reason =
          text.stopped() || at_eos || generated < r.start.objective.max_output_tokens ? "stop" : "length";
      out.output_text = text.text();
      out.completion_tokens = generated;
      out.metrics = self.metrics(r, queue_ms, timing);
      return out;
    }

    Impl& self;
    Request& r;
    double queue_ms;
    SamplingParams sampling;
    TextStream text;
    std::string pending;
    int64_t pending_tokens = 0;
    double last_flush;
    bool deadline_hit = false;
  };

  // A request that cannot start: cancelled or out of time in the queue.
  std::optional<Outcome> refused_at_start(Request& r, double started) {
    const double queue_ms = (started - r.admitted) * 1000.0;
    if (r.cancelled) return cancelled_outcome(r, "not_started", "cancelled before it started", queue_ms);
    if (unix_ms() >= r.start.deadline_unix_ms) {
      return failed_outcome(r, failure("deadline-exceeded", "not_started", false, "the deadline passed in the queue"),
                            queue_ms, 0, {});
    }
    return std::nullopt;
  }

  Outcome run(Request& r) {
    const double started = now_seconds();
    if (std::optional<Outcome> refused = refused_at_start(r, started)) return std::move(*refused);
    Model& model = *r.model->model;
    Generation g(*this, r, started);
    GenerationTiming timing;
    std::vector<int32_t> tokens;
    try {
      model.reset();
      tokens = model.generate(r.prompt, static_cast<int>(r.start.objective.max_output_tokens), g.sampling, true,
                              &timing, [&](int32_t token) { return g.on_token(token); });
    } catch (const std::exception& error) {
      const char* stage = r.sent_text.empty() ? "running" : "partial_output";
      return failed_outcome(r, failure("engine-error", stage, false, error.what()), g.queue_ms, 0, timing);
    }
    return g.complete(tokens, timing);
  }

  // ---- batched execution --------------------------------------------------------

  // A running request of the batched worker: its model's sequence, the tokens
  // so far and when each arrived.
  struct Slot {
    std::shared_ptr<Request> r;
    std::unique_ptr<Generation> g;
    int sequence = -1;
    std::vector<int32_t> tokens;
    GenerationTiming timing;
    int64_t largest_batch = 1;  // the most requests a pass of it decoded at once
  };

  // Continuous batching: queued requests start in their model's free
  // sequences (prefilled one after another), then every step decodes each
  // model's running requests in one batched pass; a request that ends frees
  // its sequence for the next.
  void work_batched() {
    std::vector<std::unique_ptr<Slot>> slots;
    for (;;) {
      std::vector<std::shared_ptr<Request>> starting;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return stopping || !queue.empty() || !slots.empty(); });
        if (stopping && queue.empty() && slots.empty()) return;
        // In admission order, while the front request's model has a free
        // sequence (a full model holds back the queue behind it).
        while (!queue.empty()) {
          LoadedModel* m = queue.front()->model;
          const auto holding = [&](const std::shared_ptr<Request>& other) { return other->model == m; };
          const int64_t held = std::count_if(running.begin(), running.end(), holding);
          if (held >= options.max_sequences) break;
          running.push_back(queue.front());
          starting.push_back(queue.front());
          queue.pop_front();
        }
      }
      for (const auto& r : starting) begin(slots, r);
      step(slots);
    }
  }

  // Prefills a starting request into a free sequence and takes its first
  // token; it joins `slots` unless it already ended.
  void begin(std::vector<std::unique_ptr<Slot>>& slots, const std::shared_ptr<Request>& r) {
    const double started = now_seconds();
    if (std::optional<Outcome> refused = refused_at_start(*r, started)) {
      finish(*r, std::move(*refused));
      release(r);
      return;
    }
    LoadedModel& m = *r->model;
    auto slot = std::make_unique<Slot>();
    slot->r = r;
    slot->g = std::make_unique<Generation>(*this, *r, started);
    slot->sequence = static_cast<int>(std::find(m.busy.begin(), m.busy.end(), false) - m.busy.begin());
    LK_CHECK(slot->sequence < options.max_sequences, "service: no free sequence for a starting request");
    m.busy[static_cast<size_t>(slot->sequence)] = true;
    int32_t token = 0;
    try {
      m.model->select_sequence(slot->sequence);
      m.model->reset();
      const float* logits = m.model->prefill(r->prompt.data(), static_cast<int>(r->prompt.size()));
      token = sample_token(logits, m.model->vocab_size(), slot->g->sampling, 0);
    } catch (const std::exception& error) {
      end(*slot, failed_outcome(*r, failure("engine-error", "running", false, error.what()), slot->g->queue_ms, 0,
                                slot->timing));
      return;
    }
    const double first = now_seconds();
    slot->timing.prefill_seconds = first - started;
    slot->timing.token_times = {first};
    if (take(*slot, token)) {
      slots.push_back(std::move(slot));
    } else {
      end(*slot, slot->g->complete(slot->tokens, slot->timing));
    }
  }

  // One batched pass per model over its running requests; those that end
  // leave `slots`.
  void step(std::vector<std::unique_ptr<Slot>>& slots) {
    std::vector<LoadedModel*> stepped;
    for (const auto& slot : slots) {
      LoadedModel* m = slot->r->model;
      if (std::find(stepped.begin(), stepped.end(), m) != stepped.end()) continue;
      stepped.push_back(m);
      std::vector<Slot*> rows;
      std::vector<BatchStep> steps;
      for (const auto& other : slots) {
        if (other->r->model != m) continue;
        rows.push_back(other.get());
        steps.push_back(BatchStep{other->sequence, other->tokens.back(), other->g->sampling,
                                  static_cast<int>(other->tokens.size())});
      }
      std::vector<int32_t> next;
      try {
        next = m->model->decode_batch(steps);
      } catch (const std::exception& error) {
        for (Slot* row : rows) {
          const char* stage = row->r->sent_text.empty() ? "running" : "partial_output";
          end(*row, failed_outcome(*row->r, failure("engine-error", stage, false, error.what()), row->g->queue_ms,
                                   static_cast<int64_t>(row->tokens.size()), row->timing));
        }
        continue;
      }
      const double now = now_seconds();
      for (size_t i = 0; i < rows.size(); ++i) {
        Slot& row = *rows[i];
        row.largest_batch = std::max(row.largest_batch, static_cast<int64_t>(rows.size()));
        row.timing.token_times.push_back(now);
        if (!take(row, next[i])) end(row, row.g->complete(row.tokens, row.timing));
      }
    }
    slots.erase(std::remove_if(slots.begin(), slots.end(), [](const auto& slot) { return slot->sequence < 0; }),
                slots.end());
  }

  // Records a generated token; false once the request must stop after it
  // (its callback, an EOS token, or its output limit), as Model::generate
  // stops.
  static bool take(Slot& slot, int32_t token) {
    slot.tokens.push_back(token);
    const auto& times = slot.timing.token_times;
    slot.timing.decode_seconds = times.back() - times.front();
    const bool more = slot.g->on_token(token);
    const auto& eos = slot.r->model->model->eos_token_ids();
    const bool at_eos = std::find(eos.begin(), eos.end(), token) != eos.end();
    return more && !at_eos && static_cast<int64_t>(slot.tokens.size()) < slot.r->start.objective.max_output_tokens;
  }

  // Delivers a slot's receipt and frees its sequence.
  void end(Slot& slot, Outcome out) {
    out.metrics.maximum_batch_size = slot.largest_batch;
    finish(*slot.r, std::move(out));
    slot.r->model->busy[static_cast<size_t>(slot.sequence)] = false;
    slot.sequence = -1;
    release(slot.r);
  }

  // ---- receipts ---------------------------------------------------------------

  std::string backend_name() const { return models.front()->model->backend(); }

  Outcome base_outcome(const Request& r) const {
    Outcome out;
    out.backend = r.model ? r.model->model->backend() : backend_name();
    out.plan_id = r.model ? std::string("lokahi-native-") + r.model->model->architecture() + "-v1"
                          : std::string("lokahi-admission-v1");
    out.metrics.peak_memory_bytes = peak_rss_bytes();
    return out;
  }

  wire::Metrics metrics(const Request& r, double queue_ms, const GenerationTiming& timing) const {
    wire::Metrics m;
    m.queue_ms = queue_ms;
    m.peak_memory_bytes = peak_rss_bytes();
    const size_t n = timing.token_times.size();
    if (n > 0) m.ttft_ms = std::max(0.0, (timing.token_times.front() - r.admitted) * 1000.0);
    if (n > 1 && timing.decode_seconds > 0.0) {
      m.inter_token_latency_ms = timing.decode_seconds * 1000.0 / static_cast<double>(n - 1);
      m.decode_tokens_per_second = static_cast<double>(n - 1) / timing.decode_seconds;
    }
    return m;
  }

  Outcome rejected_outcome(wire::Failure why) const {
    Outcome out;
    out.backend = backend_name();
    out.plan_id = "lokahi-admission-v1";
    out.status = "rejected";
    out.finish_reason = "rejected";
    out.failure = std::move(why);
    return out;
  }

  Outcome cancelled_outcome(const Request& r, const char* stage, const std::string& message,
                            double queue_ms = 0.0) const {
    Outcome out = base_outcome(r);
    out.status = "cancelled";
    out.finish_reason = "cancelled";
    out.failure = failure("cancelled", stage, false, message);
    out.metrics.queue_ms = queue_ms;
    if (!r.sent_text.empty()) out.output_text = r.sent_text;
    return out;
  }

  Outcome failed_outcome(const Request& r, wire::Failure why, double queue_ms, int64_t generated,
                         const GenerationTiming& timing) const {
    Outcome out = base_outcome(r);
    out.status = "failed";
    out.finish_reason = "error";
    out.failure = std::move(why);
    out.completion_tokens = generated;
    out.metrics = metrics(r, queue_ms, timing);
    if (!r.sent_text.empty()) out.output_text = r.sent_text;
    return out;
  }

  // ---- delivery ---------------------------------------------------------------

  void deliver(Request& r, const std::string& document) {
    try {
      r.sink(document);
    } catch (...) {
      // A failing sink must not take the service down; the host owns its end.
    }
  }

  void send(Request& r, wire::Event event) {
    event.request_id = r.start.request_id;
    event.sequence = r.next_sequence++;
    event.emitted_at_unix_ms = unix_ms();
    std::string document = wire::encode(event);
    r.stream.accept(event);
    deliver(r, document);
  }

  // Exactly one receipt per request. If the planned receipt cannot be
  // encoded (a bug), a minimal failure receipt is sent instead.
  void finish(Request& r, Outcome out) {
    wire::Receipt receipt;
    receipt.request_id = r.start.request_id;
    receipt.terminal_sequence = r.next_sequence++;
    receipt.completed_at_unix_ms = unix_ms();
    receipt.status = out.status;
    receipt.finish_reason = out.finish_reason;
    receipt.output_text = out.output_text;
    const int64_t prompt = static_cast<int64_t>(r.prompt.size());
    receipt.usage = wire::Usage{prompt, out.completion_tokens, prompt + out.completion_tokens};
    receipt.metrics = out.metrics;
    const char* route = options.max_sequences > 1 ? "batch_generator" : "direct_single_sequence";
    receipt.execution = wire::Execution{out.backend, route, out.plan_id, {}, false, std::nullopt};
    receipt.failure = out.failure;
    std::string document;
    try {
      document = wire::encode(receipt);
    } catch (const std::exception& error) {
      receipt.status = "failed";
      receipt.finish_reason = "error";
      receipt.output_text.reset();
      receipt.metrics = wire::Metrics{};
      receipt.failure = failure("receipt-error", r.sent_text.empty() ? "running" : "partial_output", false,
                                std::string("could not encode the receipt: ") + error.what());
      document = wire::encode(receipt);
    }
    r.stream.accept(receipt);
    deliver(r, document);
  }

  void forget(const Request& r) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto it = active.find(r.start.request_id);
      if (it != active.end() && it->second.get() == &r) active.erase(it);
    }
    idle_cv.notify_all();
  }

  ServiceOptions options;
  std::vector<std::unique_ptr<LoadedModel>> models;
  std::mutex mutex;
  std::condition_variable cv, idle_cv;
  std::deque<std::shared_ptr<Request>> queue;
  std::vector<std::shared_ptr<Request>> running;  // started and not yet terminal
  std::map<std::string, std::shared_ptr<Request>> active;  // admitted and not yet terminal, by request_id
  bool draining = false;
  bool stopping = false;
  std::thread worker;
};

Service::Service(ServiceOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
Service::~Service() = default;
void Service::start(std::string_view document, EnvelopeSink sink) { impl_->start(document, std::move(sink)); }
void Service::control(std::string_view document) { impl_->control(document); }
void Service::wait_idle() { impl_->wait_idle(); }

}  // namespace lokahi
