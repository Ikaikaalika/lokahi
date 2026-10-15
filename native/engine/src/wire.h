// Lōkahi wire protocol v1 (integrations/commoncompute/lokahi-wire-v1): the
// codec and per-request stream rules between the engine and the Common Compute
// provider. This is a trust boundary, so every rejection throws ContractError.
//
// Document verdicts follow the kit's reference verifier, tools/verify_kit.py,
// except where it accepts input its own Swift decoders reject:
//  - Documents must be UTF-8; one leading byte order mark is skipped, as the
//    reference does. The reference also decodes UTF-16 and UTF-32 and accepts
//    surrogates encoded directly in UTF-8.
//  - Integers must fit in int64. The reference gives fields such as sequence
//    and created_at_unix_ms no upper bound.
//
// The reference accepts lone surrogate escapes such as "\ud800", so decoded
// strings may hold a surrogate in its three-byte form. Lengths count code
// points, as Python's len does. Encoders emit compact UTF-8 JSON in schema key
// order. Besides everything the reference rejects, they refuse the two things
// only the Swift decoders reject: lone surrogates, and stop strings that are
// canonically equivalent (Swift compares strings that way).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common.h"

namespace lokahi::wire {

inline constexpr int kSchemaVersion = 1;

class ContractError : public lokahi::Error {
  using lokahi::Error::Error;
};

struct Model {
  std::string id, revision, artifact_digest, adapter_id, quantization;
};

struct Message {
  std::string role, content;
};

struct Sampling {
  double temperature = 0;
  double top_p = 1;
  std::optional<int64_t> seed;
  std::vector<std::string> stop;
};

struct Objective {
  std::string workload_class;
  int64_t context_tokens = 0, max_output_tokens = 0, batch_size = 1;
  std::optional<double> target_ttft_ms, min_decode_tokens_per_second;
};

struct Memory {
  int64_t max_runtime_memory_bytes = 0;
  std::optional<int64_t> max_resident_weight_bytes;
  std::string ssd_offload;
};

struct Start {
  std::string request_id, task_id, operation, runtime_revision;
  int64_t created_at_unix_ms = 0, deadline_unix_ms = 0;
  Model model;
  std::vector<Message> messages;
  Sampling sampling;
  Objective objective;
  Memory memory;
  std::string deployment_mode;
};

struct Control {
  std::string control_id, action;
  std::optional<std::string> target_request_id;
  int64_t issued_at_unix_ms = 0;
};

struct Event {
  std::string request_id;
  int64_t sequence = 0;
  std::string kind;
  int64_t emitted_at_unix_ms = 0;
  std::optional<double> progress;
  std::optional<std::string> text_delta;
  std::optional<int64_t> generated_tokens_delta, queue_position, queue_depth;
  std::optional<std::string> message;
};

struct Usage {
  int64_t prompt_tokens = 0, completion_tokens = 0, total_tokens = 0;
};

struct Metrics {
  double queue_ms = 0;
  std::optional<double> ttft_ms, inter_token_latency_ms, decode_tokens_per_second;
  int64_t peak_memory_bytes = 0, maximum_batch_size = 1;
};

struct Execution {
  std::string backend, route, plan_id;
  std::vector<std::string> evidence_ids;
  bool fallback_used = false;
  std::optional<std::string> fallback_reason;
};

struct Failure {
  std::string code, stage;
  bool retryable = false;
  std::string message;
};

struct Receipt {
  std::string request_id;
  int64_t terminal_sequence = 0;
  std::string status, finish_reason;
  int64_t completed_at_unix_ms = 0;
  std::optional<std::string> output_text;
  Usage usage;
  Metrics metrics;
  Execution execution;
  std::optional<Failure> failure;
};

enum class MessageType { Start, Control, Event, Receipt };

// The message_type spelling: "start", "control", "event" or "receipt".
std::string_view name(MessageType type);

// Full validation of one encoded document (size limit, strict JSON, schema,
// semantic rules); returns its type.
MessageType validate(std::string_view document);
Start decode_start(std::string_view document);
Control decode_control(std::string_view document);
Event decode_event(std::string_view document);
Receipt decode_receipt(std::string_view document);

// Encoders validate their own output (fail closed) and emit compact UTF-8 JSON
// with a fixed key order.
std::string encode(const Start&);
std::string encode(const Control&);
std::string encode(const Event&);
std::string encode(const Receipt&);

// Per-request stream rules from the Swift reference (LokahiStreamStateV1):
// the first event is `accepted`, exactly one receipt ends the stream and
// nothing follows it. Sequences are also contiguous: every event and the
// receipt carry next_sequence(), so the receipt's terminal_sequence is one past
// the last event. Swift only requires them to increase. Each message must be
// valid on its own, and a rejected message leaves the state unchanged.
class StreamValidator {
 public:
  explicit StreamValidator(std::string request_id);
  void accept(const Event& event);
  void accept(const Receipt& receipt);
  bool terminated() const { return terminated_; }
  int64_t next_sequence() const { return next_sequence_; }

 private:
  std::string request_id_;
  int64_t next_sequence_ = 0;
  bool terminated_ = false;
};

}  // namespace lokahi::wire
