// The engine side of the Common Compute boundary (Lōkahi wire v1).
//
// The host hands over encoded `start` and `control` documents and receives
// encoded `event` and `receipt` documents through a sink. Every document
// sent back has passed the codec's own validation and the per-request stream
// rules, so a bug here fails closed instead of reaching the host. The rules
// are: accepted first at sequence 0, sequences increasing, and exactly one
// terminal receipt.
//
// Requests run one at a time (v1 batch size 1) on a worker thread, in
// admission order, behind a bounded queue. Prompts are rendered with the
// model's chat template and tokenized at admission, so oversized requests are
// rejected before they are accepted. Text is streamed through TextStream
// (stop strings, no rewrites) and coalesced into one event per
// `text_flush_ms` rather than one per token.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace lokahi {

class Json;

// A model the service may run, and the identity a request must name exactly.
struct ServiceModel {
  std::string id;
  std::string revision;
  std::string artifact_digest;
  std::string adapter_id;  // empty accepts any
  std::string directory;   // Hugging Face snapshot (config, safetensors, tokenizer, chat template)
};

struct ServiceOptions {
  std::vector<ServiceModel> models;
  std::string backend = "auto";
  int max_context = 4096;
  std::string kv_cache = "auto";  // LoadOptions::kv_cache
  int queue_limit = 8;           // requests that may wait behind the running one
  double text_flush_ms = 50.0;   // coalescing interval for text_delta events
  std::string runtime_revision;  // 64 hex; when set, requests for another revision are rejected

  // {"models": [{"id", "revision", "artifact_digest", "adapter_id"?, "directory"}],
  //  "backend"?, "max_context"?, "kv_cache"?, "queue_limit"?, "text_flush_ms"?, "runtime_revision"?}
  static ServiceOptions from_json(const Json& config);
};

// Receives each encoded event or receipt. Called from service threads, one
// call at a time per request; it must not call back into the service.
using EnvelopeSink = std::function<void(const std::string& envelope)>;

class Service {
 public:
  explicit Service(ServiceOptions options);  // loads every model, tokenizer and chat template
  ~Service();                                // cancels outstanding requests and delivers their receipts
  Service(const Service&) = delete;
  Service& operator=(const Service&) = delete;

  // Admits a start document. Throws wire::ContractError when the document is
  // invalid or its request_id is already in flight: no receipt can be
  // addressed then. Every other outcome, rejection included, reaches `sink`.
  void start(std::string_view document, EnvelopeSink sink);

  // Applies a cancel or drain control; throws wire::ContractError when the
  // document is invalid. Cancelling an unknown or finished request does
  // nothing, so cancellation is idempotent. Drain refuses new requests and
  // lets admitted ones finish.
  void control(std::string_view document);

  // Blocks until nothing is queued or running.
  void wait_idle();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lokahi
