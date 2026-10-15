#include "wire.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <limits>

#include "unicode.h"

namespace lokahi::wire {
namespace {

struct TypeInfo {
  MessageType type;
  std::string_view name;
  size_t limit;  // MESSAGE_LIMITS in verify_kit.py: encoded bytes
};

// In MessageType order.
constexpr TypeInfo kTypes[] = {
    {MessageType::Start, "start", 1024 * 1024},
    {MessageType::Control, "control", 16 * 1024},
    {MessageType::Event, "event", 128 * 1024},
    {MessageType::Receipt, "receipt", 2 * 1024 * 1024},
};
constexpr size_t kLargestLimit = 2 * 1024 * 1024;
constexpr int64_t kMaxDeadlineMs = 24 * 60 * 60 * 1000;
constexpr size_t kMaxCombinedContent = 524288;
constexpr int64_t kMinRuntimeMemoryBytes = 256 * 1024 * 1024;
constexpr double kNoMaximum = std::numeric_limits<double>::infinity();

[[noreturn]] void reject(const std::string& message) { throw ContractError("wire: " + message); }

// A member name fit for an error message: printable ASCII, at most 64 bytes.
std::string printable(std::string_view text) {
  std::string out;
  for (char c : text.substr(0, 64)) out.push_back(c >= 0x20 && c < 0x7F ? c : '?');
  return out;
}

// Python's len: one per code point, counting each lone surrogate.
size_t code_points(std::string_view text) {
  size_t count = 0;
  for (char c : text) count += (static_cast<unsigned char>(c) & 0xC0) != 0x80;
  return count;
}

// A parsed JSON value. Numbers keep their lexeme so that integer fields are
// exact over int64 and number fields round once, as Python's float() does.
struct Value {
  enum class Kind : uint8_t { Null, Bool, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool boolean = false;
  bool integral = false;          // no fraction or exponent, so Python reads an int
  std::string text;               // a string's contents, or a number's lexeme
  std::vector<std::string> keys;  // object member names, in document order
  std::vector<Value> items;       // array items, or the values of `keys`
};

// RFC 8259 plus what Python's json.loads also accepts: NaN, Infinity,
// -Infinity and lone surrogate escapes. The input is known to be UTF-8.
class Parser {
 public:
  explicit Parser(std::string_view text) : text_(text) {}

  Value document() {
    Value root = value(0);
    skip_whitespace();
    if (pos_ != text_.size()) reject("invalid JSON: extra data after the document");
    return root;
  }

 private:
  // A valid document nests three containers deep and holds at most 421
  // values, so these bounds reject only invalid documents while capping the
  // stack and memory that hostile input can claim.
  static constexpr int kMaxDepth = 16;
  static constexpr size_t kMaxValues = 1024;

  bool at(char c) const { return pos_ < text_.size() && text_[pos_] == c; }

  bool consume(std::string_view literal) {
    if (text_.substr(pos_, literal.size()) != literal) return false;
    pos_ += literal.size();
    return true;
  }

  bool digits() {
    const size_t start = pos_;
    while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
    return pos_ > start;
  }

  void skip_whitespace() {
    while (at(' ') || at('\t') || at('\n') || at('\r')) ++pos_;
  }

  Value value(int depth) {
    if (depth > kMaxDepth) reject("invalid JSON: nesting too deep");
    if (++values_ > kMaxValues) reject("invalid JSON: too many values");
    skip_whitespace();
    Value out;
    if (at('{')) {
      object(out, depth);
    } else if (at('[')) {
      array(out, depth);
    } else if (at('"')) {
      out.kind = Value::Kind::String;
      out.text = string();
    } else if (consume("true")) {
      out.kind = Value::Kind::Bool;
      out.boolean = true;
    } else if (consume("false")) {
      out.kind = Value::Kind::Bool;
    } else if (!consume("null")) {
      out.kind = Value::Kind::Number;
      for (std::string_view constant : {"NaN", "Infinity", "-Infinity"}) {
        if (consume(constant)) {
          out.text = constant;
          return out;
        }
      }
      number(out);
    }
    return out;
  }

  void object(Value& out, int depth) {
    out.kind = Value::Kind::Object;
    ++pos_;
    skip_whitespace();
    if (consume("}")) return;
    while (true) {
      skip_whitespace();
      if (!at('"')) reject("invalid JSON: expected a member name");
      std::string key = string();
      // Python's object_pairs_hook compares decoded names, so an escaped spelling repeats a name.
      if (std::find(out.keys.begin(), out.keys.end(), key) != out.keys.end()) {
        reject("duplicate JSON key: " + printable(key));
      }
      skip_whitespace();
      if (!consume(":")) reject("invalid JSON: expected ':'");
      out.items.push_back(value(depth + 1));
      out.keys.push_back(std::move(key));
      skip_whitespace();
      if (consume("}")) return;
      if (!consume(",")) reject("invalid JSON: expected ',' or '}'");
    }
  }

  void array(Value& out, int depth) {
    out.kind = Value::Kind::Array;
    ++pos_;
    skip_whitespace();
    if (consume("]")) return;
    while (true) {
      out.items.push_back(value(depth + 1));
      skip_whitespace();
      if (consume("]")) return;
      if (!consume(",")) reject("invalid JSON: expected ',' or ']'");
    }
  }

  std::string string() {
    ++pos_;  // opening quote
    std::string out;
    while (true) {
      if (pos_ >= text_.size()) reject("invalid JSON: unterminated string");
      const char c = text_[pos_++];
      if (c == '"') return out;
      if (static_cast<unsigned char>(c) < 0x20) reject("invalid JSON: control character in string");
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (pos_ >= text_.size()) reject("invalid JSON: unterminated string");
      switch (text_[pos_++]) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': unicode::append_utf8(out, escape()); break;
        default: reject("invalid JSON: invalid escape");
      }
    }
  }

  // The code point of a \uXXXX escape whose "\u" was just read. As in Python,
  // a high surrogate joins a low surrogate escape that immediately follows;
  // any other surrogate stays lone and is stored in its three-byte form.
  char32_t escape() {
    const char32_t high = hex4();
    if (high >= 0xD800 && high <= 0xDBFF && text_.substr(pos_, 2) == "\\u") {
      const size_t mark = pos_;
      pos_ += 2;
      const char32_t low = hex4();
      if (low >= 0xDC00 && low <= 0xDFFF) return 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
      pos_ = mark;
    }
    return high;
  }

  char32_t hex4() {
    if (text_.size() - pos_ < 4) reject("invalid JSON: invalid \\uXXXX escape");
    char32_t cp = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = text_[pos_++];
      int digit;
      if (c >= '0' && c <= '9') digit = c - '0';
      else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
      else reject("invalid JSON: invalid \\uXXXX escape");
      cp = (cp << 4) | static_cast<char32_t>(digit);
    }
    return cp;
  }

  // A lexeme without fraction or exponent is integral: Python reads an int.
  void number(Value& out) {
    const size_t start = pos_;
    consume("-");
    if (!consume("0") && !digits()) reject("invalid JSON: invalid value");
    out.integral = true;
    if (consume(".")) {
      if (!digits()) reject("invalid JSON: invalid number");
      out.integral = false;
    }
    if (consume("e") || consume("E")) {
      if (!consume("+")) consume("-");
      if (!digits()) reject("invalid JSON: invalid number");
      out.integral = false;
    }
    out.text = std::string(text_.substr(start, pos_ - start));
  }

  std::string_view text_;
  size_t pos_ = 0;
  size_t values_ = 0;
};

// Python's float() of the parsed value: the correctly rounded double, or an
// infinity where Python's int-to-float conversion would overflow. strtod also
// reads the NaN and Infinity literals. An integer literal is a Python int, so
// "-0" is +0.0.
double to_double(const Value& v) {
  const double value = std::strtod(v.text.c_str(), nullptr);
  return v.integral && value == 0 ? 0.0 : value;
}

std::optional<int64_t> to_int64(const Value& v) {
  int64_t out = 0;
  const char* end = v.text.data() + v.text.size();
  const auto [ptr, error] = std::from_chars(v.text.data(), end, out);
  if (error != std::errc() || ptr != end) return std::nullopt;
  return out;
}

bool contains(std::initializer_list<std::string_view> names, std::string_view name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

const Value* member(const Value& object, std::string_view key) {
  for (size_t i = 0; i < object.keys.size(); ++i) {
    if (object.keys[i] == key) return &object.items[i];
  }
  return nullptr;
}

// A member that exact() has already required.
const Value& field(const Value& object, std::string_view key) { return *member(object, key); }

// The validators below port verify_kit.py's helpers of the same purpose. Names
// are views so that no reference-returning helper takes a temporary.

[[noreturn]] void reject(std::string_view name, std::string_view problem) {
  reject(std::string(name) + " " + std::string(problem));
}

void exact(const Value& object, std::string_view name, std::initializer_list<std::string_view> required,
           std::initializer_list<std::string_view> optional = {}) {
  bool invalid = false;
  std::string missing, unknown;
  for (std::string_view key : required) {
    if (member(object, key)) continue;
    missing += (invalid ? ", " : "") + std::string(key);
    invalid = true;
  }
  for (const std::string& key : object.keys) {
    if (contains(required, key) || contains(optional, key)) continue;
    unknown += (unknown.empty() ? "\"" : ", \"") + printable(key) + "\"";
    invalid = true;
  }
  if (invalid) reject(name, "keys invalid; missing=[" + missing + "], unknown=[" + unknown + "]");
}

const Value& as_object(const Value& v, std::string_view name) {
  if (v.kind != Value::Kind::Object) reject(name, "must be an object");
  return v;
}

const std::vector<Value>& as_array(const Value& v, std::string_view name, size_t maximum, size_t minimum = 0) {
  if (v.kind != Value::Kind::Array || v.items.size() < minimum || v.items.size() > maximum) {
    reject(name, "must contain " + std::to_string(minimum) + "..." + std::to_string(maximum) + " items");
  }
  return v.items;
}

const std::string& as_string(const Value& v, std::string_view name, size_t minimum, size_t maximum) {
  const size_t length = v.kind == Value::Kind::String ? code_points(v.text) : 0;
  if (v.kind != Value::Kind::String || length < minimum || length > maximum) {
    reject(name, "must be a string of length " + std::to_string(minimum) + "..." + std::to_string(maximum));
  }
  if (v.text.find('\0') != std::string::npos) reject(name, "must not contain NUL");
  return v.text;
}

bool as_bool(const Value& v, std::string_view name) {
  if (v.kind != Value::Kind::Bool) reject(name, "must be a boolean");
  return v.boolean;
}

int64_t as_integer(const Value& v, std::string_view name, int64_t minimum,
                   int64_t maximum = std::numeric_limits<int64_t>::max()) {
  if (v.kind != Value::Kind::Number || !v.integral) reject(name, "must be an integer");
  const std::optional<int64_t> value = to_int64(v);
  if (!value) reject(name, "does not fit in int64");
  if (*value < minimum || *value > maximum) reject(name, "is outside its permitted range");
  return *value;
}

enum class Minimum { Inclusive, Exclusive };

double as_number(const Value& v, std::string_view name, double minimum, double maximum = kNoMaximum,
                 Minimum bound = Minimum::Inclusive) {
  if (v.kind != Value::Kind::Number) reject(name, "must be a number");
  const double value = to_double(v);
  if (!std::isfinite(value)) reject(name, "must be finite");
  const bool below = bound == Minimum::Exclusive ? value <= minimum : value < minimum;
  if (below || value > maximum) reject(name, "is outside its permitted range");
  return value;
}

// One of a fixed set of strings; Python's `in` rejects every other value.
const std::string& as_enum(const Value& v, std::initializer_list<std::string_view> allowed, std::string_view error) {
  if (v.kind != Value::Kind::String || !contains(allowed, v.text)) reject(std::string(error));
  return v.text;
}

// Python's `value != 1` with bools excluded: 1.0 and 1e0 also equal one.
bool equals_one(const Value& v) { return v.kind == Value::Kind::Number && to_double(v) == 1.0; }

bool ascii_alnum(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'); }

// IDENTIFIER: ^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$
bool is_identifier(std::string_view s) {
  if (s.empty() || s.size() > 128 || !ascii_alnum(s[0])) return false;
  return std::all_of(s.begin() + 1, s.end(),
                     [](char c) { return ascii_alnum(c) || c == '.' || c == '_' || c == ':' || c == '-'; });
}

// ADAPTER_ID: ^[a-z0-9][a-z0-9._-]{0,63}$
bool is_adapter(std::string_view s) {
  const auto plain = [](char c) { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'); };
  if (s.empty() || s.size() > 64 || !plain(s[0])) return false;
  return std::all_of(s.begin() + 1, s.end(), [&](char c) { return plain(c) || c == '.' || c == '_' || c == '-'; });
}

bool lower_hex(std::string_view s, size_t length) {
  return s.size() == length &&
         std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

// Python's set() comparison; decoded strings compare by code point here too.
bool unique(const std::vector<std::string>& items) {
  for (size_t i = 0; i < items.size(); ++i) {
    if (std::find(items.begin() + static_cast<std::ptrdiff_t>(i) + 1, items.end(), items[i]) != items.end()) {
      return false;
    }
  }
  return true;
}

const std::string& as_identifier(const Value& v, std::string_view name) {
  const std::string& s = as_string(v, name, 1, 128);
  if (!is_identifier(s)) reject(name, "is not a valid identifier");
  return s;
}

Model read_model(const Value& value) {
  const Value& v = as_object(value, "model");
  exact(v, "model", {"id", "revision", "artifact_digest", "adapter_id", "quantization"});
  Model m;
  m.id = as_identifier(field(v, "id"), "model.id");
  m.revision = as_string(field(v, "revision"), "model.revision", 1, 64);
  if (!lower_hex(m.revision, 40) && !lower_hex(m.revision, 64)) {
    reject("model.revision must be a 40- or 64-character lowercase hex revision");
  }
  m.artifact_digest = as_string(field(v, "artifact_digest"), "model.artifact_digest", 1, 71);
  if (!m.artifact_digest.starts_with("sha256:") || !lower_hex(std::string_view(m.artifact_digest).substr(7), 64)) {
    reject("model.artifact_digest must be sha256:<64 lowercase hex>");
  }
  m.adapter_id = as_string(field(v, "adapter_id"), "model.adapter_id", 1, 64);
  if (!is_adapter(m.adapter_id)) reject("model.adapter_id is invalid");
  m.quantization = as_string(field(v, "quantization"), "model.quantization", 1, 64);
  return m;
}

Sampling read_sampling(const Value& value) {
  const Value& v = as_object(value, "sampling");
  exact(v, "sampling", {"temperature", "top_p", "seed", "stop"});
  Sampling s;
  s.temperature = as_number(field(v, "temperature"), "sampling.temperature", 0, 2);
  s.top_p = as_number(field(v, "top_p"), "sampling.top_p", 0, 1, Minimum::Exclusive);
  const Value& seed = field(v, "seed");
  if (seed.kind != Value::Kind::Null) {
    s.seed = as_integer(seed, "sampling.seed", std::numeric_limits<int64_t>::min());
  }
  for (const Value& item : as_array(field(v, "stop"), "sampling.stop", 4)) {
    s.stop.push_back(as_string(item, "sampling.stop[]", 1, 128));
  }
  if (!unique(s.stop)) reject("sampling.stop must be unique");
  return s;
}

Objective read_objective(const Value& value) {
  const Value& v = as_object(value, "objective");
  exact(v, "objective", {"workload_class", "context_tokens", "max_output_tokens", "batch_size"},
        {"target_ttft_ms", "min_decode_tokens_per_second"});
  Objective o;
  o.workload_class = as_enum(field(v, "workload_class"), {"interactive", "throughput", "background"},
                             "objective.workload_class is invalid");
  o.context_tokens = as_integer(field(v, "context_tokens"), "objective.context_tokens", 1, 131072);
  o.max_output_tokens = as_integer(field(v, "max_output_tokens"), "objective.max_output_tokens", 1, 8192);
  if (!equals_one(field(v, "batch_size"))) reject("objective.batch_size must be 1 in v1");
  if (const Value* target = member(v, "target_ttft_ms")) {
    o.target_ttft_ms = as_number(*target, "objective.target_ttft_ms", 0, kNoMaximum, Minimum::Exclusive);
  }
  if (const Value* rate = member(v, "min_decode_tokens_per_second")) {
    o.min_decode_tokens_per_second =
        as_number(*rate, "objective.min_decode_tokens_per_second", 0, kNoMaximum, Minimum::Exclusive);
  }
  return o;
}

Memory read_memory(const Value& value) {
  const Value& v = as_object(value, "memory");
  exact(v, "memory", {"max_runtime_memory_bytes", "ssd_offload"}, {"max_resident_weight_bytes"});
  Memory m;
  m.max_runtime_memory_bytes =
      as_integer(field(v, "max_runtime_memory_bytes"), "memory.max_runtime_memory_bytes", kMinRuntimeMemoryBytes);
  if (const Value* resident = member(v, "max_resident_weight_bytes")) {
    m.max_resident_weight_bytes = as_integer(*resident, "memory.max_resident_weight_bytes", 1);
    if (*m.max_resident_weight_bytes > m.max_runtime_memory_bytes) {
      reject("resident weight ceiling cannot exceed runtime memory ceiling");
    }
  }
  m.ssd_offload = as_enum(field(v, "ssd_offload"), {"disabled"}, "ssd_offload must be disabled in v1");
  return m;
}

Start read_start(const Value& v) {
  exact(v, "start",
        {"schema_version", "message_type", "request_id", "task_id", "operation", "runtime_revision",
         "created_at_unix_ms", "deadline_unix_ms", "model", "messages", "sampling", "objective", "memory",
         "deployment_mode"});
  Start s;
  s.request_id = as_identifier(field(v, "request_id"), "request_id");
  s.task_id = as_identifier(field(v, "task_id"), "task_id");
  s.operation = as_enum(field(v, "operation"), {"llm.generate"}, "operation must be llm.generate");
  s.runtime_revision = as_string(field(v, "runtime_revision"), "runtime_revision", 1, 64);
  if (!lower_hex(s.runtime_revision, 64)) reject("runtime_revision must be 64 lowercase hex characters");
  s.created_at_unix_ms = as_integer(field(v, "created_at_unix_ms"), "created_at_unix_ms", 0);
  s.deadline_unix_ms = as_integer(field(v, "deadline_unix_ms"), "deadline_unix_ms", 1);
  // Both are non-negative, so the difference cannot overflow.
  if (s.deadline_unix_ms <= s.created_at_unix_ms || s.deadline_unix_ms - s.created_at_unix_ms > kMaxDeadlineMs) {
    reject("deadline must be after creation and no more than 24 hours later");
  }
  s.model = read_model(field(v, "model"));
  const std::vector<Value>& messages = as_array(field(v, "messages"), "messages", 128, 1);
  size_t content = 0;
  for (size_t i = 0; i < messages.size(); ++i) {
    const std::string name = "messages[" + std::to_string(i) + "]";
    const Value& raw = as_object(messages[i], name);
    exact(raw, name, {"role", "content"});
    Message& message = s.messages.emplace_back();
    message.role = as_enum(field(raw, "role"), {"system", "user", "assistant"}, name + ".role is unsupported in v1");
    message.content = as_string(field(raw, "content"), name + ".content", 1, 65536);
    content += code_points(message.content);
  }
  if (content > kMaxCombinedContent) reject("combined message content exceeds 524288 characters");
  s.sampling = read_sampling(field(v, "sampling"));
  s.objective = read_objective(field(v, "objective"));
  s.memory = read_memory(field(v, "memory"));
  s.deployment_mode = as_enum(field(v, "deployment_mode"), {"supported"}, "deployment_mode must be supported in v1");
  return s;
}

Control read_control(const Value& v) {
  exact(v, "control", {"schema_version", "message_type", "control_id", "action", "issued_at_unix_ms"},
        {"target_request_id"});
  Control c;
  c.control_id = as_identifier(field(v, "control_id"), "control_id");
  c.issued_at_unix_ms = as_integer(field(v, "issued_at_unix_ms"), "issued_at_unix_ms", 0);
  c.action = as_enum(field(v, "action"), {"cancel", "drain"}, "control.action is invalid");
  const Value* target = member(v, "target_request_id");
  if (c.action == "cancel") {
    if (!target) reject("cancel requires target_request_id");
    c.target_request_id = as_identifier(*target, "target_request_id");
  } else if (target) {
    reject("drain must not include target_request_id");
  }
  return c;
}

// Optional event members as bits, and the members each kind allows and requires.
constexpr std::string_view kEventFields[] = {"progress",       "text_delta",  "generated_tokens_delta",
                                             "queue_position", "queue_depth", "message"};
enum : unsigned { kProgress = 1, kTextDelta = 2, kTokens = 4, kQueuePosition = 8, kQueueDepth = 16, kMessage = 32 };
struct EventKind {
  std::string_view name;
  unsigned allowed, required;
};
constexpr EventKind kEventKinds[] = {
    {"accepted", kMessage, 0},
    {"progress", kProgress | kMessage, kProgress},
    {"text_delta", kTextDelta | kTokens, kTextDelta | kTokens},
    {"queue_state", kQueuePosition | kQueueDepth | kMessage, kQueuePosition | kQueueDepth},
    {"warning", kMessage, kMessage},
};

Event read_event(const Value& v) {
  exact(v, "event", {"schema_version", "message_type", "request_id", "sequence", "kind", "emitted_at_unix_ms"},
        {"progress", "text_delta", "generated_tokens_delta", "queue_position", "queue_depth", "message"});
  Event e;
  e.request_id = as_identifier(field(v, "request_id"), "request_id");
  e.sequence = as_integer(field(v, "sequence"), "sequence", 0);
  e.emitted_at_unix_ms = as_integer(field(v, "emitted_at_unix_ms"), "emitted_at_unix_ms", 0);
  const Value& kind = field(v, "kind");
  const auto rule = std::find_if(std::begin(kEventKinds), std::end(kEventKinds), [&](const EventKind& k) {
    return kind.kind == Value::Kind::String && kind.text == k.name;
  });
  if (rule == std::end(kEventKinds)) reject("event.kind is invalid");
  e.kind = kind.text;
  unsigned present = 0;
  for (size_t i = 0; i < std::size(kEventFields); ++i) {
    if (member(v, kEventFields[i])) present |= 1u << i;
  }
  if (present & ~rule->allowed) reject("event fields are invalid for kind " + e.kind);
  if (rule->required & ~present) reject("event kind " + e.kind + " is missing required fields");
  if (const Value* x = member(v, "progress")) e.progress = as_number(*x, "progress", 0, 1);
  if (const Value* x = member(v, "text_delta")) e.text_delta = as_string(*x, "text_delta", 1, 65536);
  if (const Value* x = member(v, "generated_tokens_delta")) {
    e.generated_tokens_delta = as_integer(*x, "generated_tokens_delta", 0, 8192);
  }
  if (const Value* x = member(v, "queue_position")) e.queue_position = as_integer(*x, "queue_position", 0);
  if (const Value* x = member(v, "queue_depth")) e.queue_depth = as_integer(*x, "queue_depth", 0);
  if (const Value* x = member(v, "message")) e.message = as_string(*x, "message", 1, 1024);
  return e;
}

Usage read_usage(const Value& value) {
  const Value& v = as_object(value, "usage");
  exact(v, "usage", {"prompt_tokens", "completion_tokens", "total_tokens"});
  Usage u;
  u.prompt_tokens = as_integer(field(v, "prompt_tokens"), "usage.prompt_tokens", 0);
  u.completion_tokens = as_integer(field(v, "completion_tokens"), "usage.completion_tokens", 0);
  u.total_tokens = as_integer(field(v, "total_tokens"), "usage.total_tokens", 0);
  // All three are non-negative, so this subtraction cannot overflow where the sum could.
  if (u.total_tokens - u.prompt_tokens != u.completion_tokens) {
    reject("usage.total_tokens must equal prompt_tokens + completion_tokens");
  }
  return u;
}

Metrics read_metrics(const Value& value) {
  const Value& v = as_object(value, "metrics");
  exact(v, "metrics",
        {"queue_ms", "ttft_ms", "inter_token_latency_ms", "decode_tokens_per_second", "peak_memory_bytes",
         "maximum_batch_size"});
  const auto nullable = [&](const char* key, Minimum bound) -> std::optional<double> {
    const Value& x = field(v, key);
    if (x.kind == Value::Kind::Null) return std::nullopt;
    return as_number(x, std::string("metrics.") + key, 0, kNoMaximum, bound);
  };
  Metrics m;
  m.queue_ms = as_number(field(v, "queue_ms"), "metrics.queue_ms", 0);
  m.ttft_ms = nullable("ttft_ms", Minimum::Inclusive);
  m.inter_token_latency_ms = nullable("inter_token_latency_ms", Minimum::Inclusive);
  m.decode_tokens_per_second = nullable("decode_tokens_per_second", Minimum::Exclusive);
  m.peak_memory_bytes = as_integer(field(v, "peak_memory_bytes"), "metrics.peak_memory_bytes", 0);
  m.maximum_batch_size = as_integer(field(v, "maximum_batch_size"), "metrics.maximum_batch_size", 1);
  return m;
}

Execution read_execution(const Value& value) {
  const Value& v = as_object(value, "execution");
  exact(v, "execution", {"backend", "route", "plan_id", "evidence_ids", "fallback_used"}, {"fallback_reason"});
  Execution e;
  e.backend = as_enum(field(v, "backend"), {"mlx", "metal", "coreml", "ane", "cpu", "hybrid"},
                      "execution.backend is invalid");
  e.route = as_enum(field(v, "route"), {"batch_generator", "direct_single_sequence", "native_phase_program"},
                    "execution.route is invalid");
  e.plan_id = as_identifier(field(v, "plan_id"), "execution.plan_id");
  for (const Value& item : as_array(field(v, "evidence_ids"), "execution.evidence_ids", 16)) {
    e.evidence_ids.push_back(as_identifier(item, "execution.evidence_ids[]"));
  }
  if (!unique(e.evidence_ids)) reject("execution.evidence_ids must be unique");
  e.fallback_used = as_bool(field(v, "fallback_used"), "execution.fallback_used");
  const Value* reason = member(v, "fallback_reason");
  if (e.fallback_used) {
    if (!reason) reject("fallback_reason is required when fallback_used is true");
    e.fallback_reason = as_string(*reason, "execution.fallback_reason", 1, 512);
  } else if (reason) {
    reject("fallback_reason is forbidden when fallback_used is false");
  }
  return e;
}

Failure read_failure(const Value& value) {
  const Value& v = as_object(value, "failure");
  exact(v, "failure", {"code", "stage", "retryable", "message"});
  Failure f;
  f.code = as_identifier(field(v, "code"), "failure.code");
  f.stage = as_enum(field(v, "stage"), {"not_started", "running", "partial_output"}, "failure.stage is invalid");
  f.retryable = as_bool(field(v, "retryable"), "failure.retryable");
  f.message = as_string(field(v, "message"), "failure.message", 1, 1024);
  return f;
}

// verify_kit's allowed_finish table.
bool consistent(const Value& status, const Value& finish) {
  struct Pair {
    std::string_view status, finish;
  };
  constexpr Pair kAllowed[] = {{"completed", "stop"},
                               {"completed", "length"},
                               {"failed", "error"},
                               {"cancelled", "cancelled"},
                               {"rejected", "rejected"}};
  return status.kind == Value::Kind::String && finish.kind == Value::Kind::String &&
         std::any_of(std::begin(kAllowed), std::end(kAllowed),
                     [&](const Pair& p) { return status.text == p.status && finish.text == p.finish; });
}

Receipt read_receipt(const Value& v) {
  exact(v, "receipt",
        {"schema_version", "message_type", "request_id", "terminal_sequence", "status", "finish_reason",
         "completed_at_unix_ms", "usage", "metrics", "execution"},
        {"output_text", "failure"});
  Receipt r;
  r.request_id = as_identifier(field(v, "request_id"), "request_id");
  r.terminal_sequence = as_integer(field(v, "terminal_sequence"), "terminal_sequence", 0);
  r.completed_at_unix_ms = as_integer(field(v, "completed_at_unix_ms"), "completed_at_unix_ms", 0);
  if (!consistent(field(v, "status"), field(v, "finish_reason"))) {
    reject("receipt status and finish_reason are inconsistent");
  }
  r.status = field(v, "status").text;
  r.finish_reason = field(v, "finish_reason").text;
  const Value* output = member(v, "output_text");
  const Value* failure = member(v, "failure");
  if (r.status == "completed") {
    if (!output || failure) reject("completed receipt requires output_text and forbids failure");
  } else {
    if (!failure) reject("non-completed receipt requires failure");
    r.failure = read_failure(*failure);
    if (r.status == "rejected" && r.failure->stage != "not_started") {
      reject("rejected receipt failure stage must be not_started");
    }
  }
  if (output) r.output_text = as_string(*output, "output_text", 0, 1048576);
  r.usage = read_usage(field(v, "usage"));
  r.metrics = read_metrics(field(v, "metrics"));
  r.execution = read_execution(field(v, "execution"));
  return r;
}

struct Document {
  Value root;
  MessageType type;
};

// verify_kit.load_document and the envelope checks of validate_document.
Document parse(std::string_view document) {
  // Such a document exceeds every message's limit; refusing it first bounds the parse.
  if (document.size() > kLargestLimit) reject("document exceeds the largest message limit");
  std::string_view text = document;
  if (text.starts_with("\xEF\xBB\xBF")) text.remove_prefix(3);  // Python decodes as utf-8-sig
  for (size_t i = 0; i < text.size();) {
    if (static_cast<unsigned char>(text[i]) < 0x80) {
      ++i;
      continue;
    }
    const size_t length = unicode::utf8_sequence_length(text, i);
    if (length == 0) reject("document is not valid UTF-8");
    i += length;
  }
  Value root = Parser(text).document();
  if (root.kind != Value::Kind::Object) reject("wire document must be a JSON object");
  const Value* version = member(root, "schema_version");
  if (!version || !equals_one(*version)) reject("schema_version must be integer 1");
  const Value* type = member(root, "message_type");
  const auto info = std::find_if(std::begin(kTypes), std::end(kTypes), [&](const TypeInfo& t) {
    return type && type->kind == Value::Kind::String && type->text == t.name;
  });
  if (info == std::end(kTypes)) reject("message_type is invalid");
  if (document.size() > info->limit) reject(std::string(info->name) + " exceeds its encoded byte limit");
  return {std::move(root), info->type};
}

Value root_of(std::string_view document, MessageType expected) {
  Document parsed = parse(document);
  if (parsed.type != expected) {
    reject("expected a " + std::string(name(expected)) + " document, not " + std::string(name(parsed.type)));
  }
  return std::move(parsed.root);
}

// Appends a JSON string. Only well-formed UTF-8 is written: the decoders keep
// lone surrogates because the reference accepts them, but Swift rejects them.
void put_string(std::string& out, std::string_view text) {
  constexpr char kHex[] = "0123456789abcdef";
  out.push_back('"');
  for (size_t i = 0; i < text.size();) {
    const auto c = static_cast<unsigned char>(text[i]);
    if (c >= 0x80) {
      const size_t length = unicode::utf8_sequence_length(text, i);
      if (length == 0) reject("cannot encode a string that is not valid UTF-8");
      out.append(text.substr(i, length));
      i += length;
      continue;
    }
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          out += "\\u00";
          out.push_back(kHex[c >> 4]);
          out.push_back(kHex[c & 0xF]);
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
    ++i;
  }
  out.push_back('"');
}

void put_integer(std::string& out, int64_t value) {
  char buffer[24];
  out.append(buffer, std::to_chars(buffer, buffer + sizeof(buffer), value).ptr);
}

void put_number(std::string& out, double value) {
  if (!std::isfinite(value)) reject("cannot encode a non-finite number");
  char buffer[32];
  const std::string_view text(buffer, static_cast<size_t>(std::to_chars(buffer, buffer + sizeof(buffer), value).ptr - buffer));
  out += text;
  // The shortest form of an integral double ("510") reads back as a Python
  // int; keep it a float so the number's JSON type survives a round trip.
  if (text.find_first_of(".e") == std::string_view::npos) out += ".0";
}

void put_strings(std::string& out, const std::vector<std::string>& items) {
  out.push_back('[');
  for (size_t i = 0; i < items.size(); ++i) {
    if (i) out.push_back(',');
    put_string(out, items[i]);
  }
  out.push_back(']');
}

// Writes one JSON object with its members in call order.
class ObjectWriter {
 public:
  explicit ObjectWriter(std::string& out) : out_(out) { out_.push_back('{'); }

  // Starts a member whose value the caller writes to the output.
  std::string& key(std::string_view name) {
    if (!first_) out_.push_back(',');
    first_ = false;
    out_.push_back('"');
    out_ += name;
    out_ += "\":";
    return out_;
  }
  ObjectWriter& string(std::string_view name, std::string_view value) {
    put_string(key(name), value);
    return *this;
  }
  ObjectWriter& integer(std::string_view name, int64_t value) {
    put_integer(key(name), value);
    return *this;
  }
  ObjectWriter& number(std::string_view name, double value) {
    put_number(key(name), value);
    return *this;
  }
  ObjectWriter& boolean(std::string_view name, bool value) {
    key(name) += value ? "true" : "false";
    return *this;
  }
  // A member the schema requires but allows to be null.
  ObjectWriter& nullable(std::string_view name, const std::optional<double>& value) {
    if (value) return number(name, *value);
    key(name) += "null";
    return *this;
  }
  void close() { out_.push_back('}'); }

 private:
  std::string& out_;
  bool first_ = true;
};

void envelope(ObjectWriter& w, MessageType type) {
  w.integer("schema_version", kSchemaVersion).string("message_type", name(type));
}

}  // namespace

std::string_view name(MessageType type) { return kTypes[static_cast<size_t>(type)].name; }

MessageType validate(std::string_view document) {
  const Document parsed = parse(document);
  switch (parsed.type) {
    case MessageType::Start: read_start(parsed.root); break;
    case MessageType::Control: read_control(parsed.root); break;
    case MessageType::Event: read_event(parsed.root); break;
    case MessageType::Receipt: read_receipt(parsed.root); break;
  }
  return parsed.type;
}

Start decode_start(std::string_view document) { return read_start(root_of(document, MessageType::Start)); }
Control decode_control(std::string_view document) { return read_control(root_of(document, MessageType::Control)); }
Event decode_event(std::string_view document) { return read_event(root_of(document, MessageType::Event)); }
Receipt decode_receipt(std::string_view document) { return read_receipt(root_of(document, MessageType::Receipt)); }

std::string encode(const Start& s) {
  std::string out;
  ObjectWriter w(out);
  envelope(w, MessageType::Start);
  w.string("request_id", s.request_id).string("task_id", s.task_id).string("operation", s.operation);
  w.string("runtime_revision", s.runtime_revision).integer("created_at_unix_ms", s.created_at_unix_ms);
  w.integer("deadline_unix_ms", s.deadline_unix_ms);
  w.key("model");
  ObjectWriter model(out);
  model.string("id", s.model.id).string("revision", s.model.revision);
  model.string("artifact_digest", s.model.artifact_digest).string("adapter_id", s.model.adapter_id);
  model.string("quantization", s.model.quantization).close();
  w.key("messages").push_back('[');
  for (size_t i = 0; i < s.messages.size(); ++i) {
    if (i) out.push_back(',');
    ObjectWriter message(out);
    message.string("role", s.messages[i].role).string("content", s.messages[i].content).close();
  }
  out.push_back(']');
  w.key("sampling");
  ObjectWriter sampling(out);
  sampling.number("temperature", s.sampling.temperature).number("top_p", s.sampling.top_p);
  if (s.sampling.seed) {
    sampling.integer("seed", *s.sampling.seed);
  } else {
    sampling.key("seed") += "null";
  }
  put_strings(sampling.key("stop"), s.sampling.stop);
  sampling.close();
  w.key("objective");
  ObjectWriter objective(out);
  objective.string("workload_class", s.objective.workload_class);
  objective.integer("context_tokens", s.objective.context_tokens);
  objective.integer("max_output_tokens", s.objective.max_output_tokens);
  objective.integer("batch_size", s.objective.batch_size);
  if (s.objective.target_ttft_ms) objective.number("target_ttft_ms", *s.objective.target_ttft_ms);
  if (s.objective.min_decode_tokens_per_second) {
    objective.number("min_decode_tokens_per_second", *s.objective.min_decode_tokens_per_second);
  }
  objective.close();
  w.key("memory");
  ObjectWriter memory(out);
  memory.integer("max_runtime_memory_bytes", s.memory.max_runtime_memory_bytes);
  if (s.memory.max_resident_weight_bytes) {
    memory.integer("max_resident_weight_bytes", *s.memory.max_resident_weight_bytes);
  }
  memory.string("ssd_offload", s.memory.ssd_offload).close();
  w.string("deployment_mode", s.deployment_mode).close();
  validate(out);
  // Swift compares strings by canonical equivalence, so its uniqueness check
  // also rejects stop strings that differ only in normalization.
  for (size_t i = 0; i < s.sampling.stop.size(); ++i) {
    for (size_t j = i + 1; j < s.sampling.stop.size(); ++j) {
      if (unicode::nfc(s.sampling.stop[i]) == unicode::nfc(s.sampling.stop[j])) {
        reject("cannot encode sampling.stop strings that are canonically equivalent");
      }
    }
  }
  return out;
}

std::string encode(const Control& c) {
  std::string out;
  ObjectWriter w(out);
  envelope(w, MessageType::Control);
  w.string("control_id", c.control_id).string("action", c.action);
  if (c.target_request_id) w.string("target_request_id", *c.target_request_id);
  w.integer("issued_at_unix_ms", c.issued_at_unix_ms).close();
  validate(out);
  return out;
}

std::string encode(const Event& e) {
  std::string out;
  ObjectWriter w(out);
  envelope(w, MessageType::Event);
  w.string("request_id", e.request_id).integer("sequence", e.sequence).string("kind", e.kind);
  w.integer("emitted_at_unix_ms", e.emitted_at_unix_ms);
  if (e.progress) w.number("progress", *e.progress);
  if (e.text_delta) w.string("text_delta", *e.text_delta);
  if (e.generated_tokens_delta) w.integer("generated_tokens_delta", *e.generated_tokens_delta);
  if (e.queue_position) w.integer("queue_position", *e.queue_position);
  if (e.queue_depth) w.integer("queue_depth", *e.queue_depth);
  if (e.message) w.string("message", *e.message);
  w.close();
  validate(out);
  return out;
}

std::string encode(const Receipt& r) {
  std::string out;
  ObjectWriter w(out);
  envelope(w, MessageType::Receipt);
  w.string("request_id", r.request_id).integer("terminal_sequence", r.terminal_sequence);
  w.string("status", r.status).string("finish_reason", r.finish_reason);
  w.integer("completed_at_unix_ms", r.completed_at_unix_ms);
  if (r.output_text) w.string("output_text", *r.output_text);
  w.key("usage");
  ObjectWriter usage(out);
  usage.integer("prompt_tokens", r.usage.prompt_tokens).integer("completion_tokens", r.usage.completion_tokens);
  usage.integer("total_tokens", r.usage.total_tokens).close();
  w.key("metrics");
  ObjectWriter metrics(out);
  metrics.number("queue_ms", r.metrics.queue_ms).nullable("ttft_ms", r.metrics.ttft_ms);
  metrics.nullable("inter_token_latency_ms", r.metrics.inter_token_latency_ms);
  metrics.nullable("decode_tokens_per_second", r.metrics.decode_tokens_per_second);
  metrics.integer("peak_memory_bytes", r.metrics.peak_memory_bytes);
  metrics.integer("maximum_batch_size", r.metrics.maximum_batch_size).close();
  w.key("execution");
  ObjectWriter execution(out);
  execution.string("backend", r.execution.backend).string("route", r.execution.route);
  execution.string("plan_id", r.execution.plan_id);
  put_strings(execution.key("evidence_ids"), r.execution.evidence_ids);
  execution.boolean("fallback_used", r.execution.fallback_used);
  if (r.execution.fallback_reason) execution.string("fallback_reason", *r.execution.fallback_reason);
  execution.close();
  if (r.failure) {
    w.key("failure");
    ObjectWriter failure(out);
    failure.string("code", r.failure->code).string("stage", r.failure->stage);
    failure.boolean("retryable", r.failure->retryable).string("message", r.failure->message).close();
  }
  w.close();
  validate(out);
  return out;
}

StreamValidator::StreamValidator(std::string request_id) : request_id_(std::move(request_id)) {
  if (!is_identifier(request_id_)) reject("stream request_id is not a valid identifier");
}

void StreamValidator::accept(const Event& event) {
  encode(event);  // throws unless the event is valid on its own
  if (terminated_) reject("stream: event after the terminal receipt");
  if (event.request_id != request_id_) reject("stream: event belongs to another request");
  if (event.sequence != next_sequence_) {
    reject("stream: event sequence " + std::to_string(event.sequence) + " where " +
           std::to_string(next_sequence_) + " was expected");
  }
  if (next_sequence_ == 0 && event.kind != "accepted") reject("stream: the first event must be accepted");
  ++next_sequence_;
}

void StreamValidator::accept(const Receipt& receipt) {
  encode(receipt);  // throws unless the receipt is valid on its own
  if (terminated_) reject("stream: second terminal receipt");
  if (receipt.request_id != request_id_) reject("stream: receipt belongs to another request");
  if (receipt.terminal_sequence != next_sequence_) {
    reject("stream: terminal_sequence " + std::to_string(receipt.terminal_sequence) + " where " +
           std::to_string(next_sequence_) + " was expected");
  }
  ++next_sequence_;
  terminated_ = true;
}

}  // namespace lokahi::wire
