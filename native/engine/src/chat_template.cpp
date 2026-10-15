// Native renderer for Hugging Face chat templates.
//
// transformers renders a chat template with jinja2's ImmutableSandboxedEnvironment
// (trim_blocks, lstrip_blocks, the loopcontrols extension, its own tojson,
// raise_exception and strftime_now, and a {% generation %} tag). This file
// implements the subset of Jinja that model templates use, with jinja2's
// lexing and scoping rules and Python's value semantics, so the output is
// byte-identical. Syntax outside the subset is rejected at parse time;
// operations whose Python result this code cannot reproduce exactly (Unicode
// case mapping, big integers, printf formatting) fail at render time.
#include "chat_template.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "common.h"
#include "json.h"
#include "safetensors.h"  // read_text_file
#include "unicode.h"

namespace lokahi {
namespace {

using Str = std::shared_ptr<const std::string>;

Str make_text(std::string s) { return std::make_shared<const std::string>(std::move(s)); }

template <size_t N>
bool one_of(std::string_view name, const std::string_view (&list)[N]) {
  return std::find(std::begin(list), std::end(list), name) != std::end(list);
}

// ---------------------------------------------------------------------------
// Text. Template strings are Python str values: lengths, indices and searches
// count code points, and whitespace is str.isspace(). Every string is valid
// UTF-8 (inputs are checked), so decoding cannot fail.

bool is_ascii(std::string_view s) {
  for (unsigned char c : s) {
    if (c >= 0x80) return false;
  }
  return true;
}

bool valid_utf8(std::string_view s) {
  for (size_t i = 0; i < s.size();) {
    if (static_cast<unsigned char>(s[i]) < 0x80) {
      ++i;
      continue;
    }
    const size_t n = unicode::utf8_sequence_length(s, i);
    if (n == 0) return false;
    i += n;
  }
  return true;
}

bool py_space(char32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// str.splitlines() boundaries other than "\r\n", which counts once.
bool py_line_break(char32_t c) {
  return c == '\n' || c == '\r' || c == 0x0B || c == 0x0C || c == 0x1C || c == 0x1D || c == 0x1E || c == 0x85 ||
         c == 0x2028 || c == 0x2029;
}

char32_t cp_at(std::string_view s, size_t i, size_t* len) {
  const unsigned char b = static_cast<unsigned char>(s[i]);
  if (b < 0x80) {
    *len = 1;
    return b;
  }
  return unicode::decode_utf8_at(s, i, len);
}

// Start of the code point that ends at byte `end` (> 0).
size_t cp_back(std::string_view s, size_t end) {
  size_t i = end - 1;
  while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) --i;
  return i;
}

size_t cp_count(std::string_view s) {
  size_t n = 0;
  for (unsigned char c : s) n += (c & 0xC0) != 0x80;
  return n;
}

// Code-point indexing over UTF-8; byte offsets are only built for non-ASCII text.
class Text {
 public:
  explicit Text(std::string_view s) : s_(s), ascii_(is_ascii(s)) {
    if (ascii_) return;
    bounds_.reserve(s.size() + 1);
    for (size_t i = 0; i < s.size(); ++i) {
      if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) bounds_.push_back(i);
    }
    bounds_.push_back(s.size());
  }
  int64_t size() const { return static_cast<int64_t>(ascii_ ? s_.size() : bounds_.size() - 1); }
  size_t byte(int64_t i) const { return ascii_ ? static_cast<size_t>(i) : bounds_[static_cast<size_t>(i)]; }
  int64_t index(size_t byte) const {
    if (ascii_) return static_cast<int64_t>(byte);
    return std::lower_bound(bounds_.begin(), bounds_.end(), byte) - bounds_.begin();
  }
  std::string_view sub(int64_t i, int64_t j) const { return s_.substr(byte(i), byte(j) - byte(i)); }
  char32_t at(int64_t i) const {
    size_t len = 0;
    return cp_at(s_, byte(i), &len);
  }

 private:
  std::string_view s_;
  bool ascii_;
  std::vector<size_t> bounds_;
};

// The code points str.strip(chars) removes: whitespace when chars is None.
class CharSet {
 public:
  CharSet() = default;
  explicit CharSet(const std::string& chars) : whitespace_(false), set_(unicode::decode_utf8(chars)) {}
  bool has(char32_t c) const {
    return whitespace_ ? py_space(c) : std::find(set_.begin(), set_.end(), c) != set_.end();
  }

 private:
  bool whitespace_ = true;
  std::u32string set_;
};

std::string_view py_strip(std::string_view s, const CharSet& set, bool left, bool right) {
  size_t begin = 0, end = s.size(), len = 0;
  if (left) {
    while (begin < end && set.has(cp_at(s, begin, &len))) begin += len;
  }
  if (right) {
    while (end > begin) {
      const size_t start = cp_back(s, end);
      if (!set.has(cp_at(s, start, &len))) break;
      end = start;
    }
  }
  return s.substr(begin, end - begin);
}

// Non-ASCII code points outside Lu, Ll, Lt and Nl that Python's case functions
// change or treat as cased (Python 3.13, Unicode 15.1).
constexpr std::pair<char32_t, char32_t> kCasedExtra[] = {
    {0xAA, 0xAA},       {0xBA, 0xBA},       {0x2B0, 0x2B8},     {0x2C0, 0x2C1},     {0x2E0, 0x2E4},
    {0x345, 0x345},     {0x37A, 0x37A},     {0x10FC, 0x10FC},   {0x1D2C, 0x1D6A},   {0x1D78, 0x1D78},
    {0x1D9B, 0x1DBF},   {0x2071, 0x2071},   {0x207F, 0x207F},   {0x2090, 0x209C},   {0x24B6, 0x24E9},
    {0x2C7C, 0x2C7D},   {0xA69C, 0xA69D},   {0xA770, 0xA770},   {0xA7F2, 0xA7F4},   {0xA7F8, 0xA7F9},
    {0xAB5C, 0xAB5F},   {0xAB69, 0xAB69},   {0x10780, 0x10780}, {0x10783, 0x10785}, {0x10787, 0x107B0},
    {0x107B2, 0x107BA}, {0x1E030, 0x1E06D}, {0x1F130, 0x1F149}, {0x1F150, 0x1F169}, {0x1F170, 0x1F189},
};

bool maybe_cased(char32_t c) {
  using namespace unicode;
  if (in(c, bit(Lu) | bit(Ll) | bit(Lt) | bit(Nl))) return true;
  for (const auto& [first, last] : kCasedExtra) {
    if (c < first) return false;
    if (c <= last) return true;
  }
  return false;
}

// A render-time error; render() adds the template line.
struct RuntimeFailure {
  std::string message;
};
[[noreturn]] void runtime_fail(std::string message) { throw RuntimeFailure{std::move(message)}; }

// Case mapping is implemented for ASCII; non-ASCII text that Python's full
// Unicode case mapping could change fails rather than render differently.
void require_caseless(std::string_view s) {
  if (is_ascii(s)) return;
  for (size_t i = 0, len = 0; i < s.size(); i += len) {
    const char32_t c = cp_at(s, i, &len);
    if (c >= 0x80 && maybe_cased(c)) runtime_fail("Unicode case mapping of non-ASCII letters is not supported");
  }
}

char lower_ascii(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : c; }
char upper_ascii(char c) { return c >= 'a' && c <= 'z' ? static_cast<char>(c - 32) : c; }
bool alpha_ascii(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

std::string py_lower(std::string_view s) {
  require_caseless(s);
  std::string out(s);
  for (char& c : out) c = lower_ascii(c);
  return out;
}

std::string py_upper(std::string_view s) {
  require_caseless(s);
  std::string out(s);
  for (char& c : out) c = upper_ascii(c);
  return out;
}

// str.title(): a cased character after a cased one is lowered, otherwise uppered.
std::string py_title(std::string_view s) {
  require_caseless(s);
  std::string out(s);
  bool previous_cased = false;
  for (char& c : out) {
    const bool cased = alpha_ascii(c);
    if (cased) c = previous_cased ? lower_ascii(c) : upper_ascii(c);
    previous_cased = cased;
  }
  return out;
}

std::string py_capitalize(std::string_view s) {
  require_caseless(s);
  std::string out(s);
  for (size_t i = 0; i < out.size(); ++i) out[i] = i == 0 ? upper_ascii(out[i]) : lower_ascii(out[i]);
  return out;
}

// jinja2's |title: the first character of every run between [-\s({[<]
// separators is uppered and the rest lowered.
std::string jinja_title(std::string_view s) {
  require_caseless(s);
  std::string out;
  out.reserve(s.size());
  bool start = true;
  for (size_t i = 0, len = 0; i < s.size(); i += len) {
    const char32_t c = cp_at(s, i, &len);
    const bool separator = c == '-' || c == '(' || c == '{' || c == '[' || c == '<' || py_space(c);
    if (len == 1) {
      out += separator ? s[i] : (start ? upper_ascii(s[i]) : lower_ascii(s[i]));
    } else {
      out.append(s.substr(i, len));
    }
    start = separator;
  }
  return out;
}

std::string html_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '\'': out += "&#39;"; break;
      case '"': out += "&#34;"; break;
      default: out += c;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Values: the Python objects a template sees.

enum class Kind : uint8_t {
  Undefined, None, Bool, Int, Float, String, List, Dict, Namespace, Macro, Method, Function, Loop,
};
// Python sequence types that share the list representation.
enum class Flavor : uint8_t { List, Tuple, Range, Items, Keys, Values, Iterator };
enum class Builtin : uint8_t { Range, Dict, Namespace, RaiseException, StrftimeNow };
// Why a value is undefined, for the error raised when it is used.
enum class Missing : uint8_t { Name, Hint, Attribute, Key };

struct Object {
  virtual ~Object() = default;
};

struct Value {
  Kind kind = Kind::Undefined;
  uint8_t aux = 0;  // String: 1 for markupsafe.Markup. Undefined: Missing.
  union {
    int64_t integer = 0;  // Int, Function (Builtin)
    double number;
    bool boolean;
    const char* owner;  // Undefined attribute or key: the owner's type name
  };
  Str text;                        // String; Undefined name, key repr or hint
  std::shared_ptr<Object> object;  // List, Dict, Namespace, Macro, Method, Loop
};

struct ListObject final : Object {
  std::vector<Value> items;
  Flavor flavor = Flavor::List;
  size_t cursor = 0;                      // Iterator: items before it were consumed
  int64_t start = 0, stop = 0, step = 1;  // Range
};

struct DictObject final : Object {
  std::vector<std::pair<Value, Value>> items;  // insertion order
};

struct NamespaceObject final : Object {
  std::vector<std::pair<std::string, Value>> attrs;
};

struct Scope;
struct MacroDef;

struct MacroObject final : Object {
  const MacroDef* def = nullptr;
  // Weak, because a macro usually lives in the scope it closes over.
  std::weak_ptr<Scope> closure;
};

struct MethodObject final : Object {
  Value self;
  std::string name;
};

// jinja2's LoopContext; one object advances through the iterations.
struct LoopObject final : Object {
  std::vector<Value> items;
  size_t index = 0;
};

ListObject& as_list(const Value& v) { return static_cast<ListObject&>(*v.object); }
DictObject& as_dict(const Value& v) { return static_cast<DictObject&>(*v.object); }
NamespaceObject& as_namespace(const Value& v) { return static_cast<NamespaceObject&>(*v.object); }

Value none_value() {
  Value v;
  v.kind = Kind::None;
  return v;
}

Value bool_value(bool b) {
  Value v;
  v.kind = Kind::Bool;
  v.boolean = b;
  return v;
}

Value int_value(int64_t i) {
  Value v;
  v.kind = Kind::Int;
  v.integer = i;
  return v;
}

Value float_value(double d) {
  Value v;
  v.kind = Kind::Float;
  v.number = d;
  return v;
}

Value str_value(Str s, bool markup = false) {
  Value v;
  v.kind = Kind::String;
  v.aux = markup;
  v.text = std::move(s);
  return v;
}

Value str_value(std::string s, bool markup = false) { return str_value(make_text(std::move(s)), markup); }

Value list_value(std::vector<Value> items, Flavor flavor = Flavor::List) {
  auto list = std::make_shared<ListObject>();
  list->items = std::move(items);
  list->flavor = flavor;
  Value v;
  v.kind = Kind::List;
  v.object = std::move(list);
  return v;
}

Value dict_value(std::vector<std::pair<Value, Value>> items) {
  auto dict = std::make_shared<DictObject>();
  dict->items = std::move(items);
  Value v;
  v.kind = Kind::Dict;
  v.object = std::move(dict);
  return v;
}

Value object_value(Kind kind, std::shared_ptr<Object> object) {
  Value v;
  v.kind = kind;
  v.object = std::move(object);
  return v;
}

Value builtin_value(Builtin b) {
  Value v;
  v.kind = Kind::Function;
  v.integer = static_cast<int64_t>(b);
  return v;
}

Value undefined_value(Missing why, Str text, const char* owner = nullptr) {
  Value v;
  v.aux = static_cast<uint8_t>(why);
  v.text = std::move(text);
  v.owner = owner;
  return v;
}

Value undefined_hint(std::string hint) { return undefined_value(Missing::Hint, make_text(std::move(hint))); }

bool is_markup(const Value& v) { return v.kind == Kind::String && v.aux; }
bool is_number(const Value& v) { return v.kind == Kind::Bool || v.kind == Kind::Int || v.kind == Kind::Float; }
bool is_int_like(const Value& v) { return v.kind == Kind::Bool || v.kind == Kind::Int; }
int64_t as_int(const Value& v) { return v.kind == Kind::Bool ? (v.boolean ? 1 : 0) : v.integer; }
double as_float(const Value& v) { return v.kind == Kind::Float ? v.number : static_cast<double>(as_int(v)); }
bool is_list_or_tuple(const Value& v) {
  return v.kind == Kind::List && (as_list(v).flavor == Flavor::List || as_list(v).flavor == Flavor::Tuple);
}

const char* type_name(const Value& v) {
  switch (v.kind) {
    case Kind::Undefined: return "Undefined";
    case Kind::None: return "NoneType";
    case Kind::Bool: return "bool";
    case Kind::Int: return "int";
    case Kind::Float: return "float";
    case Kind::String: return v.aux ? "Markup" : "str";
    case Kind::List:
      switch (as_list(v).flavor) {
        case Flavor::List: return "list";
        case Flavor::Tuple: return "tuple";
        case Flavor::Range: return "range";
        case Flavor::Items: return "dict_items";
        case Flavor::Keys: return "dict_keys";
        case Flavor::Values: return "dict_values";
        case Flavor::Iterator: return "generator";
      }
      return "list";
    case Kind::Dict: return "dict";
    case Kind::Namespace: return "Namespace";
    case Kind::Macro: return "Macro";
    case Kind::Method: return "builtin_function_or_method";
    case Kind::Function: return "function";
    case Kind::Loop: return "LoopContext";
  }
  return "object";
}

std::string undefined_message(const Value& v) {
  switch (static_cast<Missing>(v.aux)) {
    case Missing::Name: return "'" + *v.text + "' is undefined";
    case Missing::Hint: return *v.text;
    case Missing::Attribute: return "'" + std::string(v.owner) + " object' has no attribute '" + *v.text + "'";
    case Missing::Key: return "'" + std::string(v.owner) + " object' has no attribute " + *v.text;
  }
  return "undefined value";
}

[[noreturn]] void fail_undefined(const Value& v) { runtime_fail(undefined_message(v)); }

// ---------------------------------------------------------------------------
// str() and repr().

void append_int(std::string& out, int64_t i) {
  char buffer[24];
  const int n = std::snprintf(buffer, sizeof(buffer), "%lld", static_cast<long long>(i));
  out.append(buffer, static_cast<size_t>(n));
}

// Adds one unit in the last place of a decimal digit string; false when the
// digit count would change.
bool step_digits(std::string& digits, bool up) {
  for (size_t i = digits.size(); i-- > 0;) {
    if (up && digits[i] < '9') {
      ++digits[i];
      return true;
    }
    if (!up && digits[i] > '0') {
      --digits[i];
      return digits[0] != '0';
    }
    digits[i] = up ? '0' : '9';
  }
  return false;
}

// The shortest digits that round-trip to v > 0, as Python's repr chooses
// them, and the position of the decimal point relative to the first digit.
void shortest_digits(double v, std::string* digits, int* point) {
  char buffer[64];
  auto split = [](const char* text, std::string* mantissa) {
    mantissa->clear();
    const char* p = text;
    for (; *p && *p != 'e'; ++p) {
      if (*p != '.') mantissa->push_back(*p);
    }
    return std::atoi(p + 1);
  };
  for (int precision = 0; precision <= 16; ++precision) {
    std::snprintf(buffer, sizeof(buffer), "%.*e", precision, v);
    std::string mantissa;
    const int exponent = split(buffer, &mantissa);
    const double back = std::strtod(buffer, nullptr);
    bool found = back == v;
    // Where the rounding interval is asymmetric (at powers of two) the
    // correctly rounded candidate can miss while its neighbour hits.
    if (!found && step_digits(mantissa, back < v)) {
      const std::string candidate = mantissa.substr(0, 1) + "." + mantissa.substr(1) + "e" + std::to_string(exponent);
      found = std::strtod(candidate.c_str(), nullptr) == v;
    }
    if (found) {
      while (mantissa.size() > 1 && mantissa.back() == '0') mantissa.pop_back();
      *digits = mantissa;
      *point = exponent + 1;
      return;
    }
  }
  std::snprintf(buffer, sizeof(buffer), "%.16e", v);
  *point = split(buffer, digits) + 1;
}

// float.__repr__.
void append_float(std::string& out, double d) {
  if (std::isnan(d)) {
    out += "nan";
    return;
  }
  if (std::isinf(d)) {
    out += d < 0 ? "-inf" : "inf";
    return;
  }
  if (d == 0) {
    out += std::signbit(d) ? "-0.0" : "0.0";
    return;
  }
  std::string digits;
  int point = 0;
  shortest_digits(std::fabs(d), &digits, &point);
  if (d < 0) out += '-';
  const int n = static_cast<int>(digits.size());
  if (point <= -4 || point > 16) {
    out += digits[0];
    if (n > 1) out += "." + digits.substr(1);
    const int e = point - 1;
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "e%c%02d", e < 0 ? '-' : '+', std::abs(e));
    out += buffer;
  } else if (point <= 0) {
    out += "0.";
    out.append(static_cast<size_t>(-point), '0');
    out += digits;
  } else if (point >= n) {
    out += digits;
    out.append(static_cast<size_t>(point - n), '0');
    out += ".0";
  } else {
    out += digits.substr(0, static_cast<size_t>(point)) + "." + digits.substr(static_cast<size_t>(point));
  }
}

void append_hex_escape(std::string& out, char32_t c) {
  char buffer[16];
  if (c < 0x100) {
    std::snprintf(buffer, sizeof(buffer), "\\x%02x", static_cast<unsigned>(c));
  } else if (c < 0x10000) {
    std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned>(c));
  } else {
    std::snprintf(buffer, sizeof(buffer), "\\U%08x", static_cast<unsigned>(c));
  }
  out += buffer;
}

// str.isprintable() for one non-ASCII code point.
bool printable(char32_t c) {
  using namespace unicode;
  return !in(c, bit(Cc) | bit(Cf) | bit(Cs) | bit(Co) | bit(Cn) | bit(Zl) | bit(Zp) | bit(Zs));
}

void append_repr_string(std::string& out, std::string_view s) {
  const char quote = s.find('\'') != std::string_view::npos && s.find('"') == std::string_view::npos ? '"' : '\'';
  out += quote;
  for (size_t i = 0, len = 0; i < s.size(); i += len) {
    const char32_t c = cp_at(s, i, &len);
    if (c == static_cast<char32_t>(quote) || c == '\\') {
      out += '\\';
      out += static_cast<char>(c);
    } else if (c == '\t') {
      out += "\\t";
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\r') {
      out += "\\r";
    } else if (c < 0x20 || c == 0x7F) {
      append_hex_escape(out, c);
    } else if (c < 0x80 || printable(c)) {
      out.append(s.substr(i, len));
    } else {
      append_hex_escape(out, c);
    }
  }
  out += quote;
}

void append_repr(std::string& out, const Value& v);

void append_items(std::string& out, const std::vector<Value>& items, const char* open, const char* close) {
  out += open;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i) out += ", ";
    append_repr(out, items[i]);
  }
  out += close;
}

void append_repr(std::string& out, const Value& v) {
  switch (v.kind) {
    case Kind::Undefined: out += "Undefined"; return;
    case Kind::None: out += "None"; return;
    case Kind::Bool: out += v.boolean ? "True" : "False"; return;
    case Kind::Int: append_int(out, v.integer); return;
    case Kind::Float: append_float(out, v.number); return;
    case Kind::String:
      if (v.aux) out += "Markup(";
      append_repr_string(out, *v.text);
      if (v.aux) out += ")";
      return;
    case Kind::List: {
      const ListObject& list = as_list(v);
      switch (list.flavor) {
        case Flavor::List: append_items(out, list.items, "[", "]"); return;
        case Flavor::Tuple: append_items(out, list.items, "(", list.items.size() == 1 ? ",)" : ")"); return;
        case Flavor::Range:
          out += "range(";
          append_int(out, list.start);
          out += ", ";
          append_int(out, list.stop);
          if (list.step != 1) {
            out += ", ";
            append_int(out, list.step);
          }
          out += ")";
          return;
        case Flavor::Items: append_items(out, list.items, "dict_items([", "])"); return;
        case Flavor::Keys: append_items(out, list.items, "dict_keys([", "])"); return;
        case Flavor::Values: append_items(out, list.items, "dict_values([", "])"); return;
        case Flavor::Iterator: runtime_fail("printing a generator is not supported");
      }
      return;
    }
    case Kind::Dict: {
      out += "{";
      bool first = true;
      for (const auto& [key, value] : as_dict(v).items) {
        if (!first) out += ", ";
        first = false;
        append_repr(out, key);
        out += ": ";
        append_repr(out, value);
      }
      out += "}";
      return;
    }
    case Kind::Namespace: {
      out += "<Namespace {";
      bool first = true;
      for (const auto& [key, value] : as_namespace(v).attrs) {
        if (!first) out += ", ";
        first = false;
        append_repr_string(out, key);
        out += ": ";
        append_repr(out, value);
      }
      out += "}>";
      return;
    }
    case Kind::Macro: break;  // Needs the program's names; see Renderer::append_str.
    case Kind::Method:
    case Kind::Function:
    case Kind::Loop: break;
  }
  runtime_fail(std::string("printing a ") + type_name(v) + " object is not supported");
}

// ---------------------------------------------------------------------------
// Comparison, truthiness and iteration.

// Exact three-way comparison of an int with a finite or infinite double.
int compare_int_float(int64_t i, double f) {
  if (f >= 9223372036854775808.0) return -1;
  if (f < -9223372036854775808.0) return 1;
  const double t = std::trunc(f);
  const int64_t ti = static_cast<int64_t>(t);
  if (i != ti) return i < ti ? -1 : 1;
  const double fraction = f - t;
  return fraction > 0 ? -1 : (fraction < 0 ? 1 : 0);
}

// Three-way numeric comparison; 2 when unordered (NaN).
int compare_numbers(const Value& a, const Value& b) {
  if (a.kind != Kind::Float && b.kind != Kind::Float) {
    const int64_t x = as_int(a), y = as_int(b);
    return (x > y) - (x < y);
  }
  if (a.kind == Kind::Float && b.kind == Kind::Float) {
    if (std::isnan(a.number) || std::isnan(b.number)) return 2;
    return (a.number > b.number) - (a.number < b.number);
  }
  if (a.kind == Kind::Float) return std::isnan(a.number) ? 2 : -compare_int_float(as_int(b), a.number);
  return std::isnan(b.number) ? 2 : compare_int_float(as_int(a), b.number);
}

bool py_equal(const Value& a, const Value& b);

const Value* dict_find(const Value& dict, const Value& key) {
  for (const auto& [k, v] : as_dict(dict).items) {
    if (py_equal(k, key)) return &v;
  }
  return nullptr;
}

const Value* dict_find_str(const Value& dict, std::string_view key) {
  for (const auto& [k, v] : as_dict(dict).items) {
    if (k.kind == Kind::String && *k.text == key) return &v;
  }
  return nullptr;
}

bool py_equal(const Value& a, const Value& b) {
  if (is_number(a) && is_number(b)) return compare_numbers(a, b) == 0;
  if (a.kind != b.kind) return false;
  switch (a.kind) {
    case Kind::Undefined:
    case Kind::None: return true;
    case Kind::String: return *a.text == *b.text;
    case Kind::List: {
      const ListObject& x = as_list(a);
      const ListObject& y = as_list(b);
      if (&x == &y) return true;
      if (x.flavor != y.flavor) return false;
      if (x.flavor == Flavor::Iterator || x.flavor == Flavor::Values) return false;  // compared by identity
      if (x.items.size() != y.items.size()) return false;
      if (x.flavor == Flavor::Keys || x.flavor == Flavor::Items) {  // set semantics
        for (const Value& item : x.items) {
          if (std::none_of(y.items.begin(), y.items.end(), [&](const Value& other) { return py_equal(item, other); }))
            return false;
        }
        return true;
      }
      for (size_t i = 0; i < x.items.size(); ++i) {
        if (!py_equal(x.items[i], y.items[i])) return false;
      }
      return true;
    }
    case Kind::Dict: {
      if (as_dict(a).items.size() != as_dict(b).items.size()) return false;
      for (const auto& [key, value] : as_dict(a).items) {
        const Value* other = dict_find(b, key);
        if (!other || !py_equal(value, *other)) return false;
      }
      return true;
    }
    case Kind::Function: return a.integer == b.integer;
    default: return a.object == b.object;
  }
}

enum class CmpOp : uint8_t { Eq, Ne, Lt, Le, Gt, Ge, In, NotIn };

bool ordered(CmpOp op, int c) {
  switch (op) {
    case CmpOp::Lt: return c < 0;
    case CmpOp::Le: return c <= 0;
    case CmpOp::Gt: return c > 0;
    case CmpOp::Ge: return c >= 0;
    default: return false;
  }
}

const char* op_symbol(CmpOp op) {
  switch (op) {
    case CmpOp::Lt: return "<";
    case CmpOp::Le: return "<=";
    case CmpOp::Gt: return ">";
    case CmpOp::Ge: return ">=";
    default: return "==";
  }
}

// <, <=, > and >= with Python's rules: numbers, strings, and lists or tuples
// compared at their first unequal element.
bool py_order(CmpOp op, const Value& a, const Value& b) {
  if (is_number(a) && is_number(b)) {
    const int c = compare_numbers(a, b);
    return c != 2 && ordered(op, c);
  }
  if (a.kind == Kind::String && b.kind == Kind::String) {
    const int c = a.text->compare(*b.text);
    return ordered(op, (c > 0) - (c < 0));
  }
  if (is_list_or_tuple(a) && is_list_or_tuple(b) && as_list(a).flavor == as_list(b).flavor) {
    const auto& x = as_list(a).items;
    const auto& y = as_list(b).items;
    for (size_t i = 0; i < x.size() && i < y.size(); ++i) {
      if (!py_equal(x[i], y[i])) return py_order(op, x[i], y[i]);
    }
    return ordered(op, (x.size() > y.size()) - (x.size() < y.size()));
  }
  if (a.kind == Kind::Undefined) fail_undefined(a);
  if (b.kind == Kind::Undefined) fail_undefined(b);
  runtime_fail(std::string("'") + op_symbol(op) + "' not supported between instances of '" + type_name(a) + "' and '" +
               type_name(b) + "'");
}

bool truthy(const Value& v) {
  switch (v.kind) {
    case Kind::Undefined:
    case Kind::None: return false;
    case Kind::Bool: return v.boolean;
    case Kind::Int: return v.integer != 0;
    case Kind::Float: return v.number != 0.0;
    case Kind::String: return !v.text->empty();
    case Kind::List: return as_list(v).flavor == Flavor::Iterator || !as_list(v).items.empty();
    case Kind::Dict: return !as_dict(v).items.empty();
    default: return true;
  }
}

std::vector<Value> code_points(const Value& s) {
  std::vector<Value> out;
  const std::string& text = *s.text;
  for (size_t i = 0, len = 0; i < text.size(); i += len) {
    cp_at(text, i, &len);
    out.push_back(str_value(text.substr(i, len)));
  }
  return out;
}

// iter(): what a for loop or filter sees. Consumes generators.
std::vector<Value> iterate(const Value& v) {
  switch (v.kind) {
    case Kind::Undefined: return {};
    case Kind::String: return code_points(v);
    case Kind::List: {
      ListObject& list = as_list(v);
      if (list.flavor != Flavor::Iterator) return list.items;
      std::vector<Value> rest(list.items.begin() + static_cast<std::ptrdiff_t>(list.cursor), list.items.end());
      list.cursor = list.items.size();
      return rest;
    }
    case Kind::Dict: {
      std::vector<Value> keys;
      for (const auto& item : as_dict(v).items) keys.push_back(item.first);
      return keys;
    }
    default: runtime_fail(std::string("'") + type_name(v) + "' object is not iterable");
  }
}

bool hashable(const Value& v) {
  if (v.kind == Kind::Dict) return false;
  if (v.kind != Kind::List) return true;
  const ListObject& list = as_list(v);
  switch (list.flavor) {
    case Flavor::List:
    case Flavor::Items:
    case Flavor::Keys:
    case Flavor::Values: return false;
    case Flavor::Tuple: return std::all_of(list.items.begin(), list.items.end(), hashable);
    default: return true;
  }
}

// The `in` operator.
bool contains(const Value& container, const Value& item) {
  switch (container.kind) {
    case Kind::Undefined: return false;
    case Kind::String:
      if (item.kind != Kind::String) {
        runtime_fail(std::string("'in <string>' requires string as left operand, not ") + type_name(item));
      }
      return container.text->find(*item.text) != std::string::npos;
    case Kind::List: {
      ListObject& list = as_list(container);
      if (list.flavor == Flavor::Iterator) {  // consumes up to the match
        while (list.cursor < list.items.size()) {
          if (py_equal(list.items[list.cursor++], item)) return true;
        }
        return false;
      }
      if ((list.flavor == Flavor::Keys || list.flavor == Flavor::Items) && !hashable(item)) {
        runtime_fail(std::string("unhashable type: '") + type_name(item) + "'");
      }
      return std::any_of(list.items.begin(), list.items.end(), [&](const Value& x) { return py_equal(x, item); });
    }
    case Kind::Dict:
      if (!hashable(item)) runtime_fail(std::string("unhashable type: '") + type_name(item) + "'");
      return dict_find(container, item) != nullptr;
    default: runtime_fail(std::string("argument of type '") + type_name(container) + "' is not iterable");
  }
}

int64_t py_len(const Value& v) {
  switch (v.kind) {
    case Kind::Undefined: return 0;
    case Kind::String: return static_cast<int64_t>(cp_count(*v.text));
    case Kind::List:
      if (as_list(v).flavor != Flavor::Iterator) return static_cast<int64_t>(as_list(v).items.size());
      break;
    case Kind::Dict: return static_cast<int64_t>(as_dict(v).items.size());
    default: break;
  }
  runtime_fail(std::string("object of type '") + type_name(v) + "' has no len()");
}

// ---------------------------------------------------------------------------
// tojson: transformers' json.dumps(x, ensure_ascii=False, indent=None,
// separators=None, sort_keys=False) override of jinja2's filter.

struct JsonStyle {
  bool ascii = false;
  bool indent = false;
  std::string unit;
  std::string item_separator = ", ";
  std::string key_separator = ": ";
  bool sort_keys = false;
};

void append_json_string(std::string& out, std::string_view s, bool ascii) {
  out += '"';
  char buffer[16];
  for (size_t i = 0, len = 0; i < s.size(); i += len) {
    const char32_t c = cp_at(s, i, &len);
    switch (c) {
      case '"': out += "\\\""; continue;
      case '\\': out += "\\\\"; continue;
      case '\n': out += "\\n"; continue;
      case '\r': out += "\\r"; continue;
      case '\t': out += "\\t"; continue;
      case '\b': out += "\\b"; continue;
      case '\f': out += "\\f"; continue;
      default: break;
    }
    if (c < 0x20 || (ascii && c >= 0x7F)) {
      if (c >= 0x10000) {
        const char32_t v = c - 0x10000;
        std::snprintf(buffer, sizeof(buffer), "\\u%04x\\u%04x", static_cast<unsigned>(0xD800 + (v >> 10)),
                      static_cast<unsigned>(0xDC00 + (v & 0x3FF)));
      } else {
        std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned>(c));
      }
      out += buffer;
    } else {
      out.append(s.substr(i, len));
    }
  }
  out += '"';
}

void append_json_float(std::string& out, double d) {
  if (std::isnan(d)) {
    out += "NaN";
  } else if (std::isinf(d)) {
    out += d < 0 ? "-Infinity" : "Infinity";
  } else {
    append_float(out, d);
  }
}

[[noreturn]] void not_serializable(const Value& v) {
  runtime_fail(std::string("Object of type ") + type_name(v) + " is not JSON serializable");
}

void dump_json(std::string& out, const Value& v, const JsonStyle& style, int level) {
  auto newline = [&](int depth) {
    if (!style.indent) return;
    out += '\n';
    for (int i = 0; i < depth; ++i) out += style.unit;
  };
  switch (v.kind) {
    case Kind::None: out += "null"; return;
    case Kind::Bool: out += v.boolean ? "true" : "false"; return;
    case Kind::Int: append_int(out, v.integer); return;
    case Kind::Float: append_json_float(out, v.number); return;
    case Kind::String: append_json_string(out, *v.text, style.ascii); return;
    case Kind::List: {
      if (!is_list_or_tuple(v)) not_serializable(v);
      const auto& items = as_list(v).items;
      if (items.empty()) {
        out += "[]";
        return;
      }
      out += '[';
      for (size_t i = 0; i < items.size(); ++i) {
        if (i) out += style.item_separator;
        newline(level + 1);
        dump_json(out, items[i], style, level + 1);
      }
      newline(level);
      out += ']';
      return;
    }
    case Kind::Dict: {
      std::vector<const std::pair<Value, Value>*> items;
      for (const auto& item : as_dict(v).items) items.push_back(&item);
      if (items.empty()) {
        out += "{}";
        return;
      }
      if (style.sort_keys) {
        std::stable_sort(items.begin(), items.end(),
                         [](const auto* x, const auto* y) { return py_order(CmpOp::Lt, x->first, y->first); });
      }
      out += '{';
      for (size_t i = 0; i < items.size(); ++i) {
        if (i) out += style.item_separator;
        newline(level + 1);
        const Value& key = items[i]->first;
        switch (key.kind) {
          case Kind::String: append_json_string(out, *key.text, style.ascii); break;
          case Kind::Bool: out += key.boolean ? "\"true\"" : "\"false\""; break;
          case Kind::None: out += "\"null\""; break;
          case Kind::Int:
            out += '"';
            append_int(out, key.integer);
            out += '"';
            break;
          case Kind::Float:
            out += '"';
            append_json_float(out, key.number);
            out += '"';
            break;
          default:
            runtime_fail(std::string("keys must be str, int, float, bool or None, not ") + type_name(key));
        }
        out += style.key_separator;
        dump_json(out, items[i]->second, style, level + 1);
      }
      newline(level);
      out += '}';
      return;
    }
    default: not_serializable(v);
  }
}

// ---------------------------------------------------------------------------
// Python's int() and float() parsing of strings, for the |int and |float
// filters. They return false where Python raises ValueError.

std::string ascii_number_text(std::string_view s) {
  const std::string_view t = py_strip(s, CharSet(), true, true);
  if (!is_ascii(t)) runtime_fail("converting non-ASCII digits to a number is not supported");
  return std::string(t);
}

bool py_parse_float(std::string_view raw, double* out) {
  const std::string s = ascii_number_text(raw);
  size_t i = 0;
  std::string clean;
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) clean += s[i++];
  std::string word;
  for (size_t j = i; j < s.size(); ++j) word += lower_ascii(s[j]);
  if (word == "inf" || word == "infinity" || word == "nan") {
    *out = word == "nan" ? std::nan("") : HUGE_VAL;
    if (!clean.empty() && clean[0] == '-') *out = -*out;
    return true;
  }
  // digitpart: digit (["_"] digit)*
  auto digits = [&](bool required) {
    const size_t start = i;
    while (i < s.size()) {
      if (s[i] >= '0' && s[i] <= '9') {
        clean += s[i++];
      } else if (s[i] == '_' && i > start && i + 1 < s.size() && s[i + 1] >= '0' && s[i + 1] <= '9' &&
                 s[i - 1] != '_') {
        ++i;
      } else {
        break;
      }
    }
    return !required || i > start;
  };
  const size_t before = clean.size();
  digits(false);
  const bool whole = clean.size() > before;
  bool fraction = false;
  if (i < s.size() && s[i] == '.') {
    clean += s[i++];
    const size_t mark = clean.size();
    digits(false);
    fraction = clean.size() > mark;
  }
  if (!whole && !fraction) return false;
  if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
    clean += s[i++];
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) clean += s[i++];
    if (!digits(true)) return false;
  }
  if (i != s.size()) return false;
  *out = std::strtod(clean.c_str(), nullptr);
  return true;
}

bool py_parse_int(std::string_view raw, int64_t base, int64_t* out) {
  if (base != 0 && (base < 2 || base > 36)) return false;
  const std::string s = ascii_number_text(raw);
  size_t i = 0;
  bool negative = false;
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) negative = s[i++] == '-';
  bool prefixed = false;
  if (i + 1 < s.size() && s[i] == '0') {
    const char p = lower_ascii(s[i + 1]);
    const int64_t prefix_base = p == 'x' ? 16 : p == 'o' ? 8 : p == 'b' ? 2 : 0;
    if (prefix_base && (base == 0 || base == prefix_base)) {
      base = prefix_base;
      i += 2;
      prefixed = true;
    }
  }
  const bool base_zero = base == 0;
  if (base_zero) base = 10;
  if (i < s.size() && s[i] == '_' && prefixed) ++i;
  const size_t start = i;
  int64_t value = 0;
  bool leading_zero = false, nonzero = false;
  while (i < s.size()) {
    char c = lower_ascii(s[i]);
    if (c == '_') {
      if (i == start || i + 1 >= s.size() || s[i + 1] == '_') return false;
      ++i;
      continue;
    }
    const int digit = c >= '0' && c <= '9' ? c - '0' : (c >= 'a' && c <= 'z' ? c - 'a' + 10 : 99);
    if (digit >= base) return false;
    if (i == start && digit == 0) leading_zero = true;
    if (digit) nonzero = true;
    if (__builtin_mul_overflow(value, base, &value) || __builtin_add_overflow(value, digit, &value)) {
      runtime_fail("integers beyond 64 bits are not supported");
    }
    ++i;
  }
  if (i == start) return false;
  if (base_zero && !prefixed && leading_zero && nonzero) return false;  // int('010', 0)
  *out = negative ? -value : value;
  return true;
}

// ---------------------------------------------------------------------------
// The parsed program.

enum class Ex : uint8_t {
  Const, Name, Getattr, Getitem, Slice, Call, Filter, Test, Not, Neg, Pos, And, Or, Bin, Concat, Compare, Cond,
  List, Tuple, Dict,
};
enum class BinOp : uint8_t { Add, Sub, Mul, Div, FloorDiv, Mod, Pow };
enum class FilterId : uint8_t {
  Abs, Capitalize, Default, Escape, First, Float, Indent, Int, Items, Join, Last, Length, List, Lower, Map, Reject,
  RejectAttr, Replace, Reverse, Safe, Select, SelectAttr, Sort, String, Title, ToJson, Trim, Unique, Upper,
};
enum class TestId : uint8_t {
  Defined, Undefined, None, String, Number, Integer, Float, Mapping, Iterable, Sequence, Boolean, True, False, Eq,
  Ne, Lt, Le, Gt, Ge, In, Odd, Even, DivisibleBy,
};

std::optional<FilterId> filter_named(std::string_view name) {
  static const std::pair<std::string_view, FilterId> kFilters[] = {
      {"abs", FilterId::Abs},          {"capitalize", FilterId::Capitalize}, {"count", FilterId::Length},
      {"d", FilterId::Default},        {"default", FilterId::Default},       {"e", FilterId::Escape},
      {"escape", FilterId::Escape},    {"first", FilterId::First},           {"float", FilterId::Float},
      {"indent", FilterId::Indent},    {"int", FilterId::Int},               {"items", FilterId::Items},
      {"join", FilterId::Join},        {"last", FilterId::Last},             {"length", FilterId::Length},
      {"list", FilterId::List},        {"lower", FilterId::Lower},           {"map", FilterId::Map},
      {"reject", FilterId::Reject},    {"rejectattr", FilterId::RejectAttr}, {"replace", FilterId::Replace},
      {"reverse", FilterId::Reverse},  {"safe", FilterId::Safe},             {"select", FilterId::Select},
      {"selectattr", FilterId::SelectAttr}, {"sort", FilterId::Sort},        {"string", FilterId::String},
      {"title", FilterId::Title},      {"tojson", FilterId::ToJson},         {"trim", FilterId::Trim},
      {"unique", FilterId::Unique},    {"upper", FilterId::Upper},
  };
  for (const auto& [n, id] : kFilters) {
    if (n == name) return id;
  }
  return std::nullopt;
}

std::optional<TestId> test_named(std::string_view name) {
  static const std::pair<std::string_view, TestId> kTests[] = {
      {"defined", TestId::Defined}, {"undefined", TestId::Undefined}, {"none", TestId::None},
      {"string", TestId::String},   {"number", TestId::Number},       {"integer", TestId::Integer},
      {"float", TestId::Float},     {"mapping", TestId::Mapping},     {"iterable", TestId::Iterable},
      {"sequence", TestId::Sequence}, {"boolean", TestId::Boolean},   {"true", TestId::True},
      {"false", TestId::False},     {"eq", TestId::Eq},               {"equalto", TestId::Eq},
      {"==", TestId::Eq},           {"ne", TestId::Ne},               {"!=", TestId::Ne},
      {"lt", TestId::Lt},           {"lessthan", TestId::Lt},         {"<", TestId::Lt},
      {"le", TestId::Le},           {"<=", TestId::Le},               {"gt", TestId::Gt},
      {"greaterthan", TestId::Gt},  {">", TestId::Gt},                {"ge", TestId::Ge},
      {">=", TestId::Ge},           {"in", TestId::In},               {"odd", TestId::Odd},
      {"even", TestId::Even},       {"divisibleby", TestId::DivisibleBy},
  };
  for (const auto& [n, id] : kTests) {
    if (n == name) return id;
  }
  return std::nullopt;
}

// Methods a template may call; anything else fails at parse time.
constexpr std::string_view kCallableMethods[] = {
    "startswith", "endswith", "strip", "lstrip", "rstrip", "split", "rsplit", "replace", "upper", "lower",
    "title",      "capitalize", "count", "find", "rfind", "join", "items", "keys", "values", "get",
};

struct Expr {
  Ex kind = Ex::Const;
  uint8_t op = 0;  // BinOp, FilterId or TestId
  int line = 0;
  int name = -1;  // Name, Getattr attribute
  Value constant;
  std::vector<const Expr*> args;  // operands; a Slice holds nullptr for missing parts
  std::vector<std::pair<int, const Expr*>> kwargs;
  std::vector<CmpOp> ops;  // Compare: one per operand after the first
};

struct Node;
using Body = std::vector<const Node*>;

struct Target {
  int name = -1;  // a name, or a tuple of items when negative
  std::vector<Target> items;
};

struct MacroDef {
  int name = -1;
  std::vector<int> params;
  std::vector<const Expr*> defaults;  // per parameter, nullptr when it has none
  Body body;
  bool catch_varargs = false, catch_kwargs = false, uses_caller = false, explicit_caller = false;
};

enum class St : uint8_t {
  Text, Output, If, For, Set, SetAttr, SetBlock, Macro, CallBlock, Break, Continue, Generation,
};

struct Node {
  St kind = St::Text;
  int line = 0;
  std::string text;
  const Expr* expr = nullptr;  // Output, Set, SetAttr value, For iterable, CallBlock call
  const Expr* test = nullptr;  // For filter
  Target target;
  int name = -1, attr = -1;  // SetAttr namespace and attribute, Macro name
  std::vector<std::pair<const Expr*, Body>> branches;
  Body body, else_body;
  const MacroDef* macro = nullptr;
};

// Names every template can use; interned first so their ids are constants.
enum : int { kLoop, kCaller, kVarargs, kKwargs, kRange, kDict, kNamespace, kRaiseException, kStrftimeNow, kWellKnown };
constexpr const char* kWellKnownNames[] = {"loop",      "caller",          "varargs",     "kwargs", "range",
                                           "dict",      "namespace",       "raise_exception", "strftime_now"};

struct Program {
  Program() {
    for (const char* name : kWellKnownNames) intern(name);
  }
  int intern(const std::string& name) {
    auto [it, inserted] = ids.emplace(name, static_cast<int>(names.size()));
    if (inserted) names.push_back(make_text(name));
    return it->second;
  }
  int id_of(const std::string& name) const {
    auto it = ids.find(name);
    return it == ids.end() ? -1 : it->second;
  }

  std::deque<Expr> exprs;
  std::deque<Node> nodes;
  std::deque<MacroDef> macros;
  std::vector<Str> names;
  std::unordered_map<std::string, int> ids;
  Body body;
  Str role_key = make_text("role");
  Str content_key = make_text("content");
};

// ---------------------------------------------------------------------------
// Lexer: jinja2's tokenizer with trim_blocks and lstrip_blocks.

enum class Tok : uint8_t { Data, VarBegin, VarEnd, BlockBegin, BlockEnd, Name, String, Int, Float, Op, End };

struct Token {
  Tok type = Tok::End;
  int line = 1;
  std::string text;  // data, name, decoded string or operator
  int64_t integer = 0;
  double number = 0;
};

[[noreturn]] void syntax_error(int line, const std::string& message) {
  fail("lokahi: chat template line " + std::to_string(line) + ": " + message);
}

bool is_digit(char c) { return c >= '0' && c <= '9'; }
bool is_name_start(char c) { return alpha_ascii(c) || c == '_'; }

bool all_space(std::string_view s) {
  for (size_t i = 0, len = 0; i < s.size(); i += len) {
    if (!py_space(cp_at(s, i, &len))) return false;
  }
  return true;
}

// Python's unicode-escape decoding of a string literal, applied (as jinja2
// does) after non-ASCII characters are turned into \x, \u or \U escapes.
std::string unescape(std::string_view raw, int line) {
  std::string ascii;
  ascii.reserve(raw.size());
  for (size_t i = 0, len = 0; i < raw.size(); i += len) {
    const char32_t c = cp_at(raw, i, &len);
    if (c < 0x80) {
      ascii += static_cast<char>(c);
    } else {
      append_hex_escape(ascii, c);
    }
  }
  std::string out;
  auto hex = [&](size_t& i, int count) {
    if (i + static_cast<size_t>(count) > ascii.size()) syntax_error(line, "truncated escape in string literal");
    uint32_t value = 0;
    for (int k = 0; k < count; ++k) {
      const char c = lower_ascii(ascii[i++]);
      if (!std::isxdigit(static_cast<unsigned char>(c))) syntax_error(line, "truncated escape in string literal");
      value = value * 16 + static_cast<uint32_t>(is_digit(c) ? c - '0' : c - 'a' + 10);
    }
    return value;
  };
  auto put = [&](uint32_t cp) {
    if (cp > 0x10FFFF) syntax_error(line, "illegal Unicode character in string literal");
    if (cp >= 0xD800 && cp <= 0xDFFF) syntax_error(line, "surrogate code points in string literals are not supported");
    unicode::append_utf8(out, cp);
  };
  for (size_t i = 0; i < ascii.size();) {
    const char c = ascii[i++];
    if (c != '\\') {
      out += c;
      continue;
    }
    if (i >= ascii.size()) syntax_error(line, "\\ at end of string literal");
    const char e = ascii[i++];
    switch (e) {
      case '\n': break;
      case '\\': out += '\\'; break;
      case '\'': out += '\''; break;
      case '"': out += '"'; break;
      case 'a': out += '\a'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case 'n': out += '\n'; break;
      case 'r': out += '\r'; break;
      case 't': out += '\t'; break;
      case 'v': out += '\v'; break;
      case 'x': put(hex(i, 2)); break;
      case 'u': put(hex(i, 4)); break;
      case 'U': put(hex(i, 8)); break;
      case 'N': syntax_error(line, "\\N{...} escapes are not supported");
      default:
        if (e >= '0' && e <= '7') {
          uint32_t value = static_cast<uint32_t>(e - '0');
          for (int k = 0; k < 2 && i < ascii.size() && ascii[i] >= '0' && ascii[i] <= '7'; ++k) {
            value = value * 8 + static_cast<uint32_t>(ascii[i++] - '0');
          }
          put(value);
        } else {  // unknown escapes keep their backslash
          out += '\\';
          out += e;
        }
    }
  }
  return out;
}

class Lexer {
 public:
  explicit Lexer(std::string_view source) {
    // \r\n and \r become \n, and one trailing newline is dropped
    // (keep_trailing_newline=False).
    src_.reserve(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
      if (source[i] == '\r') {
        src_ += '\n';
        if (i + 1 < source.size() && source[i + 1] == '\n') ++i;
      } else {
        src_ += source[i];
      }
    }
    if (!src_.empty() && src_.back() == '\n') src_.pop_back();
  }

  std::vector<Token> run() {
    while (pos_ < src_.size()) {
      size_t at = pos_;
      char kind = 0;
      while ((at = src_.find('{', at)) != npos) {
        if (at + 1 < src_.size() && (src_[at + 1] == '{' || src_[at + 1] == '%' || src_[at + 1] == '#')) {
          kind = src_[at + 1];
          break;
        }
        ++at;
      }
      if (at == npos) {
        push(Tok::Data, src_.substr(pos_), line_);
        break;
      }
      size_t open = at + 2;
      char sign = 0;
      if (open < src_.size() && (src_[open] == '-' || src_[open] == '+')) sign = src_[open++];
      std::string_view text(src_.data() + pos_, at - pos_);
      const int text_line = line_;
      line_ += static_cast<int>(std::count(text.begin(), text.end(), '\n'));
      if (sign == '-') {
        text = py_strip(text, CharSet(), false, true);
      } else if (sign != '+' && kind != '{') {
        // lstrip_blocks: whitespace between a line start and a block or comment tag.
        const size_t newline = text.rfind('\n');
        const size_t start = newline == npos ? 0 : newline + 1;
        if ((start > 0 || line_starting_) && start < text.size() && all_space(text.substr(start))) {
          text = text.substr(0, start);
        }
      }
      if (!text.empty()) push(Tok::Data, std::string(text), text_line);
      pos_ = open;
      line_starting_ = false;
      if (kind == '#') {
        comment();
      } else {
        if (kind == '%') reject_raw();
        tag(kind == '%');
      }
    }
    push(Tok::End, "", line_);
    return std::move(tokens_);
  }

 private:
  static constexpr size_t npos = std::string::npos;

  void push(Tok type, std::string text, int line) {
    Token t;
    t.type = type;
    t.line = line;
    t.text = std::move(text);
    tokens_.push_back(std::move(t));
  }

  size_t skip_space(size_t p) {
    size_t len = 0;
    while (p < src_.size() && py_space(cp_at(src_, p, &len))) {
      if (src_[p] == '\n') ++line_;
      p += len;
    }
    return p;
  }

  void reject_raw() {
    const size_t start = skip_space_quiet(pos_);
    if (src_.compare(start, 3, "raw") != 0) return;
    const size_t end = skip_space_quiet(start + 3);
    if (src_.compare(end, 2, "%}") == 0 || src_.compare(end, 3, "-%}") == 0) {
      syntax_error(line_, "unsupported construct '{% raw %}'");
    }
  }

  size_t skip_space_quiet(size_t p) const {
    size_t len = 0;
    while (p < src_.size() && py_space(cp_at(src_, p, &len))) p += len;
    return p;
  }

  void comment() {
    const size_t close = src_.find("#}", pos_);
    if (close == npos) syntax_error(line_, "missing end of comment tag");
    char sign = 0;
    if (close > pos_ && (src_[close - 1] == '-' || src_[close - 1] == '+')) sign = src_[close - 1];
    line_ += static_cast<int>(std::count(src_.begin() + static_cast<std::ptrdiff_t>(pos_),
                                         src_.begin() + static_cast<std::ptrdiff_t>(close), '\n'));
    size_t end = close + 2;
    if (sign == '-') {
      end = skip_space(end);
    } else if (sign != '+' && end < src_.size() && src_[end] == '\n') {
      ++end;
      ++line_;
    }
    line_starting_ = end > close + 2 && src_[end - 1] == '\n';
    pos_ = end;
  }

  void tag(bool block) {
    push(block ? Tok::BlockBegin : Tok::VarBegin, "", line_);
    balance_.clear();
    while (true) {
      if (pos_ >= src_.size()) {
        syntax_error(line_, std::string("unexpected end of template, expected '") + (block ? "%}" : "}}") + "'");
      }
      if (balance_.empty() && tag_end(block)) return;
      const char c = src_[pos_];
      size_t len = 0;
      if (py_space(cp_at(src_, pos_, &len))) {
        if (c == '\n') ++line_;
        pos_ += len;
      } else if (is_digit(c)) {
        number();
      } else if (is_name_start(c)) {
        size_t end = pos_;
        while (end < src_.size() && (is_name_start(src_[end]) || is_digit(src_[end]))) ++end;
        push(Tok::Name, src_.substr(pos_, end - pos_), line_);
        pos_ = end;
      } else if (c == '\'' || c == '"') {
        string_literal();
      } else {
        op();
      }
    }
  }

  // Block ends: "+%}", "-%}" plus whitespace, or "%}" plus one newline
  // (trim_blocks). Variable ends: "-}}" plus whitespace, or "}}".
  bool tag_end(bool block) {
    const char close = block ? '%' : '}';
    const size_t left = src_.size() - pos_;
    if (block && left >= 3 && src_.compare(pos_, 3, "+%}") == 0) {
      pos_ += 3;
      line_starting_ = false;
    } else if (left >= 3 && src_[pos_] == '-' && src_[pos_ + 1] == close && src_[pos_ + 2] == '}') {
      const size_t start = pos_ + 3;
      pos_ = skip_space(start);
      line_starting_ = pos_ > start && src_[pos_ - 1] == '\n';
    } else if (left >= 2 && src_[pos_] == close && src_[pos_ + 1] == '}') {
      pos_ += 2;
      line_starting_ = false;
      if (block && pos_ < src_.size() && src_[pos_] == '\n') {
        ++pos_;
        ++line_;
        line_starting_ = true;
      }
    } else {
      return false;
    }
    push(block ? Tok::BlockEnd : Tok::VarEnd, "", line_);
    return true;
  }

  // (\d+_)*\d+, or npos.
  size_t digits_end(size_t p) const {
    if (p >= src_.size() || !is_digit(src_[p])) return npos;
    while (p < src_.size() && is_digit(src_[p])) ++p;
    while (p + 1 < src_.size() && src_[p] == '_' && is_digit(src_[p + 1])) {
      ++p;
      while (p < src_.size() && is_digit(src_[p])) ++p;
    }
    return p;
  }

  size_t exponent_end(size_t p) const {
    if (p >= src_.size() || (src_[p] != 'e' && src_[p] != 'E')) return npos;
    ++p;
    if (p < src_.size() && (src_[p] == '+' || src_[p] == '-')) ++p;
    return digits_end(p);
  }

  // jinja2's float_re: digits, then an optional fraction and a required
  // exponent, or a required fraction.
  size_t float_end(size_t p) const {
    const size_t whole = digits_end(p);
    if (whole == npos) return npos;
    size_t fraction = npos;
    if (whole < src_.size() && src_[whole] == '.') fraction = digits_end(whole + 1);
    if (fraction != npos) {
      const size_t e = exponent_end(fraction);
      if (e != npos) return e;
    }
    const size_t e = exponent_end(whole);
    if (e != npos) return e;
    return fraction;
  }

  // (_?[digit])+ from p; the end of the last digit, or npos.
  template <typename Valid>
  size_t digit_run(size_t p, Valid valid) const {
    size_t end = npos;
    while (true) {
      size_t q = p;
      if (q < src_.size() && src_[q] == '_') ++q;
      if (q >= src_.size() || !valid(src_[q])) break;
      p = q + 1;
      end = p;
    }
    return end;
  }

  void number() {
    const int line = line_;
    if (pos_ == 0 || src_[pos_ - 1] != '.') {
      const size_t end = float_end(pos_);
      if (end != npos) {
        std::string clean;
        for (size_t i = pos_; i < end; ++i) {
          if (src_[i] != '_') clean += src_[i];
        }
        Token t;
        t.type = Tok::Float;
        t.line = line;
        t.number = std::strtod(clean.c_str(), nullptr);
        t.text = src_.substr(pos_, end - pos_);
        tokens_.push_back(std::move(t));
        pos_ = end;
        return;
      }
    }
    int base = 10;
    size_t end = npos;
    if (src_[pos_] == '0' && pos_ + 1 < src_.size()) {
      const char x = lower_ascii(src_[pos_ + 1]);
      auto binary = [](char c) { return c == '0' || c == '1'; };
      auto octal = [](char c) { return c >= '0' && c <= '7'; };
      auto hexadecimal = [](char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; };
      if (x == 'b') end = digit_run(pos_ + 2, binary), base = 2;
      if (x == 'o') end = digit_run(pos_ + 2, octal), base = 8;
      if (x == 'x') end = digit_run(pos_ + 2, hexadecimal), base = 16;
      if (end == npos) base = 10;
    }
    if (end == npos) {
      if (src_[pos_] == '0') {
        end = digit_run(pos_ + 1, [](char c) { return c == '0'; });
      } else {
        end = digit_run(pos_ + 1, is_digit);
      }
      if (end == npos) end = pos_ + 1;
    }
    int64_t value = 0;
    for (size_t i = pos_ + (base == 10 ? 0 : 2); i < end; ++i) {
      const char c = lower_ascii(src_[i]);
      if (c == '_') continue;
      const int digit = is_digit(c) ? c - '0' : c - 'a' + 10;
      if (__builtin_mul_overflow(value, base, &value) || __builtin_add_overflow(value, digit, &value)) {
        syntax_error(line, "integers beyond 64 bits are not supported");
      }
    }
    Token t;
    t.type = Tok::Int;
    t.line = line;
    t.integer = value;
    t.text = src_.substr(pos_, end - pos_);
    tokens_.push_back(std::move(t));
    pos_ = end;
  }

  void string_literal() {
    const char quote = src_[pos_];
    size_t p = pos_ + 1;
    while (p < src_.size() && src_[p] != quote) {
      if (src_[p] == '\\') ++p;
      ++p;
    }
    if (p >= src_.size()) syntax_error(line_, "unterminated string literal");
    const std::string_view raw(src_.data() + pos_ + 1, p - pos_ - 1);
    push(Tok::String, unescape(raw, line_), line_);
    line_ += static_cast<int>(std::count(raw.begin(), raw.end(), '\n'));
    pos_ = p + 1;
  }

  void op() {
    static const char* const kTwo[] = {"**", "//", "==", "!=", ">=", "<="};
    for (const char* two : kTwo) {
      if (src_.compare(pos_, 2, two) == 0) {
        push(Tok::Op, two, line_);
        pos_ += 2;
        return;
      }
    }
    const char c = src_[pos_];
    if (c == 0 || !std::strchr("+-/*%~[](){}><=.:|,;", c)) {
      syntax_error(line_, static_cast<unsigned char>(c) >= 0x80 ? "unsupported non-ASCII character in a tag"
                                                                : std::string("unexpected character '") + c + "'");
    }
    if (c == '(') balance_.push_back(')');
    if (c == '[') balance_.push_back(']');
    if (c == '{') balance_.push_back('}');
    if (c == ')' || c == ']' || c == '}') {
      if (balance_.empty()) syntax_error(line_, std::string("unexpected '") + c + "'");
      if (balance_.back() != c) {
        syntax_error(line_, std::string("unexpected '") + c + "', expected '" + balance_.back() + "'");
      }
      balance_.pop_back();
    }
    push(Tok::Op, std::string(1, c), line_);
    ++pos_;
  }

  std::string src_;
  size_t pos_ = 0;
  int line_ = 1;
  bool line_starting_ = true;
  std::vector<char> balance_;
  std::vector<Token> tokens_;
};

// ---------------------------------------------------------------------------
// Parser: jinja2's grammar and operator precedence.

class Parser {
 public:
  Parser(std::vector<Token> tokens, Program& program) : t_(std::move(tokens)), p_(program) {}

  void run() {
    p_.body = subparse({});
    for (const auto& [id, line] : calls_) {
      if (!macro_names_.count(id)) syntax_error(line, "call to unsupported function '" + *p_.names[id] + "'");
    }
  }

 private:
  using Ends = std::initializer_list<std::string_view>;

  const Token& cur() const { return t_[pos_]; }
  const Token& peek() const { return t_[std::min(pos_ + 1, t_.size() - 1)]; }
  void advance() {
    if (pos_ + 1 < t_.size()) ++pos_;
  }
  bool name_is(std::string_view n) const { return cur().type == Tok::Name && cur().text == n; }
  bool op_is(std::string_view o) const { return cur().type == Tok::Op && cur().text == o; }
  bool skip_name(std::string_view n) {
    if (!name_is(n)) return false;
    advance();
    return true;
  }
  bool skip_op(std::string_view o) {
    if (!op_is(o)) return false;
    advance();
    return true;
  }
  [[noreturn]] void error(const std::string& message) const { syntax_error(cur().line, message); }

  static std::string describe(const Token& t) {
    switch (t.type) {
      case Tok::Data: return "template data";
      case Tok::VarBegin: return "begin of print statement";
      case Tok::VarEnd: return "end of print statement";
      case Tok::BlockBegin: return "begin of statement block";
      case Tok::BlockEnd: return "end of statement block";
      case Tok::Name:
      case Tok::Op: return "'" + t.text + "'";
      case Tok::String: return "string";
      case Tok::Int: return "integer";
      case Tok::Float: return "float";
      case Tok::End: return "end of template";
    }
    return "token";
  }

  void expect(Tok type, const std::string& what) {
    if (cur().type != type) error("expected " + what + ", got " + describe(cur()));
    advance();
  }
  void expect_op(std::string_view o) {
    if (!skip_op(o)) error("expected '" + std::string(o) + "', got " + describe(cur()));
  }
  void expect_name(std::string_view n) {
    if (!skip_name(n)) error("expected '" + std::string(n) + "', got " + describe(cur()));
  }
  std::string expect_identifier() {
    if (cur().type != Tok::Name) error("expected a name, got " + describe(cur()));
    std::string name = cur().text;
    advance();
    return name;
  }
  static bool is_constant_name(std::string_view n) {
    return n == "true" || n == "false" || n == "none" || n == "True" || n == "False" || n == "None";
  }
  int assignable(const std::string& name, int line) {
    if (is_constant_name(name)) syntax_error(line, "can't assign to a constant");
    return p_.intern(name);
  }

  Expr* make(Ex kind, int line) {
    p_.exprs.emplace_back();
    Expr* e = &p_.exprs.back();
    e->kind = kind;
    e->line = line;
    return e;
  }
  Expr* constant(Value v, int line) {
    Expr* e = make(Ex::Const, line);
    e->constant = std::move(v);
    return e;
  }
  Node* node(St kind, int line) {
    p_.nodes.emplace_back();
    Node* n = &p_.nodes.back();
    n->kind = kind;
    n->line = line;
    return n;
  }
  MacroDef* new_macro() {
    p_.macros.emplace_back();
    return &p_.macros.back();
  }

  // Statements --------------------------------------------------------------

  Body subparse(Ends ends) {
    Body body;
    while (cur().type != Tok::End) {
      const Token& token = cur();
      if (token.type == Tok::Data) {
        Node* n = node(St::Text, token.line);
        n->text = token.text;
        body.push_back(n);
        advance();
      } else if (token.type == Tok::VarBegin) {
        advance();
        Node* n = node(St::Output, token.line);
        n->expr = parse_tuple();
        expect(Tok::VarEnd, "end of print statement");
        body.push_back(n);
      } else if (token.type == Tok::BlockBegin) {
        advance();
        if (cur().type == Tok::Name) {
          for (std::string_view end : ends) {
            if (cur().text == end) return body;
          }
        }
        body.push_back(statement());
        expect(Tok::BlockEnd, "end of statement block");
      } else {
        error("unexpected " + describe(token));
      }
    }
    return body;
  }

  // parse_statements: an optional colon, the block end, the body up to one of
  // `ends`, and with `drop` the end tag's name.
  Body statements(Ends ends, bool drop) {
    skip_op(":");
    expect(Tok::BlockEnd, "end of statement block");
    Body body = subparse(ends);
    if (cur().type == Tok::End) {
      std::string expected;
      for (std::string_view end : ends) expected += (expected.empty() ? "'" : " or '") + std::string(end) + "'";
      error("unexpected end of template, expected " + expected);
    }
    if (drop) advance();
    return body;
  }

  const Node* statement() {
    if (cur().type != Tok::Name) error("tag name expected");
    const std::string word = cur().text;
    const int line = cur().line;
    if (word == "for") return parse_for();
    if (word == "if") return parse_if();
    if (word == "set") return parse_set();
    if (word == "macro") return parse_macro();
    if (word == "call") return parse_call_block();
    if (word == "break" || word == "continue") {
      if (loop_depth_ == 0) error("'" + word + "' outside a loop");
      advance();
      return node(word == "break" ? St::Break : St::Continue, line);
    }
    if (word == "generation") {
      advance();
      Node* n = node(St::Generation, line);
      n->body = nested_body({"endgeneration"}, nullptr);
      return n;
    }
    static const std::string_view kUnsupported[] = {"block", "extends", "print", "include",    "from", "import", "with",
                                                     "autoescape", "filter", "raw",  "do", "trans", "pluralize"};
    if (one_of(word, kUnsupported)) error("unsupported construct '{% " + word + " %}'");
    error("unknown tag '" + word + "'");
  }

  // A macro, call-block or generation body: loop controls do not reach outside it.
  Body nested_body(Ends ends, MacroDef* macro) {
    const int saved = loop_depth_;
    loop_depth_ = 0;
    if (macro) open_macros_.push_back(macro);
    Body body = statements(ends, true);
    if (macro) open_macros_.pop_back();
    loop_depth_ = saved;
    return body;
  }

  const Node* parse_for() {
    Node* n = node(St::For, cur().line);
    advance();
    n->target = parse_target({"in"});
    expect_name("in");
    n->expr = parse_tuple(false, {"recursive"});
    if (skip_name("if")) n->test = parse_expression();
    if (name_is("recursive")) error("unsupported construct 'recursive' loops");
    ++loop_depth_;
    n->body = statements({"endfor", "else"}, false);
    --loop_depth_;
    if (skip_name("else")) {
      n->else_body = statements({"endfor"}, true);
    } else {
      advance();  // endfor
    }
    return n;
  }

  const Node* parse_if() {
    Node* n = node(St::If, cur().line);
    advance();
    while (true) {
      const Expr* test = parse_tuple(false);
      Body body = statements({"elif", "else", "endif"}, false);
      n->branches.emplace_back(test, std::move(body));
      if (skip_name("elif")) continue;
      if (skip_name("else")) {
        n->else_body = statements({"endif"}, true);
      } else {
        advance();  // endif
      }
      return n;
    }
  }

  const Node* parse_set() {
    const int line = cur().line;
    advance();
    if (cur().type == Tok::Name && !is_constant_name(cur().text) && peek().type == Tok::Op && peek().text == ".") {
      Node* n = node(St::SetAttr, line);
      n->name = p_.intern(cur().text);
      advance();
      advance();
      n->attr = p_.intern(expect_identifier());
      if (!skip_op("=")) error("unsupported construct: block '{% set %}' of a namespace attribute");
      n->expr = parse_tuple();
      return n;
    }
    Target target = parse_target({});
    if (skip_op("=")) {
      Node* n = node(St::Set, line);
      n->target = std::move(target);
      n->expr = parse_tuple();
      return n;
    }
    if (op_is("|")) error("unsupported construct: filtered '{% set %}' blocks");
    if (target.name < 0) error("can't assign a '{% set %}' block to a tuple");
    Node* n = node(St::SetBlock, line);
    n->target = std::move(target);
    n->body = nested_body({"endset"}, nullptr);
    return n;
  }

  const Node* parse_macro() {
    Node* n = node(St::Macro, cur().line);
    advance();
    MacroDef* m = new_macro();
    m->name = assignable(expect_identifier(), n->line);
    signature(m);
    m->body = nested_body({"endmacro"}, m);
    macro_names_.insert(m->name);
    n->name = m->name;
    n->macro = m;
    return n;
  }

  const Node* parse_call_block() {
    Node* n = node(St::CallBlock, cur().line);
    advance();
    MacroDef* caller = new_macro();
    caller->name = kCaller;
    if (op_is("(")) signature(caller);
    n->expr = parse_expression();
    if (n->expr->kind != Ex::Call) error("expected a call after '{% call %}'");
    caller->body = nested_body({"endcall"}, caller);
    n->macro = caller;
    return n;
  }

  void signature(MacroDef* m) {
    expect_op("(");
    bool defaults = false;
    while (!op_is(")")) {
      if (!m->params.empty()) expect_op(",");
      const int line = cur().line;
      const int id = assignable(expect_identifier(), line);
      const Expr* value = nullptr;
      if (skip_op("=")) {
        value = parse_expression();
        defaults = true;
      } else if (defaults) {
        syntax_error(line, "non-default argument follows default argument");
      }
      if (id == kCaller) {
        if (!value) {
          syntax_error(line, "the special \"caller\" argument must be omitted or be given a default");
        }
        m->explicit_caller = true;
      }
      m->params.push_back(id);
      m->defaults.push_back(value);
    }
    advance();
  }

  // An assignment target: a name or a (possibly parenthesized) tuple of them.
  Target parse_target(Ends ends) {
    std::vector<Target> items;
    bool tuple = false;
    while (true) {
      if (!items.empty()) expect_op(",");
      if (tuple_end(ends)) break;
      items.push_back(target_item());
      if (op_is(",")) {
        tuple = true;
      } else {
        break;
      }
    }
    if (tuple) {
      Target t;
      t.items = std::move(items);
      return t;
    }
    if (items.empty()) error("expected an assignment target, got " + describe(cur()));
    return std::move(items[0]);
  }

  Target target_item() {
    if (skip_op("(")) {
      Target t = parse_target({});
      expect_op(")");
      return t;
    }
    if (cur().type != Tok::Name) error("can't assign to " + describe(cur()));
    Target t;
    t.name = assignable(cur().text, cur().line);
    advance();
    return t;
  }

  bool tuple_end(Ends ends) const {
    const Token& t = cur();
    if (t.type == Tok::VarEnd || t.type == Tok::BlockEnd || (t.type == Tok::Op && t.text == ")")) return true;
    if (t.type == Tok::Name) {
      for (std::string_view end : ends) {
        if (t.text == end) return true;
      }
    }
    return false;
  }

  // Expressions -------------------------------------------------------------

  struct Depth {
    explicit Depth(Parser& p) : parser(p) {
      if (++parser.depth_ > 128) parser.error("expression nested too deeply");
    }
    ~Depth() { --parser.depth_; }
    Parser& parser;
  };

  const Expr* parse_expression() { return parse_condexpr(); }

  const Expr* parse_tuple(bool with_condexpr = true, Ends ends = {}, bool explicit_parens = false) {
    const int line = cur().line;
    std::vector<const Expr*> items;
    bool tuple = false;
    while (true) {
      if (!items.empty()) expect_op(",");
      if (tuple_end(ends)) break;
      items.push_back(with_condexpr ? parse_condexpr() : parse_or());
      if (op_is(",")) {
        tuple = true;
      } else {
        break;
      }
    }
    if (!tuple) {
      if (!items.empty()) return items[0];
      if (!explicit_parens) error("expected an expression, got " + describe(cur()));
    }
    Expr* e = make(Ex::Tuple, line);
    e->args = std::move(items);
    return e;
  }

  const Expr* parse_condexpr() {
    const int line = cur().line;
    const Expr* value = parse_or();
    while (skip_name("if")) {
      Expr* e = make(Ex::Cond, line);
      const Expr* test = parse_or();
      const Expr* otherwise = skip_name("else") ? parse_condexpr() : nullptr;
      e->args = {test, value, otherwise};
      value = e;
    }
    return value;
  }

  const Expr* logical(Ex kind, const Expr* left, const Expr* right, int line) {
    Expr* e = make(kind, line);
    e->args = {left, right};
    return e;
  }

  const Expr* parse_or() {
    const Expr* left = parse_and();
    while (name_is("or")) {
      const int line = cur().line;
      advance();
      left = logical(Ex::Or, left, parse_and(), line);
    }
    return left;
  }

  const Expr* parse_and() {
    const Expr* left = parse_not();
    while (name_is("and")) {
      const int line = cur().line;
      advance();
      left = logical(Ex::And, left, parse_not(), line);
    }
    return left;
  }

  const Expr* parse_not() {
    if (name_is("not")) {
      Depth depth(*this);
      Expr* e = make(Ex::Not, cur().line);
      advance();
      e->args = {parse_not()};
      return e;
    }
    return parse_compare();
  }

  const Expr* parse_compare() {
    const int line = cur().line;
    const Expr* left = parse_math1();
    Expr* compare = nullptr;
    while (true) {
      CmpOp op;
      if (op_is("==")) {
        op = CmpOp::Eq;
      } else if (op_is("!=")) {
        op = CmpOp::Ne;
      } else if (op_is("<")) {
        op = CmpOp::Lt;
      } else if (op_is("<=")) {
        op = CmpOp::Le;
      } else if (op_is(">")) {
        op = CmpOp::Gt;
      } else if (op_is(">=")) {
        op = CmpOp::Ge;
      } else if (name_is("in")) {
        op = CmpOp::In;
      } else if (name_is("not") && peek().type == Tok::Name && peek().text == "in") {
        advance();
        op = CmpOp::NotIn;
      } else {
        break;
      }
      advance();
      if (!compare) {
        compare = make(Ex::Compare, line);
        compare->args.push_back(left);
      }
      compare->ops.push_back(op);
      compare->args.push_back(parse_math1());
    }
    return compare ? compare : left;
  }

  const Expr* binary(BinOp op, const Expr* left, const Expr* right, int line) {
    Expr* e = make(Ex::Bin, line);
    e->op = static_cast<uint8_t>(op);
    e->args = {left, right};
    return e;
  }

  const Expr* parse_math1() {
    const Expr* left = parse_concat();
    while (op_is("+") || op_is("-")) {
      const int line = cur().line;
      const BinOp op = cur().text == "+" ? BinOp::Add : BinOp::Sub;
      advance();
      left = binary(op, left, parse_concat(), line);
    }
    return left;
  }

  const Expr* parse_concat() {
    const int line = cur().line;
    const Expr* first = parse_math2();
    if (!op_is("~")) return first;
    Expr* e = make(Ex::Concat, line);
    e->args.push_back(first);
    while (skip_op("~")) e->args.push_back(parse_math2());
    return e;
  }

  const Expr* parse_math2() {
    const Expr* left = parse_pow();
    while (op_is("*") || op_is("/") || op_is("//") || op_is("%")) {
      const int line = cur().line;
      const std::string& o = cur().text;
      const BinOp op = o == "*" ? BinOp::Mul : o == "/" ? BinOp::Div : o == "//" ? BinOp::FloorDiv : BinOp::Mod;
      advance();
      left = binary(op, left, parse_pow(), line);
    }
    return left;
  }

  const Expr* parse_pow() {
    const Expr* left = parse_unary(true);
    while (op_is("**")) {
      const int line = cur().line;
      advance();
      left = binary(BinOp::Pow, left, parse_unary(true), line);
    }
    return left;
  }

  const Expr* parse_unary(bool with_filter) {
    Depth depth(*this);
    const int line = cur().line;
    const Expr* e;
    if (op_is("-") || op_is("+")) {
      Expr* u = make(cur().text == "-" ? Ex::Neg : Ex::Pos, line);
      advance();
      u->args = {parse_unary(false)};
      e = u;
    } else {
      e = parse_primary();
    }
    e = parse_postfix(e);
    if (with_filter) e = parse_filter_expr(e);
    return e;
  }

  const Expr* parse_primary() {
    const Token& t = cur();
    const int line = t.line;
    switch (t.type) {
      case Tok::Name: {
        if (t.text == "true" || t.text == "True" || t.text == "false" || t.text == "False") {
          const bool value = t.text[0] == 't' || t.text[0] == 'T';
          advance();
          return constant(bool_value(value), line);
        }
        if (t.text == "none" || t.text == "None") {
          advance();
          return constant(none_value(), line);
        }
        // jinja2 globals and its template reference, which this renderer does not provide.
        if (t.text == "lipsum" || t.text == "cycler" || t.text == "joiner" || t.text == "self") {
          error("unsupported global '" + t.text + "'");
        }
        Expr* e = make(Ex::Name, line);
        e->name = p_.intern(t.text);
        for (MacroDef* m : open_macros_) {
          if (e->name == kCaller) m->uses_caller = true;
          if (e->name == kVarargs) m->catch_varargs = true;
          if (e->name == kKwargs) m->catch_kwargs = true;
        }
        advance();
        return e;
      }
      case Tok::String: {
        std::string s = t.text;
        advance();
        while (cur().type == Tok::String) {
          s += cur().text;
          advance();
        }
        return constant(str_value(std::move(s)), line);
      }
      case Tok::Int: {
        const int64_t value = t.integer;
        advance();
        return constant(int_value(value), line);
      }
      case Tok::Float: {
        const double value = t.number;
        advance();
        return constant(float_value(value), line);
      }
      case Tok::Op:
        if (t.text == "(") {
          advance();
          const Expr* e = parse_tuple(true, {}, true);
          expect_op(")");
          return e;
        }
        if (t.text == "[") return parse_list();
        if (t.text == "{") return parse_dict();
        break;
      default: break;
    }
    error("unexpected " + describe(t));
  }

  const Expr* parse_list() {
    Expr* e = make(Ex::List, cur().line);
    advance();
    while (!op_is("]")) {
      if (!e->args.empty()) expect_op(",");
      if (op_is("]")) break;
      e->args.push_back(parse_expression());
    }
    advance();
    return e;
  }

  const Expr* parse_dict() {
    Expr* e = make(Ex::Dict, cur().line);
    advance();
    while (!op_is("}")) {
      if (!e->args.empty()) expect_op(",");
      if (op_is("}")) break;
      e->args.push_back(parse_expression());
      expect_op(":");
      e->args.push_back(parse_expression());
    }
    advance();
    return e;
  }

  const Expr* parse_postfix(const Expr* e) {
    while (true) {
      if (op_is(".") || op_is("[")) {
        e = parse_subscript(e);
      } else if (op_is("(")) {
        e = parse_call(e);
      } else {
        return e;
      }
    }
  }

  const Expr* parse_filter_expr(const Expr* e) {
    while (true) {
      if (op_is("|")) {
        e = parse_filter(e);
      } else if (name_is("is")) {
        e = parse_test(e);
      } else if (op_is("(")) {
        e = parse_call(e);
      } else {
        return e;
      }
    }
  }

  const Expr* parse_subscript(const Expr* object) {
    const int line = cur().line;
    if (skip_op(".")) {
      const Token& a = cur();
      if (a.type == Tok::Name) {
        Expr* e = make(Ex::Getattr, line);
        e->args = {object};
        e->name = p_.intern(a.text);
        advance();
        return e;
      }
      if (a.type != Tok::Int) error("expected name or number");
      const Expr* key = constant(int_value(a.integer), a.line);
      advance();
      Expr* e = make(Ex::Getitem, line);
      e->args = {object, key};
      return e;
    }
    advance();  // [
    std::vector<const Expr*> keys;
    while (!op_is("]")) {
      if (!keys.empty()) expect_op(",");
      keys.push_back(parse_subscribed());
    }
    advance();
    Expr* e = make(Ex::Getitem, line);
    if (keys.size() == 1) {
      e->args = {object, keys[0]};
    } else {
      Expr* tuple = make(Ex::Tuple, line);
      tuple->args = std::move(keys);
      e->args = {object, tuple};
    }
    return e;
  }

  const Expr* parse_subscribed() {
    const int line = cur().line;
    const Expr* start = nullptr;
    if (!skip_op(":")) {
      const Expr* key = parse_expression();
      if (!skip_op(":")) return key;
      start = key;
    }
    const Expr* stop = nullptr;
    const Expr* step = nullptr;
    if (!op_is(":") && !op_is("]") && !op_is(",")) stop = parse_expression();
    if (skip_op(":") && !op_is("]") && !op_is(",")) step = parse_expression();
    Expr* e = make(Ex::Slice, line);
    e->args = {start, stop, step};
    return e;
  }

  void call_args(Expr* e) {
    expect_op("(");
    bool comma = false;
    while (!op_is(")")) {
      if (comma) {
        expect_op(",");
        if (op_is(")")) break;
      }
      if (op_is("*") || op_is("**")) error("unsupported construct: '*' and '**' call arguments");
      if (cur().type == Tok::Name && peek().type == Tok::Op && peek().text == "=") {
        const int key = p_.intern(cur().text);
        advance();
        advance();
        e->kwargs.emplace_back(key, parse_expression());
      } else {
        if (!e->kwargs.empty()) error("invalid syntax for function call expression");
        e->args.push_back(parse_expression());
      }
      comma = true;
    }
    advance();
  }

  const Expr* parse_call(const Expr* callee) {
    const int line = cur().line;
    if (callee->kind == Ex::Name) {
      if (callee->name != kCaller && (callee->name < kRange || callee->name > kStrftimeNow)) {
        calls_.emplace_back(callee->name, line);
      }
    } else if (callee->kind == Ex::Getattr) {
      if (!one_of(*p_.names[callee->name], kCallableMethods)) {
        syntax_error(line, "unsupported method '." + *p_.names[callee->name] + "()'");
      }
    } else {
      syntax_error(line, "unsupported call expression");
    }
    Expr* e = make(Ex::Call, line);
    e->args.push_back(callee);
    call_args(e);
    return e;
  }

  // A dotted name, as jinja2 reads filter and test names.
  std::string dotted_name() {
    std::string name = expect_identifier();
    while (skip_op(".")) name += "." + expect_identifier();
    return name;
  }

  const Expr* parse_filter(const Expr* subject) {
    while (skip_op("|")) {
      const int line = cur().line;
      const std::string name = dotted_name();
      const auto id = filter_named(name);
      if (!id) syntax_error(line, "unsupported filter '" + name + "'");
      Expr* e = make(Ex::Filter, line);
      e->op = static_cast<uint8_t>(*id);
      e->args.push_back(subject);
      if (op_is("(")) call_args(e);
      subject = e;
    }
    return subject;
  }

  const Expr* parse_test(const Expr* subject) {
    const int line = cur().line;
    advance();  // is
    const bool negated = skip_name("not");
    const std::string name = dotted_name();
    const auto id = test_named(name);
    if (!id) syntax_error(line, "unsupported test '" + name + "'");
    Expr* e = make(Ex::Test, line);
    e->op = static_cast<uint8_t>(*id);
    e->args.push_back(subject);
    const Token& t = cur();
    const bool bare_argument =
        (t.type == Tok::Name && t.text != "else" && t.text != "or" && t.text != "and") || t.type == Tok::String ||
        t.type == Tok::Int || t.type == Tok::Float || (t.type == Tok::Op && (t.text == "[" || t.text == "{"));
    if (op_is("(")) {
      call_args(e);
    } else if (bare_argument) {
      if (name_is("is")) error("You cannot chain multiple tests with is");
      e->args.push_back(parse_postfix(parse_primary()));
    }
    if (!negated) return e;
    Expr* n = make(Ex::Not, line);
    n->args = {e};
    return n;
  }

  std::vector<Token> t_;
  size_t pos_ = 0;
  Program& p_;
  int loop_depth_ = 0;
  int depth_ = 0;
  std::vector<MacroDef*> open_macros_;
  std::unordered_set<int> macro_names_;
  std::vector<std::pair<int, int>> calls_;  // calls to names that must be macros: (id, line)
};

// ---------------------------------------------------------------------------
// Runtime.

struct Scope {
  explicit Scope(std::shared_ptr<Scope> up) : parent(std::move(up)) {}
  Value* find(int id) {
    for (auto& v : vars) {
      if (v.first == id) return &v.second;
    }
    return nullptr;
  }
  void set(int id, Value value) {
    if (Value* v = find(id)) {
      *v = std::move(value);
    } else {
      vars.emplace_back(id, std::move(value));
    }
  }
  std::shared_ptr<Scope> parent;
  std::vector<std::pair<int, Value>> vars;
};
using ScopePtr = std::shared_ptr<Scope>;

constexpr std::string_view kStrAttributes[] = {
    "capitalize", "casefold",  "center",     "count",        "encode",     "endswith",  "expandtabs", "find",
    "format",     "format_map", "index",     "isalnum",      "isalpha",    "isascii",   "isdecimal",  "isdigit",
    "isidentifier", "islower", "isnumeric",  "isprintable",  "isspace",    "istitle",   "isupper",    "join",
    "ljust",      "lower",     "lstrip",     "maketrans",    "partition",  "removeprefix", "removesuffix",
    "replace",    "rfind",     "rindex",     "rjust",        "rpartition", "rsplit",    "rstrip",     "split",
    "splitlines", "startswith", "strip",     "swapcase",     "title",      "translate", "upper",      "zfill",
};
constexpr std::string_view kDictAttributes[] = {"clear", "copy", "fromkeys", "get",    "items",     "keys",
                                                "pop",   "popitem", "setdefault", "update", "values"};
constexpr std::string_view kDictMutators[] = {"clear", "pop", "popitem", "setdefault", "update"};
constexpr std::string_view kListAttributes[] = {"append", "clear", "copy",   "count",   "extend", "index",
                                                "insert", "pop",   "remove", "reverse", "sort"};
constexpr std::string_view kListMutators[] = {"append", "clear",  "extend",  "insert",
                                              "pop",    "remove", "reverse", "sort"};
constexpr std::string_view kNumberAttributes[] = {"as_integer_ratio", "bit_count", "bit_length", "conjugate",
                                                  "denominator",      "from_bytes", "fromhex",   "hex",
                                                  "imag",             "is_integer", "numerator", "real",
                                                  "to_bytes"};

// Lowercases a sort or unique key when case_sensitive is false.
Value ignore_case(const Value& v) { return v.kind == Kind::String ? str_value(py_lower(*v.text)) : v; }

class Renderer {
 public:
  explicit Renderer(const Program& program) : p_(program) {}

  std::string render(const std::vector<ChatMessage>& messages, const ChatTemplateContext& context) {
    auto root = std::make_shared<Scope>(nullptr);
    // Variables the template never mentions cannot be read, so they are skipped.
    auto bind = [&](const std::string& name, Value v) {
      const int id = p_.id_of(name);
      if (id >= 0) root->set(id, std::move(v));
    };
    auto text = [](const std::string& s) {
      LK_CHECK(valid_utf8(s), "chat template input is not valid UTF-8");
      return str_value(s);
    };
    std::vector<Value> list;
    list.reserve(messages.size());
    for (const ChatMessage& m : messages) {
      list.push_back(
          dict_value({{str_value(p_.role_key), text(m.role)}, {str_value(p_.content_key), text(m.content)}}));
    }
    // transformers passes tools and documents as None; unset special tokens stay undefined.
    bind("messages", list_value(std::move(list)));
    bind("tools", none_value());
    bind("documents", none_value());
    bind("add_generation_prompt", bool_value(context.add_generation_prompt));
    if (!context.bos_token.empty()) bind("bos_token", text(context.bos_token));
    if (!context.eos_token.empty()) bind("eos_token", text(context.eos_token));
    for (const auto& [name, value] : context.strings) bind(name, text(value));
    for (const auto& [name, value] : context.flags) bind(name, bool_value(value));
    std::string out;
    try {
      exec(p_.body, root, out);
    } catch (const RuntimeFailure& f) {
      fail("lokahi: chat template line " + std::to_string(std::max(line_, 1)) + ": " + f.message);
    }
    return out;
  }

 private:
  enum class Flow : uint8_t { Normal, Break, Continue };
  struct Args {
    std::vector<Value> pos;
    std::vector<std::pair<int, Value>> kw;
  };
  using Bound = std::vector<std::optional<Value>>;
  static constexpr int kMaxCallDepth = 200;

  const std::string& name(int id) const { return *p_.names[static_cast<size_t>(id)]; }

  // Statements --------------------------------------------------------------

  Flow exec(const Body& body, const ScopePtr& scope, std::string& out) {
    for (const Node* np : body) {
      const Node& n = *np;
      line_ = n.line;
      switch (n.kind) {
        case St::Text: out += n.text; break;
        case St::Output: append_str(out, eval(*n.expr, scope)); break;
        case St::If: {
          const Body* chosen = &n.else_body;
          for (const auto& [test, branch] : n.branches) {
            if (truthy(eval(*test, scope))) {
              chosen = &branch;
              break;
            }
          }
          const Flow flow = exec(*chosen, scope, out);
          if (flow != Flow::Normal) return flow;
          break;
        }
        case St::For: exec_for(n, scope, out); break;
        case St::Set: assign(n.target, eval(*n.expr, scope), *scope); break;
        case St::SetAttr: {
          Value ns = lookup(n.name, *scope);
          if (ns.kind != Kind::Namespace) runtime_fail("cannot assign attribute on non-namespace object");
          Value value = eval(*n.expr, scope);
          auto& attrs = as_namespace(ns).attrs;
          auto it = std::find_if(attrs.begin(), attrs.end(), [&](const auto& a) { return a.first == name(n.attr); });
          if (it != attrs.end()) {
            it->second = std::move(value);
          } else {
            attrs.emplace_back(name(n.attr), std::move(value));
          }
          break;
        }
        case St::SetBlock: {
          std::string text;
          exec(n.body, std::make_shared<Scope>(scope), text);
          scope->set(n.target.name, str_value(std::move(text)));
          break;
        }
        case St::Macro: scope->set(n.name, macro_value(n.macro, scope)); break;
        case St::CallBlock: {
          const Value caller = macro_value(n.macro, scope);
          append_str(out, call(*n.expr, scope, &caller));
          break;
        }
        case St::Break: return Flow::Break;
        case St::Continue: return Flow::Continue;
        case St::Generation: exec(n.body, std::make_shared<Scope>(scope), out); break;
      }
    }
    return Flow::Normal;
  }

  static Value macro_value(const MacroDef* def, const ScopePtr& scope) {
    auto m = std::make_shared<MacroObject>();
    m->def = def;
    m->closure = scope;
    return object_value(Kind::Macro, std::move(m));
  }

  void exec_for(const Node& n, const ScopePtr& scope, std::string& out) {
    std::vector<Value> items = iterate(eval(*n.expr, scope));
    if (n.test) {
      std::vector<Value> kept;
      for (Value& item : items) {
        auto filter_scope = std::make_shared<Scope>(scope);
        assign(n.target, item, *filter_scope);
        if (truthy(eval(*n.test, filter_scope))) kept.push_back(std::move(item));
      }
      items = std::move(kept);
    }
    if (items.empty()) {
      if (!n.else_body.empty()) exec(n.else_body, std::make_shared<Scope>(scope), out);
      return;
    }
    auto loop = std::make_shared<LoopObject>();
    loop->items = std::move(items);
    const Value loop_value = object_value(Kind::Loop, loop);
    for (size_t i = 0; i < loop->items.size(); ++i) {
      loop->index = i;
      auto iteration = std::make_shared<Scope>(scope);  // a set inside the body does not leak
      iteration->vars.reserve(4);
      assign(n.target, loop->items[i], *iteration);
      iteration->set(kLoop, loop_value);
      if (exec(n.body, iteration, out) == Flow::Break) break;
    }
  }

  void assign(const Target& t, const Value& v, Scope& scope) {
    if (t.name >= 0) {
      scope.set(t.name, v);
      return;
    }
    const std::vector<Value> items = iterate(v);
    if (items.size() != t.items.size()) {
      runtime_fail(items.size() > t.items.size()
                       ? "too many values to unpack (expected " + std::to_string(t.items.size()) + ")"
                       : "not enough values to unpack (expected " + std::to_string(t.items.size()) + ", got " +
                             std::to_string(items.size()) + ")");
    }
    for (size_t i = 0; i < items.size(); ++i) assign(t.items[i], items[i], scope);
  }

  // Expressions -------------------------------------------------------------

  Value lookup(int id, const Scope& scope) const {
    for (const Scope* s = &scope; s; s = s->parent.get()) {
      for (const auto& [key, value] : s->vars) {
        if (key == id) return value;
      }
    }
    if (id >= kRange && id <= kStrftimeNow) return builtin_value(static_cast<Builtin>(id - kRange));
    return undefined_value(Missing::Name, p_.names[static_cast<size_t>(id)]);
  }

  Value eval(const Expr& e, const ScopePtr& scope) {
    line_ = e.line;
    switch (e.kind) {
      case Ex::Const: return e.constant;
      case Ex::Name: return lookup(e.name, *scope);
      case Ex::Getattr: return getattr(eval(*e.args[0], scope), p_.names[static_cast<size_t>(e.name)]);
      case Ex::Getitem: {
        const Value object = eval(*e.args[0], scope);
        if (e.args[1]->kind == Ex::Slice) return slice(object, *e.args[1], scope);
        return getitem(object, eval(*e.args[1], scope));
      }
      case Ex::Slice: runtime_fail("a slice outside a subscript");
      case Ex::Call: return call(e, scope, nullptr);
      case Ex::Filter: {
        Value subject = eval(*e.args[0], scope);
        Args args = eval_args(e, scope);
        line_ = e.line;
        return filter(static_cast<FilterId>(e.op), std::move(subject), std::move(args));
      }
      case Ex::Test: {
        const Value subject = eval(*e.args[0], scope);
        Args args = eval_args(e, scope);
        line_ = e.line;
        return bool_value(test(static_cast<TestId>(e.op), subject, std::move(args)));
      }
      case Ex::Not: return bool_value(!truthy(eval(*e.args[0], scope)));
      case Ex::Neg:
      case Ex::Pos: return unary(e.kind == Ex::Neg, eval(*e.args[0], scope));
      case Ex::And: {
        Value left = eval(*e.args[0], scope);
        return truthy(left) ? eval(*e.args[1], scope) : left;
      }
      case Ex::Or: {
        Value left = eval(*e.args[0], scope);
        return truthy(left) ? left : eval(*e.args[1], scope);
      }
      case Ex::Bin: {
        const Value left = eval(*e.args[0], scope);
        const Value right = eval(*e.args[1], scope);
        line_ = e.line;
        return binary(static_cast<BinOp>(e.op), left, right);
      }
      case Ex::Concat: {  // str() of each operand; never Markup
        std::string s;
        for (const Expr* arg : e.args) append_str(s, eval(*arg, scope));
        return str_value(std::move(s));
      }
      case Ex::Compare: {
        Value left = eval(*e.args[0], scope);
        for (size_t i = 0; i < e.ops.size(); ++i) {
          Value right = eval(*e.args[i + 1], scope);
          line_ = e.line;
          if (!compare(e.ops[i], left, right)) return bool_value(false);
          left = std::move(right);
        }
        return bool_value(true);
      }
      case Ex::Cond:
        if (truthy(eval(*e.args[0], scope))) return eval(*e.args[1], scope);
        if (e.args[2]) return eval(*e.args[2], scope);
        return undefined_hint("the inline if-expression on line " + std::to_string(e.line) +
                              " evaluated to false and no else section was defined.");
      case Ex::List:
      case Ex::Tuple: {
        std::vector<Value> items;
        items.reserve(e.args.size());
        for (const Expr* arg : e.args) items.push_back(eval(*arg, scope));
        return list_value(std::move(items), e.kind == Ex::List ? Flavor::List : Flavor::Tuple);
      }
      case Ex::Dict: {
        std::vector<std::pair<Value, Value>> items;
        for (size_t i = 0; i < e.args.size(); i += 2) {
          Value key = eval(*e.args[i], scope);
          Value value = eval(*e.args[i + 1], scope);
          if (!hashable(key)) runtime_fail(std::string("unhashable type: '") + type_name(key) + "'");
          auto it = std::find_if(items.begin(), items.end(), [&](const auto& p) { return py_equal(p.first, key); });
          if (it != items.end()) {
            it->second = std::move(value);  // the first key object is kept
          } else {
            items.emplace_back(std::move(key), std::move(value));
          }
        }
        return dict_value(std::move(items));
      }
    }
    runtime_fail("unknown expression");
  }

  Args eval_args(const Expr& e, const ScopePtr& scope) {
    Args a;
    a.pos.reserve(e.args.size() - 1);
    for (size_t i = 1; i < e.args.size(); ++i) a.pos.push_back(eval(*e.args[i], scope));
    for (const auto& [key, arg] : e.kwargs) a.kw.emplace_back(key, eval(*arg, scope));
    return a;
  }

  // str() of a value, appended.
  void append_str(std::string& out, const Value& v) {
    if (v.kind == Kind::String) {
      out += *v.text;
    } else if (v.kind == Kind::Macro) {
      out += "<Macro '" + name(static_cast<const MacroObject&>(*v.object).def->name) + "'>";
    } else if (v.kind != Kind::Undefined) {
      append_repr(out, v);
    }
  }

  std::string to_str(const Value& v) {
    if (v.kind == Kind::String) return *v.text;
    std::string out;
    append_str(out, v);
    return out;
  }

  // jinja2's soft_str: strings (and Markup) unchanged, anything else str().
  Value soft_str(const Value& v) { return v.kind == Kind::String ? v : str_value(to_str(v)); }

  bool compare(CmpOp op, const Value& a, const Value& b) {
    switch (op) {
      case CmpOp::Eq: return py_equal(a, b);
      case CmpOp::Ne: return !py_equal(a, b);
      case CmpOp::In: return contains(b, a);
      case CmpOp::NotIn: return !contains(b, a);
      default: return py_order(op, a, b);
    }
  }

  Value unary(bool negate, const Value& v) {
    if (v.kind == Kind::Undefined) fail_undefined(v);
    if (v.kind == Kind::Float) return float_value(negate ? -v.number : v.number);
    if (!is_int_like(v)) runtime_fail(std::string("bad operand type for unary ") + (negate ? "-" : "+") + ": '" +
                                      type_name(v) + "'");
    const int64_t i = as_int(v);
    if (negate && i == INT64_MIN) runtime_fail("integers beyond 64 bits are not supported");
    return int_value(negate ? -i : i);
  }

  static Value repeat(const Value& sequence, int64_t count) {
    count = std::max<int64_t>(count, 0);
    if (sequence.kind == Kind::String) {
      if (static_cast<double>(sequence.text->size()) * static_cast<double>(count) > (1 << 26)) {
        runtime_fail("repeated string is too large");
      }
      std::string out;
      for (int64_t i = 0; i < count; ++i) out += *sequence.text;
      return str_value(std::move(out), sequence.aux);
    }
    const ListObject& list = as_list(sequence);
    if (static_cast<double>(list.items.size()) * static_cast<double>(count) > (1 << 22)) {
      runtime_fail("repeated list is too large");
    }
    std::vector<Value> out;
    for (int64_t i = 0; i < count; ++i) out.insert(out.end(), list.items.begin(), list.items.end());
    return list_value(std::move(out), list.flavor);
  }

  Value binary(BinOp op, const Value& a, const Value& b) {
    if (a.kind == Kind::Undefined) fail_undefined(a);
    if (b.kind == Kind::Undefined) fail_undefined(b);
    if (is_number(a) && is_number(b)) return arithmetic(op, a, b);
    if (op == BinOp::Add && a.kind == Kind::String && b.kind == Kind::String) {
      // Markup + str escapes the str, in either order.
      const bool markup = a.aux || b.aux;
      std::string out = a.aux || !markup ? *a.text : html_escape(*a.text);
      out += b.aux || !markup ? *b.text : html_escape(*b.text);
      return str_value(std::move(out), markup);
    }
    if (op == BinOp::Add && is_list_or_tuple(a) && is_list_or_tuple(b) && as_list(a).flavor == as_list(b).flavor) {
      std::vector<Value> items = as_list(a).items;
      items.insert(items.end(), as_list(b).items.begin(), as_list(b).items.end());
      return list_value(std::move(items), as_list(a).flavor);
    }
    if (op == BinOp::Mul) {
      if ((a.kind == Kind::String || is_list_or_tuple(a)) && is_int_like(b)) return repeat(a, as_int(b));
      if ((b.kind == Kind::String || is_list_or_tuple(b)) && is_int_like(a)) return repeat(b, as_int(a));
    }
    if (op == BinOp::Mod && a.kind == Kind::String) runtime_fail("printf-style '%' string formatting is not supported");
    static const char* const kSymbols[] = {"+", "-", "*", "/", "//", "%", "**"};
    runtime_fail(std::string("unsupported operand type(s) for ") + kSymbols[static_cast<int>(op)] + ": '" +
                 type_name(a) + "' and '" + type_name(b) + "'");
  }

  static Value arithmetic(BinOp op, const Value& a, const Value& b) {
    if (a.kind != Kind::Float && b.kind != Kind::Float) {
      const int64_t x = as_int(a), y = as_int(b);
      int64_t r = 0;
      auto overflow = [] { runtime_fail("integers beyond 64 bits are not supported"); };
      switch (op) {
        case BinOp::Add:
          if (__builtin_add_overflow(x, y, &r)) overflow();
          return int_value(r);
        case BinOp::Sub:
          if (__builtin_sub_overflow(x, y, &r)) overflow();
          return int_value(r);
        case BinOp::Mul:
          if (__builtin_mul_overflow(x, y, &r)) overflow();
          return int_value(r);
        case BinOp::Div: {
          if (y == 0) runtime_fail("division by zero");
          constexpr int64_t kExact = int64_t{1} << 53;
          if (x > kExact || x < -kExact || y > kExact || y < -kExact) {
            runtime_fail("dividing integers beyond 2**53 is not supported");
          }
          return float_value(static_cast<double>(x) / static_cast<double>(y));
        }
        case BinOp::FloorDiv: {
          if (y == 0) runtime_fail("integer division or modulo by zero");
          if (x == INT64_MIN && y == -1) overflow();
          int64_t q = x / y;
          if (x % y != 0 && ((x < 0) != (y < 0))) --q;
          return int_value(q);
        }
        case BinOp::Mod: {
          if (y == 0) runtime_fail("integer division or modulo by zero");
          if (y == -1) return int_value(0);
          r = x % y;
          if (r != 0 && ((r < 0) != (y < 0))) r += y;
          return int_value(r);
        }
        case BinOp::Pow: {
          if (y < 0) {
            if (x == 0) runtime_fail("0.0 cannot be raised to a negative power");
            return float_value(std::pow(static_cast<double>(x), static_cast<double>(y)));
          }
          int64_t result = 1, base = x, e = y;
          while (e > 0) {
            if ((e & 1) && __builtin_mul_overflow(result, base, &result)) overflow();
            e >>= 1;
            if (e > 0 && __builtin_mul_overflow(base, base, &base)) overflow();
          }
          return int_value(result);
        }
      }
    }
    const double x = as_float(a), y = as_float(b);
    switch (op) {
      case BinOp::Add: return float_value(x + y);
      case BinOp::Sub: return float_value(x - y);
      case BinOp::Mul: return float_value(x * y);
      case BinOp::Div:
        if (y == 0) runtime_fail("float division by zero");
        return float_value(x / y);
      case BinOp::FloorDiv:
      case BinOp::Mod: {
        // CPython's float_divmod.
        if (y == 0) runtime_fail(op == BinOp::Mod ? "float modulo by zero" : "float floor division by zero");
        double mod = std::fmod(x, y);
        double div = (x - mod) / y;
        if (mod != 0) {
          if ((y < 0) != (mod < 0)) {
            mod += y;
            div -= 1.0;
          }
        } else {
          mod = std::copysign(0.0, y);
        }
        double floordiv;
        if (div != 0) {
          floordiv = std::floor(div);
          if (div - floordiv > 0.5) floordiv += 1.0;
        } else {
          floordiv = std::copysign(0.0, x / y);
        }
        return float_value(op == BinOp::Mod ? mod : floordiv);
      }
      case BinOp::Pow: {
        if (x == 0 && y < 0) runtime_fail("0.0 cannot be raised to a negative power");
        if (x < 0 && std::isfinite(y) && y != std::floor(y)) runtime_fail("complex powers are not supported");
        const double r = std::pow(x, y);
        if (std::isinf(r) && std::isfinite(x) && std::isfinite(y)) runtime_fail("numerical result out of range");
        return float_value(r);
      }
    }
    runtime_fail("unknown operator");
  }

  // Attribute and item access with the sandbox's rules: obj.name tries the
  // Python attribute first and then the item; obj[key] the other way round.
  Value method_value(const Value& self, const std::string& method) {
    auto m = std::make_shared<MethodObject>();
    m->self = self;
    m->name = method;
    return object_value(Kind::Method, std::move(m));
  }

  static Value unsafe(const Value& owner, const std::string& attr) {
    return undefined_hint("access to attribute '" + attr + "' of '" + type_name(owner) + "' object is unsafe.");
  }

  Value getattr(const Value& object, const Str& attr) {
    const std::string& n = *attr;
    const bool dunder = n.size() > 4 && n.compare(0, 2, "__") == 0 && n.compare(n.size() - 2, 2, "__") == 0;
    auto missing = [&] { return undefined_value(Missing::Attribute, attr, type_name(object)); };
    if (dunder && object.kind != Kind::Undefined) return missing();
    switch (object.kind) {
      case Kind::Undefined: fail_undefined(object);
      case Kind::Dict:
        if (one_of(n, kDictAttributes)) return one_of(n, kDictMutators) ? unsafe(object, n) : method_value(object, n);
        if (const Value* v = dict_find_str(object, n)) return *v;
        return missing();
      case Kind::Namespace:
        for (const auto& [key, value] : as_namespace(object).attrs) {
          if (key == n) return value;
        }
        return missing();
      case Kind::String: return one_of(n, kStrAttributes) ? method_value(object, n) : missing();
      case Kind::List: {
        const ListObject& list = as_list(object);
        switch (list.flavor) {
          case Flavor::List:
            if (one_of(n, kListAttributes)) {
              return one_of(n, kListMutators) ? unsafe(object, n) : method_value(object, n);
            }
            break;
          case Flavor::Tuple:
            if (n == "count" || n == "index") return method_value(object, n);
            break;
          case Flavor::Range:
            if (n == "start") return int_value(list.start);
            if (n == "stop") return int_value(list.stop);
            if (n == "step") return int_value(list.step);
            if (n == "count" || n == "index") return method_value(object, n);
            break;
          case Flavor::Items:
          case Flavor::Keys:
          case Flavor::Values:
            if (n == "mapping" || n == "isdisjoint") {
              runtime_fail("attribute '" + n + "' of a dict view is not supported");
            }
            break;
          case Flavor::Iterator:
            if (n == "send" || n == "throw" || n == "close" || n.compare(0, 3, "gi_") == 0) {
              runtime_fail("attribute '" + n + "' of a generator is not supported");
            }
            break;
        }
        return missing();
      }
      case Kind::Loop: {
        const LoopObject& loop = static_cast<const LoopObject&>(*object.object);
        const int64_t i = static_cast<int64_t>(loop.index), length = static_cast<int64_t>(loop.items.size());
        if (n == "index") return int_value(i + 1);
        if (n == "index0") return int_value(i);
        if (n == "revindex") return int_value(length - i);
        if (n == "revindex0") return int_value(length - i - 1);
        if (n == "first") return bool_value(i == 0);
        if (n == "last") return bool_value(i == length - 1);
        if (n == "length") return int_value(length);
        if (n == "depth") return int_value(1);
        if (n == "depth0") return int_value(0);
        if (n == "previtem") return i > 0 ? loop.items[loop.index - 1] : undefined_hint("there is no previous item");
        if (n == "nextitem") {
          return i + 1 < length ? loop.items[loop.index + 1] : undefined_hint("there is no next item");
        }
        if (n == "cycle" || n == "changed") return method_value(object, n);
        return missing();
      }
      case Kind::Bool:
      case Kind::Int:
      case Kind::Float:
        if (one_of(n, kNumberAttributes)) runtime_fail("attribute '" + n + "' of a number is not supported");
        return missing();
      case Kind::Macro:
        if (n == "name" || n == "arguments" || n == "catch_kwargs" || n == "catch_varargs" || n == "caller") {
          runtime_fail("attribute '" + n + "' of a macro is not supported");
        }
        return missing();
      default: return missing();
    }
  }

  Value missing_key(const Value& object, const Value& key) {
    std::string text;
    append_repr(text, key);
    return undefined_value(Missing::Key, make_text(std::move(text)), type_name(object));
  }

  Value getitem(const Value& object, const Value& key) {
    switch (object.kind) {
      case Kind::Undefined: fail_undefined(object);
      case Kind::Dict:
        if (hashable(key)) {
          if (const Value* v = dict_find(object, key)) return *v;
        }
        break;
      case Kind::List: {
        const ListObject& list = as_list(object);
        if (list.flavor != Flavor::List && list.flavor != Flavor::Tuple && list.flavor != Flavor::Range) break;
        if (!is_int_like(key)) break;
        int64_t i = as_int(key);
        const int64_t n = static_cast<int64_t>(list.items.size());
        if (i < 0) i += n;
        if (i >= 0 && i < n) return list.items[static_cast<size_t>(i)];
        return missing_key(object, key);
      }
      case Kind::String: {
        if (!is_int_like(key)) break;
        const Text text(*object.text);
        int64_t i = as_int(key);
        if (i < 0) i += text.size();
        if (i >= 0 && i < text.size()) return str_value(std::string(text.sub(i, i + 1)), object.aux);
        return missing_key(object, key);
      }
      default: break;
    }
    if (key.kind == Kind::String) return getattr(object, key.text);
    return missing_key(object, key);
  }

  // Slices bypass the sandbox's getitem: jinja2 compiles them to a plain
  // Python subscript, so anything but a string, list or tuple raises. (Its
  // constant folder slices literals through getitem instead and can yield
  // undefined; that case fails here too.)
  Value slice(const Value& object, const Expr& s, const ScopePtr& scope) {
    std::optional<int64_t> part[3];
    bool integers = true;
    for (int i = 0; i < 3; ++i) {
      if (!s.args[static_cast<size_t>(i)]) continue;
      const Value v = eval(*s.args[static_cast<size_t>(i)], scope);
      if (v.kind == Kind::None) continue;
      if (is_int_like(v)) {
        part[i] = as_int(v);
      } else {
        integers = false;
      }
    }
    line_ = s.line;
    if (object.kind == Kind::Undefined) fail_undefined(object);
    if (object.kind == Kind::List && as_list(object).flavor == Flavor::Range) {
      runtime_fail("slicing a range is not supported");
    }
    if (object.kind != Kind::String && !is_list_or_tuple(object)) {
      runtime_fail(std::string("slicing a '") + type_name(object) + "' object is not supported");
    }
    if (!integers) runtime_fail("slice indices other than integers or None are not supported");
    const int64_t step = part[2].value_or(1);
    if (step == 0) runtime_fail("slice step cannot be zero");
    std::optional<Text> text;
    if (object.kind == Kind::String) text.emplace(*object.text);
    const int64_t n = text ? text->size() : static_cast<int64_t>(as_list(object).items.size());
    // PySlice_AdjustIndices.
    auto adjust = [&](std::optional<int64_t> v, int64_t fallback) {
      if (!v) return fallback;
      int64_t x = *v;
      if (x < 0) {
        x += n;
        if (x < 0) x = step < 0 ? -1 : 0;
      } else if (x >= n) {
        x = step < 0 ? n - 1 : n;
      }
      return x;
    };
    const int64_t start = adjust(part[0], step < 0 ? n - 1 : 0);
    const int64_t stop = adjust(part[1], step < 0 ? -1 : n);
    if (text) {
      std::string out;
      for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) out.append(text->sub(i, i + 1));
      return str_value(std::move(out), object.aux);
    }
    const ListObject& list = as_list(object);
    std::vector<Value> out;
    for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) {
      out.push_back(list.items[static_cast<size_t>(i)]);
    }
    return list_value(std::move(out), list.flavor);
  }

  // Calls -------------------------------------------------------------------

  Value call(const Expr& e, const ScopePtr& scope, const Value* caller) {
    const Value callee = eval(*e.args[0], scope);
    Args args = eval_args(e, scope);
    if (caller) args.kw.emplace_back(kCaller, *caller);
    line_ = e.line;
    switch (callee.kind) {
      case Kind::Macro: return call_macro(static_cast<const MacroObject&>(*callee.object), std::move(args));
      case Kind::Method: return call_method(static_cast<const MethodObject&>(*callee.object), std::move(args));
      case Kind::Function: return call_builtin(static_cast<Builtin>(callee.integer), std::move(args));
      case Kind::Undefined: fail_undefined(callee);
      default: runtime_fail(std::string("'") + type_name(callee) + "' object is not callable");
    }
  }

  // Python argument binding: positional arguments, then keywords by name.
  // Parameters before `positional_only` cannot be passed by keyword.
  Bound bind(Args& a, std::initializer_list<std::string_view> params, size_t positional_only, const char* fn) {
    if (a.pos.size() > params.size()) {
      runtime_fail(std::string(fn) + "() takes at most " + std::to_string(params.size()) + " arguments (" +
                   std::to_string(a.pos.size()) + " given)");
    }
    Bound out(params.size());
    for (size_t i = 0; i < a.pos.size(); ++i) out[i] = std::move(a.pos[i]);
    for (auto& [key, value] : a.kw) {
      size_t i = 0;
      for (std::string_view param : params) {
        if (param == name(key)) break;
        ++i;
      }
      if (i == params.size() || i < positional_only) {
        runtime_fail(std::string(fn) + "() got an unexpected keyword argument '" + name(key) + "'");
      }
      if (out[i]) runtime_fail(std::string(fn) + "() got multiple values for argument '" + name(key) + "'");
      out[i] = std::move(value);
    }
    return out;
  }

  static const Value& required(const Bound& b, size_t i, const char* fn) {
    if (!b[i]) runtime_fail(std::string(fn) + "() is missing a required argument");
    return *b[i];
  }

  static int64_t int_arg(const Value& v, const char* fn) {
    if (!is_int_like(v)) runtime_fail(std::string(fn) + "() expects an integer, not '" + type_name(v) + "'");
    return as_int(v);
  }

  static const std::string& str_arg(const Value& v, const char* fn) {
    if (v.kind != Kind::String) runtime_fail(std::string(fn) + "() expects a string, not '" + type_name(v) + "'");
    return *v.text;
  }

  Value call_macro(const MacroObject& m, Args a) {
    const MacroDef& d = *m.def;
    const std::string& macro = name(d.name);
    const ScopePtr closure = m.closure.lock();
    if (!closure) runtime_fail("macro '" + macro + "' was called after its scope ended");
    if (depth_ >= kMaxCallDepth) runtime_fail("maximum macro recursion depth exceeded");
    struct Depth {
      int& depth;
      ~Depth() { --depth; }
    } depth{++depth_};
    // jinja2's Macro.__call__.
    const size_t count = d.params.size();
    Bound values(count);
    const size_t positional = std::min(a.pos.size(), count);
    for (size_t i = 0; i < positional; ++i) values[i] = a.pos[i];
    auto take = [&](int id) -> std::optional<Value> {
      for (auto it = a.kw.begin(); it != a.kw.end(); ++it) {
        if (it->first == id) {
          Value v = std::move(it->second);
          a.kw.erase(it);
          return v;
        }
      }
      return std::nullopt;
    };
    bool found_caller = d.explicit_caller;
    if (positional != count) {
      found_caller = false;
      for (size_t i = positional; i < count; ++i) {
        values[i] = take(d.params[i]);
        if (d.params[i] == kCaller) found_caller = true;
      }
    }
    auto scope = std::make_shared<Scope>(closure);
    if (d.uses_caller && !found_caller) {
      std::optional<Value> caller = take(kCaller);
      scope->set(kCaller, caller && caller->kind != Kind::None ? *caller : undefined_hint("No caller defined"));
    }
    if (d.catch_kwargs) {
      std::vector<std::pair<Value, Value>> kwargs;
      for (auto& [key, value] : a.kw) kwargs.emplace_back(str_value(p_.names[static_cast<size_t>(key)]), value);
      scope->set(kKwargs, dict_value(std::move(kwargs)));
    } else if (!a.kw.empty()) {
      runtime_fail("macro '" + macro + "' takes no keyword argument '" + name(a.kw.front().first) + "'");
    }
    if (d.catch_varargs) {
      std::vector<Value> rest;
      for (size_t i = count; i < a.pos.size(); ++i) rest.push_back(a.pos[i]);
      scope->set(kVarargs, list_value(std::move(rest), Flavor::Tuple));
    } else if (a.pos.size() > count) {
      runtime_fail("macro '" + macro + "' takes not more than " + std::to_string(count) + " argument(s)");
    }
    for (size_t i = 0; i < count; ++i) {
      if (!values[i]) {
        values[i] = d.defaults[i] ? eval(*d.defaults[i], scope)
                                  : undefined_hint("parameter '" + name(d.params[i]) + "' was not provided");
      }
      scope->set(d.params[i], std::move(*values[i]));
    }
    std::string out;
    exec(d.body, scope, out);
    return str_value(std::move(out));
  }

  Value call_builtin(Builtin b, Args a) {
    switch (b) {
      case Builtin::Range: {
        if (!a.kw.empty()) runtime_fail("range() takes no keyword arguments");
        if (a.pos.empty() || a.pos.size() > 3) runtime_fail("range expected 1 to 3 arguments");
        int64_t start = 0, stop = 0, step = 1;
        if (a.pos.size() == 1) {
          stop = int_arg(a.pos[0], "range");
        } else {
          start = int_arg(a.pos[0], "range");
          stop = int_arg(a.pos[1], "range");
          if (a.pos.size() == 3) step = int_arg(a.pos[2], "range");
        }
        if (step == 0) runtime_fail("range() arg 3 must not be zero");
        const double length = std::max(0.0, std::ceil((static_cast<double>(stop) - static_cast<double>(start)) /
                                                      static_cast<double>(step)));
        if (length > 100000) runtime_fail("Range too big. The sandbox blocks ranges larger than MAX_RANGE (100000).");
        std::vector<Value> items;
        for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) items.push_back(int_value(i));
        Value v = list_value(std::move(items), Flavor::Range);
        as_list(v).start = start;
        as_list(v).stop = stop;
        as_list(v).step = step;
        return v;
      }
      case Builtin::Dict:
      case Builtin::Namespace: {
        std::vector<std::pair<Value, Value>> items;
        auto put = [&](Value key, Value value) {
          auto it = std::find_if(items.begin(), items.end(), [&](const auto& p) { return py_equal(p.first, key); });
          if (it != items.end()) {
            it->second = std::move(value);
          } else {
            items.emplace_back(std::move(key), std::move(value));
          }
        };
        if (a.pos.size() > 1) runtime_fail("dict expected at most 1 argument");
        if (!a.pos.empty()) {
          const Value& source = a.pos[0];
          if (source.kind == Kind::Dict) {
            for (const auto& [key, value] : as_dict(source).items) put(key, value);
          } else {
            for (const Value& pair : iterate(source)) {
              const std::vector<Value> kv = iterate(pair);
              if (kv.size() != 2) runtime_fail("dictionary update sequence element has the wrong length");
              if (!hashable(kv[0])) runtime_fail(std::string("unhashable type: '") + type_name(kv[0]) + "'");
              put(kv[0], kv[1]);
            }
          }
        }
        for (auto& [key, value] : a.kw) put(str_value(p_.names[static_cast<size_t>(key)]), std::move(value));
        if (b == Builtin::Dict) return dict_value(std::move(items));
        auto ns = std::make_shared<NamespaceObject>();
        for (auto& [key, value] : items) {
          if (key.kind != Kind::String) runtime_fail("namespace attributes must be strings");
          ns->attrs.emplace_back(*key.text, std::move(value));
        }
        return object_value(Kind::Namespace, std::move(ns));
      }
      case Builtin::RaiseException: {
        Bound args = bind(a, {"message"}, 0, "raise_exception");
        // transformers raises TemplateError(message); its text is the message itself.
        fail(to_str(required(args, 0, "raise_exception")));
      }
      case Builtin::StrftimeNow: {
        Bound args = bind(a, {"format"}, 0, "strftime_now");
        return str_value(strftime_now(str_arg(required(args, 0, "strftime_now"), "strftime_now")));
      }
    }
    runtime_fail("unknown builtin");
  }

  // datetime.now().strftime(format): Python handles %f, %z and %Z of a naive
  // datetime itself and passes the rest to the C library.
  static std::string strftime_now(const std::string& format) {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const std::time_t seconds = system_clock::to_time_t(now);
    const auto micros = duration_cast<microseconds>(now.time_since_epoch()).count() % 1000000;
    std::tm tm{};
    localtime_r(&seconds, &tm);
    std::string out;
    for (size_t i = 0; i < format.size(); ++i) {
      if (format[i] != '%') {
        out += format[i];
        continue;
      }
      if (i + 1 >= format.size()) runtime_fail("strftime format ends with a lone '%'");
      const char d = format[++i];
      if (d == '%') {
        out += '%';
      } else if (d == 'f') {
        char buffer[8];
        std::snprintf(buffer, sizeof(buffer), "%06lld", static_cast<long long>(micros < 0 ? micros + 1000000 : micros));
        out += buffer;
      } else if (d == 'z' || d == 'Z') {
        // A naive datetime has no zone.
      } else if (d != 0 && std::strchr("aAbBcdDeFgGHIjmMnprRStTuUVwWxXyYC", d)) {
        const char spec[3] = {'%', d, 0};
        char buffer[128];
        const size_t n = std::strftime(buffer, sizeof(buffer), spec, &tm);
        out.append(buffer, n);
      } else {
        runtime_fail(std::string("unsupported strftime directive '%") + d + "'");
      }
    }
    return out;
  }

  // Code-point bounds [start, end) for str.find-style optional arguments
  // (CPython's ADJUST_INDICES).
  static void adjust_indices(const Bound& b, size_t first, int64_t length, int64_t* start, int64_t* end) {
    *start = 0;
    *end = length;
    if (b[first] && b[first]->kind != Kind::None) *start = int_arg(*b[first], "slice index");
    if (b[first + 1] && b[first + 1]->kind != Kind::None) *end = int_arg(*b[first + 1], "slice index");
    if (*end > length) {
      *end = length;
    } else if (*end < 0) {
      *end = std::max<int64_t>(*end + length, 0);
    }
    if (*start < 0) *start = std::max<int64_t>(*start + length, 0);
  }

  Value call_method(const MethodObject& m, Args a) {
    const Value& self = m.self;
    const std::string& method = m.name;
    const char* fn = method.c_str();
    if (self.kind == Kind::String) return string_method(self, method, std::move(a));
    if (self.kind == Kind::Dict) {
      const auto& items = as_dict(self).items;
      if (method == "get") {
        Bound b = bind(a, {"key", "default"}, 2, fn);
        const Value& key = required(b, 0, fn);
        if (!hashable(key)) runtime_fail(std::string("unhashable type: '") + type_name(key) + "'");
        if (const Value* v = dict_find(self, key)) return *v;
        return b[1] ? *b[1] : none_value();
      }
      if (method == "items" || method == "keys" || method == "values") {
        bind(a, {}, 0, fn);
        std::vector<Value> out;
        for (const auto& [key, value] : items) {
          if (method == "items") {
            out.push_back(list_value({key, value}, Flavor::Tuple));
          } else {
            out.push_back(method == "keys" ? key : value);
          }
        }
        return list_value(std::move(out), method == "items" ? Flavor::Items
                                                            : method == "keys" ? Flavor::Keys : Flavor::Values);
      }
    }
    if (self.kind == Kind::List && method == "count") {
      Bound b = bind(a, {"value"}, 1, fn);
      const Value& x = required(b, 0, fn);
      const auto& items = as_list(self).items;
      return int_value(std::count_if(items.begin(), items.end(), [&](const Value& v) { return py_equal(v, x); }));
    }
    runtime_fail("unsupported method '" + method + "' of a " + type_name(self));
  }

  Value string_method(const Value& self, const std::string& method, Args a) {
    const std::string& s = *self.text;
    const bool markup = self.aux;
    const char* fn = method.c_str();
    if (method == "startswith" || method == "endswith") {
      Bound b = bind(a, {"prefix", "start", "end"}, 3, fn);
      const Value& affix = required(b, 0, fn);
      const Text text(s);
      int64_t start, end;
      adjust_indices(b, 1, text.size(), &start, &end);
      std::vector<Value> options = affix.kind == Kind::List && as_list(affix).flavor == Flavor::Tuple
                                       ? as_list(affix).items
                                       : std::vector<Value>{affix};
      for (const Value& option : options) {
        const std::string& sub = str_arg(option, fn);
        const int64_t n = static_cast<int64_t>(cp_count(sub));
        if (end - start < n) continue;
        const std::string_view piece = method == "startswith" ? text.sub(start, start + n) : text.sub(end - n, end);
        if (piece == sub) return bool_value(true);
      }
      return bool_value(false);
    }
    if (method == "strip" || method == "lstrip" || method == "rstrip") {
      Bound b = bind(a, {"chars"}, 1, fn);
      const CharSet set = b[0] && b[0]->kind != Kind::None ? CharSet(str_arg(*b[0], fn)) : CharSet();
      return str_value(std::string(py_strip(s, set, method != "rstrip", method != "lstrip")), markup);
    }
    if (method == "split" || method == "rsplit") {
      Bound b = bind(a, {"sep", "maxsplit"}, 0, fn);
      const bool whitespace = !b[0] || b[0]->kind == Kind::None;
      int64_t limit = b[1] ? int_arg(*b[1], fn) : -1;
      if (limit < 0) limit = INT64_MAX;
      const bool from_right = method == "rsplit";
      std::vector<std::string> parts = whitespace ? split_whitespace(s, limit, from_right)
                                                  : split_on(s, str_arg(*b[0], fn), limit, from_right);
      std::vector<Value> out;
      for (std::string& part : parts) out.push_back(str_value(std::move(part), markup));
      return list_value(std::move(out));
    }
    if (method == "replace") {
      Bound b = bind(a, {"old", "new", "count"}, markup ? 3 : 2, fn);
      const std::string& old = str_arg(required(b, 0, fn), fn);
      const Value& replacement = required(b, 1, fn);
      std::string with = str_arg(replacement, fn);
      if (markup && !replacement.aux) with = html_escape(with);
      const int64_t count = b[2] ? int_arg(*b[2], fn) : -1;
      return str_value(replace(s, old, with, count), markup);
    }
    if (method == "upper" || method == "lower" || method == "title" || method == "capitalize") {
      bind(a, {}, 0, fn);
      std::string out = method == "upper"   ? py_upper(s)
                        : method == "lower" ? py_lower(s)
                        : method == "title" ? py_title(s)
                                            : py_capitalize(s);
      return str_value(std::move(out), markup);
    }
    if (method == "count" || method == "find" || method == "rfind") {
      Bound b = bind(a, {"sub", "start", "end"}, 3, fn);
      const std::string& sub = str_arg(required(b, 0, fn), fn);
      const Text text(s);
      int64_t start, end;
      adjust_indices(b, 1, text.size(), &start, &end);
      const int64_t n = static_cast<int64_t>(cp_count(sub));
      if (end - start < n) return int_value(method == "count" ? 0 : -1);
      const size_t lo = text.byte(start), hi = text.byte(end);
      const std::string_view window(s.data() + lo, hi - lo);
      if (method == "count") {
        if (sub.empty()) return int_value(end - start + 1);
        int64_t found = 0;
        for (size_t at = window.find(sub); at != std::string_view::npos; at = window.find(sub, at + sub.size())) {
          ++found;
        }
        return int_value(found);
      }
      const size_t at = method == "find" ? window.find(sub) : window.rfind(sub);
      return int_value(at == std::string_view::npos ? -1 : text.index(lo + at));
    }
    if (method == "join") {
      Bound b = bind(a, {"iterable"}, 1, fn);
      std::string out;
      bool first = true;
      for (const Value& item : iterate(required(b, 0, fn))) {
        if (item.kind != Kind::String) {
          runtime_fail(std::string("sequence item: expected str instance, ") + type_name(item) + " found");
        }
        if (!first) out += s;
        first = false;
        out += markup && !item.aux ? html_escape(*item.text) : *item.text;
      }
      return str_value(std::move(out), markup);
    }
    runtime_fail("unsupported string method '" + method + "'");
  }

  // str.split()/rsplit() without a separator.
  static std::vector<std::string> split_whitespace(const std::string& s, int64_t limit, bool from_right) {
    const Text text(s);
    const int64_t n = text.size();
    std::vector<std::string> out;
    auto space = [&](int64_t i) { return py_space(text.at(i)); };
    if (!from_right) {
      int64_t i = 0;
      while (limit-- > 0) {
        while (i < n && space(i)) ++i;
        if (i == n) break;
        const int64_t j = i;
        while (i < n && !space(i)) ++i;
        out.emplace_back(text.sub(j, i));
      }
      if (i < n) {
        while (i < n && space(i)) ++i;
        if (i != n) out.emplace_back(text.sub(i, n));
      }
      return out;
    }
    int64_t i = n - 1;
    while (limit-- > 0) {
      while (i >= 0 && space(i)) --i;
      if (i < 0) break;
      const int64_t j = i;
      while (i >= 0 && !space(i)) --i;
      out.emplace_back(text.sub(i + 1, j + 1));
    }
    if (i >= 0) {
      while (i >= 0 && space(i)) --i;
      if (i >= 0) out.emplace_back(text.sub(0, i + 1));
    }
    std::reverse(out.begin(), out.end());
    return out;
  }

  static std::vector<std::string> split_on(const std::string& s, const std::string& sep, int64_t limit,
                                           bool from_right) {
    if (sep.empty()) runtime_fail("empty separator");
    std::vector<std::string> out;
    if (!from_right) {
      size_t start = 0;
      for (size_t at = s.find(sep); at != std::string::npos && limit > 0; at = s.find(sep, start), --limit) {
        out.push_back(s.substr(start, at - start));
        start = at + sep.size();
      }
      out.push_back(s.substr(start));
      return out;
    }
    size_t end = s.size();
    while (limit > 0 && end >= sep.size()) {
      const size_t at = s.rfind(sep, end - sep.size());
      if (at == std::string::npos) break;
      out.push_back(s.substr(at + sep.size(), end - at - sep.size()));
      end = at;
      --limit;
    }
    out.push_back(s.substr(0, end));
    std::reverse(out.begin(), out.end());
    return out;
  }

  static std::string replace(const std::string& s, const std::string& old, const std::string& with, int64_t count) {
    if (count < 0) count = INT64_MAX;
    std::string out;
    if (old.empty()) {  // insert before every code point and at the end
      size_t i = 0, len = 0;
      for (; i < s.size() && count > 0; i += len, --count) {
        cp_at(s, i, &len);
        out += with;
        out.append(s, i, len);
      }
      if (i < s.size()) {
        out.append(s, i, std::string::npos);
      } else if (count > 0) {
        out += with;
      }
      return out;
    }
    size_t start = 0;
    for (size_t at = s.find(old); at != std::string::npos && count > 0; at = s.find(old, start), --count) {
      out.append(s, start, at - start);
      out += with;
      start = at + old.size();
    }
    out.append(s, start, std::string::npos);
    return out;
  }

  // Filters -----------------------------------------------------------------

  // make_attrgetter: dotted parts, digit parts as integers, looked up with
  // getitem; `fallback` replaces undefined results.
  Value attribute_of(Value item, const Value& attribute, const std::optional<Value>& fallback) {
    std::vector<Value> parts;
    if (attribute.kind == Kind::String) {
      const std::string& path = *attribute.text;
      size_t start = 0;
      while (true) {
        const size_t dot = path.find('.', start);
        const std::string part = path.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
        const bool digits = !part.empty() && std::all_of(part.begin(), part.end(), is_digit);
        if (!digits && !is_ascii(part)) runtime_fail("non-ASCII attribute paths are not supported");
        if (digits) {
          int64_t index = 0;
          if (!py_parse_int(part, 10, &index)) runtime_fail("invalid attribute path");
          parts.push_back(int_value(index));
        } else {
          parts.push_back(str_value(part));
        }
        if (dot == std::string::npos) break;
        start = dot + 1;
      }
    } else if (attribute.kind != Kind::None) {
      parts.push_back(attribute);
    }
    for (const Value& part : parts) {
      item = getitem(item, part);
      if (fallback && item.kind == Kind::Undefined) item = *fallback;
    }
    return item;
  }

  Value filter(FilterId id, Value v, Args a) {
    switch (id) {
      case FilterId::Abs: {
        bind(a, {}, 0, "abs");
        if (v.kind == Kind::Float) return float_value(std::fabs(v.number));
        if (!is_int_like(v)) runtime_fail(std::string("bad operand type for abs(): '") + type_name(v) + "'");
        if (as_int(v) == INT64_MIN) runtime_fail("integers beyond 64 bits are not supported");
        return int_value(std::abs(as_int(v)));
      }
      case FilterId::Capitalize:
      case FilterId::Lower:
      case FilterId::Title:
      case FilterId::Upper: {
        bind(a, {}, 0, "case");
        const Value s = soft_str(v);
        std::string out = id == FilterId::Capitalize ? py_capitalize(*s.text)
                          : id == FilterId::Lower    ? py_lower(*s.text)
                          : id == FilterId::Title    ? jinja_title(*s.text)
                                                     : py_upper(*s.text);
        return str_value(std::move(out), s.aux);
      }
      case FilterId::Default: {
        Bound b = bind(a, {"default_value", "boolean"}, 0, "default");
        const bool boolean = b[1] && truthy(*b[1]);
        if (v.kind == Kind::Undefined || (boolean && !truthy(v))) return b[0] ? *b[0] : str_value(std::string());
        return v;
      }
      case FilterId::Escape:
        bind(a, {}, 0, "escape");
        return is_markup(v) ? v : str_value(html_escape(to_str(v)), true);
      case FilterId::Safe:
        bind(a, {}, 0, "safe");
        return is_markup(v) ? v : str_value(to_str(v), true);
      case FilterId::First: {
        bind(a, {}, 0, "first");
        if (v.kind == Kind::List && as_list(v).flavor == Flavor::Iterator) {
          ListObject& list = as_list(v);
          if (list.cursor < list.items.size()) return list.items[list.cursor++];
        } else {
          const std::vector<Value> items = iterate(v);
          if (!items.empty()) return items.front();
        }
        return undefined_hint("No first item, sequence was empty.");
      }
      case FilterId::Last: {
        bind(a, {}, 0, "last");
        if (v.kind == Kind::List && as_list(v).flavor == Flavor::Iterator) {
          runtime_fail("'generator' object is not reversible");
        }
        if (v.kind != Kind::Undefined && v.kind != Kind::String && v.kind != Kind::List && v.kind != Kind::Dict) {
          runtime_fail(std::string("'") + type_name(v) + "' object is not reversible");
        }
        const std::vector<Value> items = iterate(v);
        if (!items.empty()) return items.back();
        return undefined_hint("No last item, sequence was empty.");
      }
      case FilterId::Float: {
        Bound b = bind(a, {"default"}, 0, "float");
        const Value fallback = b[0] ? *b[0] : float_value(0.0);
        if (v.kind == Kind::Undefined) fail_undefined(v);
        if (v.kind == Kind::String) {
          double d = 0;
          return py_parse_float(*v.text, &d) ? float_value(d) : fallback;
        }
        if (is_number(v)) return float_value(as_float(v));
        return fallback;
      }
      case FilterId::Int: {
        Bound b = bind(a, {"default", "base"}, 0, "int");
        const Value fallback = b[0] ? *b[0] : int_value(0);
        if (v.kind == Kind::Undefined) fail_undefined(v);
        auto from_float = [&](double d) {
          if (std::isnan(d) || std::isinf(d)) return fallback;
          if (d >= 9223372036854775808.0 || d < -9223372036854775808.0) {
            runtime_fail("integers beyond 64 bits are not supported");
          }
          return int_value(static_cast<int64_t>(d));
        };
        if (v.kind == Kind::String) {
          int64_t i = 0;
          if (py_parse_int(*v.text, b[1] ? int_arg(*b[1], "int") : 10, &i)) return int_value(i);
          double d = 0;
          return py_parse_float(*v.text, &d) ? from_float(d) : fallback;
        }
        if (is_int_like(v)) return int_value(as_int(v));
        if (v.kind == Kind::Float) {
          if (std::isinf(v.number)) runtime_fail("cannot convert float infinity to integer");
          return from_float(v.number);
        }
        return fallback;
      }
      case FilterId::Indent: {
        Bound b = bind(a, {"width", "first", "blank"}, 0, "indent");
        if (v.kind == Kind::Undefined) fail_undefined(v);
        if (v.kind != Kind::String) runtime_fail(std::string("indent expects a string, not '") + type_name(v) + "'");
        std::string unit(4, ' ');
        if (b[0]) unit = b[0]->kind == Kind::String ? *b[0]->text
                                                    : std::string(static_cast<size_t>(std::max<int64_t>(
                                                                      int_arg(*b[0], "indent"), 0)),
                                                                  ' ');
        return str_value(indent(*v.text, unit, b[1] && truthy(*b[1]), b[2] && truthy(*b[2])), v.aux);
      }
      case FilterId::Items: {
        bind(a, {}, 0, "items");
        std::vector<Value> out;
        if (v.kind == Kind::Dict) {
          for (const auto& [key, value] : as_dict(v).items) out.push_back(list_value({key, value}, Flavor::Tuple));
        } else if (v.kind != Kind::Undefined) {
          runtime_fail("Can only get item pairs from a mapping.");
        }
        return list_value(std::move(out), Flavor::Iterator);
      }
      case FilterId::Join: {
        Bound b = bind(a, {"d", "attribute"}, 0, "join");
        const std::string separator = b[0] ? to_str(*b[0]) : std::string();
        std::string out;
        bool first = true;
        for (const Value& item : iterate(v)) {
          if (!first) out += separator;
          first = false;
          append_str(out, b[1] ? attribute_of(item, *b[1], std::nullopt) : item);
        }
        return str_value(std::move(out));
      }
      case FilterId::Length:
        bind(a, {}, 0, "length");
        return int_value(py_len(v));
      case FilterId::List:
        bind(a, {}, 0, "list");
        return list_value(iterate(v));
      case FilterId::Map: return filter_map(v, std::move(a));
      case FilterId::Select: return filter_select(v, std::move(a), false, true);
      case FilterId::Reject: return filter_select(v, std::move(a), false, false);
      case FilterId::SelectAttr: return filter_select(v, std::move(a), true, true);
      case FilterId::RejectAttr: return filter_select(v, std::move(a), true, false);
      case FilterId::Replace: {
        Bound b = bind(a, {"old", "new", "count"}, 0, "replace");
        const int64_t count = b[2] && b[2]->kind != Kind::None ? int_arg(*b[2], "replace") : -1;
        return str_value(
            replace(to_str(v), to_str(required(b, 0, "replace")), to_str(required(b, 1, "replace")), count));
      }
      case FilterId::Reverse: {
        bind(a, {}, 0, "reverse");
        if (v.kind == Kind::String) {
          std::vector<Value> chars = code_points(v);
          std::string out;
          for (auto it = chars.rbegin(); it != chars.rend(); ++it) out += *it->text;
          return str_value(std::move(out), v.aux);
        }
        if (v.kind == Kind::List && as_list(v).flavor == Flavor::Iterator) {
          std::vector<Value> items = iterate(v);
          std::reverse(items.begin(), items.end());
          return list_value(std::move(items));
        }
        if (v.kind != Kind::Undefined && v.kind != Kind::List && v.kind != Kind::Dict) {
          runtime_fail("argument must be iterable");
        }
        std::vector<Value> items = iterate(v);
        std::reverse(items.begin(), items.end());
        return list_value(std::move(items), Flavor::Iterator);
      }
      case FilterId::Sort: {
        Bound b = bind(a, {"reverse", "case_sensitive", "attribute"}, 0, "sort");
        const bool reverse = b[0] && truthy(*b[0]);
        const bool case_sensitive = b[1] && truthy(*b[1]);
        // make_multi_attrgetter: comma-separated attributes give a key list.
        std::vector<Value> attributes;
        if (b[2] && b[2]->kind == Kind::String) {
          const std::string& spec = *b[2]->text;
          size_t start = 0;
          while (true) {
            const size_t comma = spec.find(',', start);
            const size_t length = comma == std::string::npos ? std::string::npos : comma - start;
            attributes.push_back(str_value(spec.substr(start, length)));
            if (comma == std::string::npos) break;
            start = comma + 1;
          }
        } else {
          attributes.push_back(b[2] ? *b[2] : none_value());
        }
        std::vector<Value> items = iterate(v);
        std::vector<std::pair<Value, size_t>> keyed;
        for (size_t i = 0; i < items.size(); ++i) {
          std::vector<Value> key;
          for (const Value& attribute : attributes) {
            Value k = attribute_of(items[i], attribute, std::nullopt);
            key.push_back(case_sensitive ? k : ignore_case(k));
          }
          keyed.emplace_back(list_value(std::move(key)), i);
        }
        std::stable_sort(keyed.begin(), keyed.end(), [&](const auto& x, const auto& y) {
          return reverse ? py_order(CmpOp::Lt, y.first, x.first) : py_order(CmpOp::Lt, x.first, y.first);
        });
        std::vector<Value> out;
        for (const auto& k : keyed) out.push_back(items[k.second]);
        return list_value(std::move(out));
      }
      case FilterId::String:
        bind(a, {}, 0, "string");
        return soft_str(v);
      case FilterId::ToJson: {
        Bound b = bind(a, {"ensure_ascii", "indent", "separators", "sort_keys"}, 0, "tojson");
        JsonStyle style;
        style.ascii = b[0] && truthy(*b[0]);
        if (b[1] && b[1]->kind != Kind::None) {
          style.indent = true;
          style.unit = b[1]->kind == Kind::String
                           ? *b[1]->text
                           : std::string(static_cast<size_t>(std::max<int64_t>(int_arg(*b[1], "tojson"), 0)), ' ');
          style.item_separator = ",";
        }
        if (b[2] && b[2]->kind != Kind::None) {
          const std::vector<Value> separators = iterate(*b[2]);
          if (separators.size() != 2) runtime_fail("tojson separators must be a pair of strings");
          style.item_separator = str_arg(separators[0], "tojson");
          style.key_separator = str_arg(separators[1], "tojson");
        }
        style.sort_keys = b[3] && truthy(*b[3]);
        std::string out;
        dump_json(out, v, style, 0);
        return str_value(std::move(out));
      }
      case FilterId::Trim: {
        Bound b = bind(a, {"chars"}, 0, "trim");
        const Value s = soft_str(v);
        const CharSet set = b[0] && b[0]->kind != Kind::None ? CharSet(str_arg(*b[0], "trim")) : CharSet();
        return str_value(std::string(py_strip(*s.text, set, true, true)), s.aux);
      }
      case FilterId::Unique: {
        Bound b = bind(a, {"case_sensitive", "attribute"}, 0, "unique");
        const bool case_sensitive = b[0] && truthy(*b[0]);
        const Value attribute = b[1] ? *b[1] : none_value();
        std::vector<Value> seen, out;
        for (const Value& item : iterate(v)) {
          Value key = attribute_of(item, attribute, std::nullopt);
          if (!case_sensitive) key = ignore_case(key);
          if (!hashable(key)) runtime_fail(std::string("unhashable type: '") + type_name(key) + "'");
          if (std::none_of(seen.begin(), seen.end(), [&](const Value& s) { return py_equal(s, key); })) {
            seen.push_back(std::move(key));
            out.push_back(item);
          }
        }
        return list_value(std::move(out), Flavor::Iterator);
      }
    }
    runtime_fail("unknown filter");
  }

  static std::string indent(const std::string& text, const std::string& unit, bool first, bool blank) {
    // do_indent appends a newline before splitlines().
    const std::string s = text + "\n";
    std::vector<std::string_view> lines;
    size_t start = 0;
    for (size_t i = 0, len = 0; i < s.size(); i += len) {
      const char32_t c = cp_at(s, i, &len);
      if (!py_line_break(c)) continue;
      lines.push_back(std::string_view(s).substr(start, i - start));
      if (c == '\r' && i + 1 < s.size() && s[i + 1] == '\n') ++len;
      start = i + len;
    }
    if (start < s.size()) lines.push_back(std::string_view(s).substr(start));
    std::string out;
    for (size_t i = 0; i < lines.size(); ++i) {
      if (i) out += '\n';
      if (i && (blank || !lines[i].empty())) out += unit;
      out += lines[i];
    }
    return first ? unit + out : out;
  }

  Value filter_map(const Value& v, Args a) {
    std::vector<Value> out;
    if (!truthy(v)) return list_value(std::move(out), Flavor::Iterator);  // `if value:`
    auto keyword = [&](std::string_view key) -> std::optional<Value> {
      for (auto it = a.kw.begin(); it != a.kw.end(); ++it) {
        if (name(it->first) == key) {
          Value value = it->second;
          a.kw.erase(it);
          return value;
        }
      }
      return std::nullopt;
    };
    if (a.pos.empty()) {
      const std::optional<Value> attribute = keyword("attribute");
      if (!attribute) runtime_fail("map requires a filter argument");
      std::optional<Value> fallback = keyword("default");
      if (fallback && fallback->kind == Kind::None) fallback.reset();
      if (!a.kw.empty()) runtime_fail("Unexpected keyword argument '" + name(a.kw.front().first) + "'");
      for (const Value& item : iterate(v)) out.push_back(attribute_of(item, *attribute, fallback));
      return list_value(std::move(out), Flavor::Iterator);
    }
    const std::string& filter_name = str_arg(a.pos[0], "map");
    const std::optional<FilterId> id = filter_named(filter_name);
    if (!id) runtime_fail("No filter named '" + filter_name + "'.");
    Args rest;
    rest.pos.assign(a.pos.begin() + 1, a.pos.end());
    rest.kw = a.kw;
    for (const Value& item : iterate(v)) out.push_back(filter(*id, item, rest));
    return list_value(std::move(out), Flavor::Iterator);
  }

  Value filter_select(const Value& v, Args a, bool by_attribute, bool keep) {
    std::vector<Value> out;
    if (!truthy(v)) return list_value(std::move(out), Flavor::Iterator);
    size_t offset = 0;
    if (by_attribute) {
      if (a.pos.empty()) runtime_fail("Missing parameter for attribute name");
      offset = 1;
    }
    std::optional<TestId> id;
    Args rest;
    if (a.pos.size() > offset) {
      const std::string& test_name = str_arg(a.pos[offset], "select");
      id = test_named(test_name);
      if (!id) runtime_fail("No test named '" + test_name + "'.");
      rest.pos.assign(a.pos.begin() + static_cast<std::ptrdiff_t>(offset + 1), a.pos.end());
      rest.kw = a.kw;
    }
    for (const Value& item : iterate(v)) {
      const Value subject = by_attribute ? attribute_of(item, a.pos[0], std::nullopt) : item;
      const bool passed = id ? test(*id, subject, rest) : truthy(subject);
      if (passed == keep) out.push_back(item);
    }
    return list_value(std::move(out), Flavor::Iterator);
  }

  // Tests -------------------------------------------------------------------

  bool test(TestId id, const Value& v, Args a) {
    auto other = [&](const char* param, size_t positional_only) -> Value {
      Bound b = bind(a, {param}, positional_only, "test");
      return required(b, 0, "test");
    };
    auto none = [&] { bind(a, {}, 0, "test"); };
    switch (id) {
      case TestId::Defined: none(); return v.kind != Kind::Undefined;
      case TestId::Undefined: none(); return v.kind == Kind::Undefined;
      case TestId::None: none(); return v.kind == Kind::None;
      case TestId::String: none(); return v.kind == Kind::String;
      case TestId::Number: none(); return is_number(v);
      case TestId::Integer: none(); return v.kind == Kind::Int;
      case TestId::Float: none(); return v.kind == Kind::Float;
      case TestId::Mapping: none(); return v.kind == Kind::Dict;
      case TestId::Iterable:
        none();
        return v.kind == Kind::Undefined || v.kind == Kind::String || v.kind == Kind::List || v.kind == Kind::Dict ||
               v.kind == Kind::Loop;
      case TestId::Sequence:  // has len() and __getitem__
        none();
        return v.kind == Kind::Undefined || v.kind == Kind::String || v.kind == Kind::Dict ||
               (v.kind == Kind::List && (is_list_or_tuple(v) || as_list(v).flavor == Flavor::Range));
      case TestId::Boolean: none(); return v.kind == Kind::Bool;
      case TestId::True: none(); return v.kind == Kind::Bool && v.boolean;
      case TestId::False: none(); return v.kind == Kind::Bool && !v.boolean;
      case TestId::Eq: return compare(CmpOp::Eq, v, other("b", 1));
      case TestId::Ne: return compare(CmpOp::Ne, v, other("b", 1));
      case TestId::Lt: return compare(CmpOp::Lt, v, other("b", 1));
      case TestId::Le: return compare(CmpOp::Le, v, other("b", 1));
      case TestId::Gt: return compare(CmpOp::Gt, v, other("b", 1));
      case TestId::Ge: return compare(CmpOp::Ge, v, other("b", 1));
      case TestId::In: return contains(other("seq", 0), v);
      case TestId::Odd: none(); return py_equal(binary(BinOp::Mod, v, int_value(2)), int_value(1));
      case TestId::Even: none(); return py_equal(binary(BinOp::Mod, v, int_value(2)), int_value(0));
      case TestId::DivisibleBy: return py_equal(binary(BinOp::Mod, v, other("num", 0)), int_value(0));
    }
    runtime_fail("unknown test");
  }

  const Program& p_;
  int line_ = 0;
  int depth_ = 0;
};

}  // namespace

struct ChatTemplate::Impl {
  Program program;
};

ChatTemplate::ChatTemplate(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ChatTemplate::ChatTemplate(ChatTemplate&&) noexcept = default;
ChatTemplate& ChatTemplate::operator=(ChatTemplate&&) noexcept = default;
ChatTemplate::~ChatTemplate() = default;

ChatTemplate ChatTemplate::parse(std::string_view source) {
  if (!valid_utf8(source)) syntax_error(1, "template is not valid UTF-8");
  auto impl = std::make_unique<Impl>();
  Parser(Lexer(source).run(), impl->program).run();
  return ChatTemplate(std::move(impl));
}

std::string ChatTemplate::render(const std::vector<ChatMessage>& messages, const ChatTemplateContext& context) const {
  LK_CHECK(impl_ != nullptr, "chat template was moved from");
  return Renderer(impl_->program).render(messages, context);
}

ChatTemplate load_chat_template(const std::string& directory, ChatTemplateContext* context) {
  auto exists = [](const std::string& path) { return std::ifstream(path, std::ios::binary).good(); };
  const std::string config_path = directory + "/tokenizer_config.json";
  Json config;
  if (exists(config_path)) config = Json::parse(read_text_file(config_path));
  if (context) {
    for (auto [key, out] : {std::pair{"bos_token", &context->bos_token}, std::pair{"eos_token", &context->eos_token}}) {
      const Json* token = config.find(key);
      if (token && token->is_object()) token = token->find("content");
      LK_CHECK(!token || token->is_null() || token->is_string(),
               std::string("tokenizer_config.json: unsupported ") + key);
      *out = token && token->is_string() ? token->as_string() : std::string();
    }
  }
  const std::string jinja_path = directory + "/chat_template.jinja";
  if (exists(jinja_path)) return ChatTemplate::parse(read_text_file(jinja_path));
  const Json* chat = config.find("chat_template");
  if (chat && chat->is_string()) return ChatTemplate::parse(chat->as_string());
  if (chat && chat->is_array()) {
    for (const Json& item : chat->as_array()) {
      const Json* text = item.find("template");
      if (item.string_or("name", "") == "default" && text && text->is_string()) {
        return ChatTemplate::parse(text->as_string());
      }
    }
    fail("lokahi: " + directory + " has several chat templates and none is named \"default\"");
  }
  fail("lokahi: " + directory +
       " has no chat template (chat_template.jinja or a \"chat_template\" in tokenizer_config.json)");
}

}  // namespace lokahi
