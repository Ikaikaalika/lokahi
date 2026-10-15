#include "text_stream.h"

#include <algorithm>
#include <utility>

#include "common.h"

namespace lokahi {

namespace {

// Decoded context kept ahead of new tokens, so a decoder that strips leading
// characters of the window (Strip, Metaspace) only ever strips context.
constexpr size_t kMinContextBytes = 8;

}  // namespace

TextStream::TextStream(const Tokenizer& tokenizer, std::vector<std::string> stop_strings, bool skip_special_tokens)
    : tokenizer_(tokenizer), stop_(std::move(stop_strings)), skip_special_(skip_special_tokens) {
  for (const std::string& stop : stop_) LK_CHECK(!stop.empty(), "text stream: stop strings must not be empty");
}

std::string TextStream::push(int32_t token) {
  LK_CHECK(!finished_, "text stream: push after finish");
  if (stopped_) return {};
  ids_.push_back(token);
  const std::vector<int32_t> tail(ids_.begin() + static_cast<std::ptrdiff_t>(window_), ids_.end());
  const size_t stable = window_ + tokenizer_.stable_prefix(tail, skip_special_);
  if (stable <= stable_) return {};
  return release(advance(stable));
}

std::string TextStream::finish() {
  LK_CHECK(!finished_, "text stream: finished twice");
  finished_ = true;
  if (stopped_) return {};
  std::string out;
  if (ids_.size() > stable_) out = release(advance(ids_.size()));
  if (!stopped_) {
    out += held_;
    emitted_ += held_;
    held_.clear();
  }
  return out;
}

std::string TextStream::decoded_through(size_t end) const {
  return tokenizer_.decode(std::vector<int32_t>(ids_.begin() + static_cast<std::ptrdiff_t>(window_),
                                                ids_.begin() + static_cast<std::ptrdiff_t>(end)),
                           skip_special_);
}

// Text of ids_[stable_, stable) given the window's decoding, then slides the
// window to the latest boundary that keeps enough context.
std::string TextStream::advance(size_t stable) {
  const std::string text = decoded_through(stable);
  LK_CHECK(text.compare(0, window_text_.size(), window_text_) == 0, "text stream: decoded text was rewritten");
  std::string delta = text.substr(window_text_.size());
  if (stable_ > window_) boundaries_.push_back(stable_);
  stable_ = stable;
  window_text_ = text;
  while (!boundaries_.empty()) {
    const size_t old_window = window_;
    window_ = boundaries_.front();
    std::string context = decoded_through(stable_);
    if (context.size() < kMinContextBytes) {
      window_ = old_window;
      break;
    }
    boundaries_.pop_front();
    window_text_ = std::move(context);
  }
  return delta;
}

std::string TextStream::release(const std::string& text) {
  held_ += text;
  size_t cut = std::string::npos;
  for (const std::string& stop : stop_) cut = std::min(cut, held_.find(stop));
  std::string out;
  if (cut != std::string::npos) {
    stopped_ = true;
    out = held_.substr(0, cut);
    held_.clear();
  } else {
    // Hold back the longest suffix that could still grow into a stop string.
    size_t keep = 0;
    for (const std::string& stop : stop_) {
      for (size_t length = std::min(stop.size() - 1, held_.size()); length > keep; --length) {
        if (held_.compare(held_.size() - length, length, stop, 0, length) == 0) {
          keep = length;
          break;
        }
      }
    }
    out = held_.substr(0, held_.size() - keep);
    held_.erase(0, held_.size() - keep);
  }
  emitted_ += out;
  return out;
}

}  // namespace lokahi
