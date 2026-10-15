// Streaming text for generated tokens.
//
// Each push returns only text that later tokens cannot change, so a client
// never sees output rewritten. Stop strings are applied to that text: once
// one appears, it and everything after it are dropped and the stream stops.
// Text that might be the start of a stop string is held back until it is
// either completed or ruled out.
//
// Only a short window of recent tokens is re-decoded per push. The window
// starts at an earlier stable boundary, so decoders that act on the first
// token (Metaspace, Strip) affect only context that was already emitted. The
// cost per token stays constant instead of growing with the output.
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "tokenizer.h"

namespace lokahi {

class TextStream {
 public:
  TextStream(const Tokenizer& tokenizer, std::vector<std::string> stop_strings, bool skip_special_tokens = true);

  // Appends one generated token and returns the text that is now final.
  std::string push(int32_t token);
  // Ends the stream and returns the remaining text, decoded as a full decode
  // of every token would (an unfinished UTF-8 sequence becomes U+FFFD).
  std::string finish();

  // True once a stop string has appeared; later pushes return nothing.
  bool stopped() const { return stopped_; }
  // All text returned so far.
  const std::string& text() const { return emitted_; }

 private:
  std::string decoded_through(size_t end) const;
  std::string advance(size_t stable);
  std::string release(const std::string& text);

  const Tokenizer& tokenizer_;
  std::vector<std::string> stop_;
  bool skip_special_;
  std::vector<int32_t> ids_;
  size_t stable_ = 0;            // ids whose text is final
  size_t window_ = 0;            // decoding restarts here, at a stable boundary
  std::string window_text_;      // decode(ids_[window_, stable_))
  std::deque<size_t> boundaries_;  // earlier stable boundaries after window_
  std::string held_;             // final text held back as a possible stop-string start
  std::string emitted_;
  bool stopped_ = false;
  bool finished_ = false;
};

}  // namespace lokahi
