#pragma once
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace lokahi {

struct ChatMessage {
  std::string role;
  std::string content;
};

// Variables a template can read besides `messages`.
struct ChatTemplateContext {
  bool add_generation_prompt = true;
  std::string bos_token;
  std::string eos_token;
  std::map<std::string, std::string> strings;  // extra string variables, e.g. date_string
  std::map<std::string, bool> flags;           // extra boolean variables, e.g. enable_thinking
};

class ChatTemplate {
 public:
  // Parses Jinja source. Throws lokahi::Error naming the line and the construct when the
  // template uses syntax outside the supported subset (fail closed, never mis-render).
  static ChatTemplate parse(std::string_view source);
  ChatTemplate(ChatTemplate&&) noexcept;
  ChatTemplate& operator=(ChatTemplate&&) noexcept;
  ~ChatTemplate();
  // Throws lokahi::Error on raise_exception(...) (message preserved) and on runtime errors.
  std::string render(const std::vector<ChatMessage>& messages, const ChatTemplateContext& context) const;
 private:
  struct Impl;
  explicit ChatTemplate(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Template for a Hugging Face snapshot directory: chat_template.jinja when present, otherwise
// tokenizer_config.json "chat_template" (a string, or a list of {"name","template"} objects where
// "default" is used). Fills context->bos_token / eos_token from tokenizer_config.json (each a
// string or an {"content": ...} object). Throws lokahi::Error when there is no chat template.
ChatTemplate load_chat_template(const std::string& directory, ChatTemplateContext* context);

}  // namespace lokahi
