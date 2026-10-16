// Speculative decoding: drafts that the model verifies in one batched pass.
//
// Greedy generation proposes up to max_draft tokens, runs the last token and
// the drafts through Model::verify, keeps the drafts the model's own greedy
// choices agree with, plus the model's next token, and rolls the rest back
// (Model::accept). The output is the same as decoding one token at a time.
#pragma once

#include <cstdint>
#include <vector>

namespace lokahi {

struct SpeculationOptions {
  int max_draft = 0;                // 0 disables speculation
  std::vector<int32_t> reference;   // a document drafts may also come from (e.g. text being edited)
};

struct SpeculationStats {
  int passes = 0;    // verify passes
  int drafted = 0;   // draft tokens proposed
  int accepted = 0;  // draft tokens the model agreed with
};

// Prompt-lookup drafts: the tokens that followed an earlier occurrence of the
// context's last n tokens (n from 3 down to 1). The reference is searched
// first, onward from where its last proposed run ended, then the context
// itself, latest occurrence first.
class PromptLookupDrafter {
 public:
  explicit PromptLookupDrafter(std::vector<int32_t> reference = {});

  std::vector<int32_t> propose(const std::vector<int32_t>& context, int max_draft);

 private:
  std::vector<int32_t> reference_;
  size_t cursor_ = 0;  // where the reference's last proposed run ended
};

}  // namespace lokahi
