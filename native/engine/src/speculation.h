// Speculative decoding: drafts that the model verifies in one batched pass.
//
// Greedy generation proposes up to max_draft tokens, runs the last token and
// the drafts through Model::verify, keeps the drafts the model's own greedy
// choices agree with, plus the model's next token, and rolls the rest back
// (Model::accept). The output is the same as decoding one token at a time.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace lokahi {

// Where drafts come from: prompt lookup (below); the model's multi-token
// prediction head (Model::mtp_draft); or lookup when it finds a draft and the
// head otherwise.
enum class Drafter { Lookup, Mtp, Both };

// "lookup", "mtp" or "both".
Drafter parse_drafter(const std::string& name);

// How long drafts are: up to max_draft whenever a drafter has one (fixed),
// or as long as the drafts' measured acceptance makes worthwhile, down to
// none (adaptive; see DraftBudget).
enum class DraftPolicy { Fixed, Adaptive };

// "fixed" or "adaptive".
DraftPolicy parse_draft_policy(const std::string& name);

struct SpeculationOptions {
  int max_draft = 0;                // 0 disables speculation
  std::vector<int32_t> reference;   // a document drafts may also come from (e.g. text being edited)
  Drafter drafter = Drafter::Lookup;
  DraftPolicy policy = DraftPolicy::Fixed;
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

  // A draft of up to max_draft tokens; *matched, when given, is the n whose
  // suffix it matched (0 without a draft).
  std::vector<int32_t> propose(const std::vector<int32_t>& context, int max_draft, int* matched = nullptr);
  // Takes back the last `count` tokens of the last proposal (a caller that
  // drafted fewer), so the reference's run resumes where the draft ended.
  void shorten(int count);

 private:
  std::vector<int32_t> reference_;
  size_t cursor_ = 0;              // where the reference's last proposed run ended
  bool from_reference_ = false;    // whether the last proposal came from the reference
};

// Adaptive draft lengths. Each draft source keeps an estimate p of how often
// its next draft token is accepted (an exponential average over the drafted
// positions passes have observed). A pass that verifies k drafts costs about
// 1 + c * k decode steps, where c is the backend's cost of one more verified
// token, and yields 1 + p + ... + p^k tokens on average; length() picks the k
// with the most tokens per step, 0 (decode plainly) when no draft pays.
// After every `kProbeEvery` plain steps a source may still draft one token,
// so an estimate that fell during one stretch of text can recover in the
// next.
class DraftBudget {
 public:
  enum Source { kLookup1, kLookup2, kLookup3, kMtp, kSources };
  static constexpr int kProbeEvery = 8;

  explicit DraftBudget(double extra_token_cost);

  // A lookup draft's source from the suffix length it matched.
  static Source lookup_source(int matched);
  int length(Source source, int available);
  void update(Source source, int drafted, int accepted);
  double acceptance(Source source) const { return p_[source]; }

 private:
  double cost_;
  std::array<double, kSources> p_;
  int idle_ = 0;  // plain steps since the last draft
};

}  // namespace lokahi
