#include "speculation.h"

#include <algorithm>
#include <utility>

#include "common.h"

namespace lokahi {

Drafter parse_drafter(const std::string& name) {
  if (name == "lookup") return Drafter::Lookup;
  if (name == "mtp") return Drafter::Mtp;
  LK_CHECK(name == "both", "the drafter must be lookup, mtp or both, not '" + name + "'");
  return Drafter::Both;
}

DraftPolicy parse_draft_policy(const std::string& name) {
  if (name == "fixed") return DraftPolicy::Fixed;
  LK_CHECK(name == "adaptive", "the draft policy must be fixed or adaptive, not '" + name + "'");
  return DraftPolicy::Adaptive;
}

PromptLookupDrafter::PromptLookupDrafter(std::vector<int32_t> reference) : reference_(std::move(reference)) {}

void PromptLookupDrafter::shorten(int count) {
  if (from_reference_ && count > 0) cursor_ -= std::min(cursor_, static_cast<size_t>(count));
}

std::vector<int32_t> PromptLookupDrafter::propose(const std::vector<int32_t>& context, int max_draft, int* matched) {
  if (matched) *matched = 0;
  from_reference_ = false;
  if (max_draft <= 0) return {};
  const auto take = [&](const std::vector<int32_t>& source, size_t from) {
    const size_t end = std::min(source.size(), from + static_cast<size_t>(max_draft));
    return std::vector<int32_t>(source.begin() + static_cast<std::ptrdiff_t>(from),
                                source.begin() + static_cast<std::ptrdiff_t>(end));
  };
  for (size_t n = std::min<size_t>(3, context.size()); n >= 1; --n) {
    const auto suffix = context.end() - static_cast<std::ptrdiff_t>(n);
    // The reference, onward from where the last run ended (whose last tokens
    // are this suffix when the model took all of it), then from its start.
    for (const size_t from : {cursor_ > n ? cursor_ - n : size_t{0}, size_t{0}}) {
      if (from >= reference_.size()) continue;
      const auto hit = std::search(reference_.begin() + static_cast<std::ptrdiff_t>(from), reference_.end(),
                                   suffix, context.end());
      const size_t next = static_cast<size_t>(hit - reference_.begin()) + n;
      if (hit != reference_.end() && next < reference_.size()) {
        std::vector<int32_t> draft = take(reference_, next);
        cursor_ = next + draft.size();
        from_reference_ = true;
        if (matched) *matched = static_cast<int>(n);
        return draft;
      }
    }
    // The context before its own suffix, latest occurrence first.
    for (size_t at = context.size() - n; at-- > 0;) {
      if (std::equal(suffix, context.end(), context.begin() + static_cast<std::ptrdiff_t>(at))) {
        if (matched) *matched = static_cast<int>(n);
        return take(context, at + n);
      }
    }
  }
  return {};
}

// Priors: longer lookup matches and the model's own head start more trusted.
DraftBudget::DraftBudget(double extra_token_cost) : cost_(extra_token_cost), p_{0.3, 0.45, 0.6, 0.6} {}

DraftBudget::Source DraftBudget::lookup_source(int matched) {
  return matched >= 3 ? kLookup3 : matched == 2 ? kLookup2 : kLookup1;
}

int DraftBudget::length(Source source, int available) {
  const double p = std::clamp(p_[source], 0.01, 0.99);
  int best = 0;
  double best_rate = 1.0, expected = 1.0, term = 1.0;
  for (int k = 1; k <= available; ++k) {
    term *= p;
    expected += term;
    const double rate = expected / (1.0 + cost_ * k);
    if (rate > best_rate) {
      best_rate = rate;
      best = k;
    }
  }
  if (best > 0) {
    idle_ = 0;
    return best;
  }
  if (available > 0 && ++idle_ >= kProbeEvery) {
    idle_ = 0;
    return 1;
  }
  return 0;
}

void DraftBudget::update(Source source, int drafted, int accepted) {
  constexpr double kRate = 0.2;
  double& p = p_[source];
  // Positions 1..accepted were accepted; the next one, if drafted, was not.
  for (int i = 0; i < accepted; ++i) p += kRate * (1.0 - p);
  if (accepted < drafted) p -= kRate * p;
}

}  // namespace lokahi
