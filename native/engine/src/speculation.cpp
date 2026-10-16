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

PromptLookupDrafter::PromptLookupDrafter(std::vector<int32_t> reference) : reference_(std::move(reference)) {}

std::vector<int32_t> PromptLookupDrafter::propose(const std::vector<int32_t>& context, int max_draft) {
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
        return draft;
      }
    }
    // The context before its own suffix, latest occurrence first.
    for (size_t at = context.size() - n; at-- > 0;) {
      if (std::equal(suffix, context.end(), context.begin() + static_cast<std::ptrdiff_t>(at))) {
        return take(context, at + n);
      }
    }
  }
  return {};
}

}  // namespace lokahi
