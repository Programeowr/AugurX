#include "markov.h"

#include <algorithm>
#include <iostream>

markov::entry markov::train_successor(entry state, champsim::block_number successor_addr)
{
  auto hit = std::find_if(std::begin(state.next), std::end(state.next),
                          [successor_addr](const auto& x) { return x.valid && x.address == successor_addr; });

  if (hit != std::end(state.next)) {
    auto updated = *hit;
    std::move_backward(std::begin(state.next), hit, std::next(hit));
    state.next[0] = updated;
    return state;
  }

  std::move_backward(std::begin(state.next), std::prev(std::end(state.next)), std::end(state.next));
  state.next[0] = {successor_addr, true};
  return state;
}

uint32_t markov::prefetcher_cache_operate(champsim::address addr, champsim::address ip, bool cache_hit, bool useful_prefetch, access_type type,
                                          uint32_t metadata_in)
{
  champsim::block_number current{addr};

  // The original Markov prefetcher is trained on the miss-address stream.
  if (cache_hit)
    return metadata_in;

  stat_lookups++;

  if (auto found = table.check_hit({current}); found.has_value()) {
    // The paper prioritizes multiple eligible successors using recency.
    // We issue higher-priority predictions first and stop once the queue
    // back-pressures us, which approximates low-priority discard.
    for (const auto& prediction : found->next) {
      if (!prediction.valid || prediction.address == current)
        continue;

      const bool success = prefetch_line(champsim::address{prediction.address}, true, metadata_in);
      if (!success)
        break;

      stat_prefetches_issued++;
    }
  }

  if (have_last_miss) {
    entry updated{last_miss};
    if (auto prev = table.check_hit(updated); prev.has_value())
      updated = *prev;

    updated = train_successor(updated, current);
    table.fill(updated);
    stat_trains++;
  }

  last_miss = current;
  have_last_miss = true;

  return metadata_in;
}

uint32_t markov::prefetcher_cache_fill(champsim::address addr, long set, long way, bool prefetch, champsim::address evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void markov::prefetcher_final_stats()
{
  std::cout << "[Markov] lookups=" << stat_lookups << " trains=" << stat_trains
            << " prefetches_issued=" << stat_prefetches_issued << '\n';
}
