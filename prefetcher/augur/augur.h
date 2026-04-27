#ifndef CHAMPSIM_PREFETCHER_AUGUR_H
#define CHAMPSIM_PREFETCHER_AUGUR_H

#include <cstdint>
#include <optional>
#include "modules.h"
#include "champsim.h"

class augur : public champsim::modules::prefetcher
{
public:
  struct lookahead_entry {
    champsim::address address{};
    champsim::address::difference_type stride{};
    int degree = 0;
  };

  using prefetcher::prefetcher;

  std::optional<lookahead_entry> active_lookahead;

  uint32_t prefetcher_cache_operate(
      champsim::address addr,
      champsim::address ip,
      uint8_t cache_hit,
      bool useful_prefetch,
      access_type type,
      uint32_t metadata_in);

  uint32_t prefetcher_cache_fill(
      champsim::address addr,
      long set,
      long way,
      uint8_t prefetch,
      champsim::address evicted_addr,
      uint32_t metadata_in);

  void prefetcher_cycle_operate();
};

#endif
