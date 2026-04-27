#ifndef PREFETCHER_AUGUR_H
#define PREFETCHER_AUGUR_H

#include <cstdint>
#include "champsim.h"
#include "modules.h"

class augur_prefetcher : public champsim::modules::prefetcher
{
public:
  using prefetcher::prefetcher;

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
};

#endif