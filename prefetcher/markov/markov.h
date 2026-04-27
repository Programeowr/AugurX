#ifndef PREFETCHER_MARKOV_H
#define PREFETCHER_MARKOV_H

#include <array>
#include <cstdint>

#include "address.h"
#include "champsim.h"
#include "modules.h"
#include "msl/lru_table.h"

struct markov : public champsim::modules::prefetcher {
  struct successor {
    champsim::block_number address{};
    bool valid = false;
  };

  struct entry {
    champsim::block_number key{};
    std::array<successor, 4> next{};

    auto index() const
    {
      using namespace champsim::data::data_literals;
      return key.slice_upper<2_b>();
    }

    auto tag() const
    {
      using namespace champsim::data::data_literals;
      return key.slice_upper<2_b>();
    }
  };

  static constexpr std::size_t TABLE_SETS = 4096;
  static constexpr std::size_t TABLE_WAYS = 4;
  static constexpr std::size_t MAX_PREDICTIONS = 4;

  champsim::msl::lru_table<entry> table{TABLE_SETS, TABLE_WAYS};

  champsim::block_number last_miss{};
  bool have_last_miss = false;

  uint64_t stat_lookups = 0;
  uint64_t stat_trains = 0;
  uint64_t stat_prefetches_issued = 0;

  static entry train_successor(entry state, champsim::block_number successor_addr);

public:
  using champsim::modules::prefetcher::prefetcher;

  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, bool cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, bool prefetch, champsim::address evicted_addr, uint32_t metadata_in);
  void prefetcher_final_stats();
};

#endif
