#ifndef hi_H
#define hi_H

#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>
#include <vector>

#include "address.h"
#include "champsim.h"
#include "modules.h"
#include "msl/lru_table.h"

struct hi : public champsim::modules::prefetcher {

  // =========================================================================
  // Constants (paper Table 1)
  // =========================================================================
  constexpr static std::size_t LHT_CAPACITY   = 32;   // Load History Table entries
  constexpr static std::size_t LIT_CAPACITY   = 16;   // Load Identity Table entries
  constexpr static std::size_t AHQ_SIZE       = 8;    // Address History Queue (prefetch distance)
  constexpr static int         MAX_CONF        = 8;    // Maximum recursive confidence
  constexpr static int         RECUR_THRESHOLD = 2;   // Min confidence to be "recursive"
  constexpr static int         MAX_LDS_ID      = 16;  // Max distinct LDS IDs tracked

  // =========================================================================
  // Load History Table entry
  // Records dynamic per-instance information to detect instruction dependencies
  // =========================================================================
  struct lht_entry {
    uint32_t hashed_pc = 0;
    uint64_t loaded_value = 0;   // value loaded from memory (potential next-node pointer)
    bool     recursive_producer = false;
  };

  // =========================================================================
  // Load Identity Table entry
  // Records static per-PC information: offset, load type, LDS membership
  // =========================================================================
  struct lit_entry {
    uint32_t hashed_pc   = 0;
    int64_t  offset      = 0;    // displacement (e.g. 8 in "load r12, r12, 8")
    bool     is_direct   = true; // direct vs indirect element
    bool     is_producer = false;
    int      recursive_confidence = MAX_CONF; // 3-bit saturating counter (0..MAX_CONF)
    uint8_t  lds_id      = 0;    // which LDS node this PC belongs to
    uint8_t  producer_index = 0; // index of producer in LIT
    bool     valid       = false;

    bool is_recursive() const { return recursive_confidence >= RECUR_THRESHOLD; }
  };

  // =========================================================================
  // Metadata entry stored in L2 (emulated as an on-chip hash map here)
  // Represents a learned (trigger_node_addr -> target_node_addr) correlation
  // =========================================================================
  struct metadata_entry {
    uint64_t target_node_addr = 0;
    bool     confident        = false; // 1-bit confidence
  };

  // =========================================================================
  // State
  // =========================================================================

  // FIFO queue acting as Load History Table
  std::deque<lht_entry> lht;

  // Fully-associative LIT (small, so linear scan is fine)
  std::vector<lit_entry> lit;
  uint8_t next_lds_id = 1;

  // Address History Queue: FIFO of recently seen node addresses
  // Eviction from AHQ triggers metadata recording
  std::deque<uint64_t> ahq;

  // Metadata table: trigger_node_addr -> metadata_entry
  // Emulates the L2-resident metadata described in the paper
  std::unordered_map<uint64_t, metadata_entry> metadata;

  // =========================================================================
  // Helpers
  // =========================================================================

  // Hash a full PC down to 24 bits (as in Figure 6)
  static uint32_t hash_pc(champsim::address ip) {
    uint64_t v = ip.to<uint64_t>();
    return static_cast<uint32_t>((v ^ (v >> 16) ^ (v >> 32)) & 0xFFFFFF);
  }

  // Find a LIT entry by hashed PC; returns nullptr if not found
  lit_entry* find_lit(uint32_t hpc);

  // Insert or update a LIT entry; evicts LRU if full (simple FIFO eviction here)
  lit_entry* insert_lit(uint32_t hpc, int64_t offset);

  // Update LIT confidence based on an evicted LHT entry
  void update_lit_from_lht(const lht_entry& evicted);

  // Process one load through the pruning mechanism (called at writeback)
  void recursive_load_identification(champsim::address ip, uint64_t loaded_value,
                                     champsim::address mem_addr);

  // Refine address to 32-byte-aligned node address by subtracting offset
  static uint64_t to_node_addr(uint64_t mem_addr, int64_t offset) {
    return (mem_addr - static_cast<uint64_t>(offset)) & ~uint64_t{0x1F}; // 32-byte align
  }

  // Insert a node address into the AHQ; if AHQ full, record metadata correlation
  void ahq_insert(uint64_t node_addr);

  // Issue prefetch requests for a target node address, using LIT offsets
  void issue_prefetches(uint64_t target_node_addr);

public:
  using champsim::modules::prefetcher::prefetcher;

  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip,
                                    uint8_t cache_hit, bool useful_prefetch,
                                    access_type type, uint32_t metadata_in);

  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way,
                                 uint8_t prefetch, champsim::address evicted_addr,
                                 uint32_t metadata_in);

  void prefetcher_cycle_operate();
};

#endif // hi_H