#ifndef AUGURX_H
#define AUGURX_H

#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>
#include <vector>

#include "address.h"
#include "champsim.h"
#include "modules.h"
#include "msl/lru_table.h"

// =============================================================================
// AugurX – extends Augur with a 2-successor metadata table.
//
// Core change vs Augur:
//   Original Augur: metadata[trigger] = { single_target, 1-bit confident }
//   AugurX:         metadata[trigger] = { target[0], conf[0],
//                                         target[1], conf[1] }
//
// Both slots are tracked independently.  On a lookup, AugurX issues prefetches
// for every slot whose confidence is above CONF_THRESHOLD, so in a binary-tree
// traversal both the left child and the right child get prefetched.
//
// Confidence model (3-bit saturating counter per slot, 0..MAX_SLOT_CONF):
//   - A new address seen for an empty slot → slot filled, conf = 1.
//   - Same address seen again            → conf++ (saturates at MAX_SLOT_CONF).
//   - Different address seen for slot 0  → try slot 1; if slot 1 also conflicts,
//     evict the lower-confidence slot and install the new address with conf = 1.
//   - Prefetch is issued only if conf >= CONF_THRESHOLD (default 1, i.e. always
//     prefetch once we have seen an address at least once).
//
// Setting CONF_THRESHOLD = 1 means "prefetch both if seen at least once", which
// is safe and avoids the problem of a true 2-child node never reaching threshold.
// =============================================================================

struct augurX : public champsim::modules::prefetcher {

  // ── tunables ──────────────────────────────────────────────────────────────
  constexpr static std::size_t LHT_CAPACITY    = 32;
  constexpr static std::size_t LIT_CAPACITY    = 16;
  constexpr static std::size_t AHQ_SIZE        = 8;
  constexpr static int         MAX_CONF        = 8;   // recursive-confidence max
  constexpr static int         RECUR_THRESHOLD = 2;   // min conf → recursive
  constexpr static int         MAX_LDS_ID      = 16;
  constexpr static int         NUM_SUCCESSORS  = 2;   // AugurX: 2 slots per entry
  constexpr static int         MAX_SLOT_CONF   = 7;   // 3-bit saturating counter
  constexpr static int         CONF_THRESHOLD  = 1;   // min slot conf to prefetch

  // ── Load History Table ────────────────────────────────────────────────────
  struct lht_entry {
    uint32_t hashed_pc         = 0;
    uint64_t loaded_value      = 0;
    bool     recursive_producer = false;
  };

  // ── Load Identity Table ───────────────────────────────────────────────────
  struct lit_entry {
    uint32_t hashed_pc            = 0;
    int64_t  offset               = 0;
    bool     is_direct            = true;
    bool     is_producer          = false;
    int      recursive_confidence = MAX_CONF;
    uint8_t  lds_id               = 0;
    uint8_t  producer_index       = 0;
    bool     valid                = false;

    bool is_recursive() const { return recursive_confidence >= RECUR_THRESHOLD; }
  };

  // ── 2-successor metadata entry ────────────────────────────────────────────
  // Each trigger node address maps to up to NUM_SUCCESSORS target addresses,
  // each with its own saturating confidence counter.
  struct successor_slot {
    uint64_t target_node_addr = 0;
    int      confidence       = 0;   // 0 means slot is empty
  };

  struct metadata_entry {
    successor_slot slots[NUM_SUCCESSORS] = {};

    // Record a new observation of `target` for this trigger.
    void observe(uint64_t target) {
      // 1. Does target already exist in a slot?
      for (auto& s : slots) {
        if (s.confidence > 0 && s.target_node_addr == target) {
          if (s.confidence < MAX_SLOT_CONF)
            ++s.confidence;
          return;
        }
      }
      // 2. Is there an empty slot?
      for (auto& s : slots) {
        if (s.confidence == 0) {
          s.target_node_addr = target;
          s.confidence       = 1;
          return;
        }
      }
      // 3. All slots occupied and target is new → evict lowest-confidence slot.
      int min_conf  = MAX_SLOT_CONF + 1;
      int min_index = 0;
      for (int i = 0; i < NUM_SUCCESSORS; ++i) {
        if (slots[i].confidence < min_conf) {
          min_conf  = slots[i].confidence;
          min_index = i;
        }
      }
      slots[min_index].target_node_addr = target;
      slots[min_index].confidence       = 1;
    }
  };

  // ── State ─────────────────────────────────────────────────────────────────
  std::deque<lht_entry>                           lht;
  std::vector<lit_entry>                          lit;
  uint8_t                                         next_lds_id = 1;
  std::deque<uint64_t>                            ahq;
  std::unordered_map<uint64_t, metadata_entry>    metadata;

  // ── Helpers ───────────────────────────────────────────────────────────────
  static uint32_t hash_pc(champsim::address ip) {
    uint64_t v = ip.to<uint64_t>();
    return static_cast<uint32_t>((v ^ (v >> 16) ^ (v >> 32)) & 0xFFFFFF);
  }

  static uint64_t to_node_addr(uint64_t mem_addr, int64_t offset) {
    // Subtract offset to recover node base; 32-byte align (paper §5.2)
    return (mem_addr - static_cast<uint64_t>(offset)) & ~uint64_t{0x1F};
  }

  lit_entry* find_lit(uint32_t hpc);
  lit_entry* insert_lit(uint32_t hpc, int64_t offset);
  void       update_lit_from_lht(const lht_entry& evicted);
  void       recursive_load_identification(champsim::address ip,
                                           uint64_t loaded_value,
                                           champsim::address mem_addr);
  void       ahq_insert(uint64_t node_addr);

  // Core change: issue prefetches for ALL confident successors
  void       issue_prefetches(const metadata_entry& entry);

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

#endif // AUGURX_H