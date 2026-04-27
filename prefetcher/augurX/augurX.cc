#include "augurX.h"

#include <algorithm>
#include <cassert>

#include "cache.h"

// =============================================================================
// LIT helpers  (unchanged from Augur)
// =============================================================================

augurX::lit_entry* augurX::find_lit(uint32_t hpc)
{
  for (auto& e : lit)
    if (e.valid && e.hashed_pc == hpc)
      return &e;
  return nullptr;
}

augurX::lit_entry* augurX::insert_lit(uint32_t hpc, int64_t offset)
{
  if (auto* e = find_lit(hpc); e != nullptr)
    return e;

  // Empty slot first
  for (auto& e : lit) {
    if (!e.valid) {
      e             = {};
      e.hashed_pc   = hpc;
      e.offset      = offset;
      e.recursive_confidence = MAX_CONF;
      e.valid       = true;
      return &e;
    }
  }

  // Full → FIFO evict
  if (lit.size() >= LIT_CAPACITY)
    lit.erase(lit.begin());

  lit.push_back({});
  auto& e           = lit.back();
  e.hashed_pc       = hpc;
  e.offset          = offset;
  e.recursive_confidence = MAX_CONF;
  e.valid           = true;
  return &e;
}

// =============================================================================
// Algorithm 2 – UpdateLIT  (unchanged from Augur)
// =============================================================================
void augurX::update_lit_from_lht(const lht_entry& evicted)
{
  auto* le = find_lit(evicted.hashed_pc);
  if (le == nullptr)
    return;

  if (evicted.recursive_producer) {
    if (le->recursive_confidence < MAX_CONF)
      ++le->recursive_confidence;
  } else {
    if (le->recursive_confidence > 0)
      --le->recursive_confidence;
  }
}

// =============================================================================
// Algorithm 2 – RecursiveLoadIdentification  (unchanged from Augur)
// =============================================================================
void augurX::recursive_load_identification(champsim::address ip,
                                           uint64_t          loaded_value,
                                           champsim::address mem_addr)
{
  if (loaded_value == 0)
    return;

  uint32_t hpc = hash_pc(ip);

  auto* le         = find_lit(hpc);
  bool  is_recursive = true;

  if (le != nullptr) {
    is_recursive = le->is_recursive();
  } else {
    int64_t estimated_offset =
        static_cast<int64_t>(mem_addr.to<uint64_t>() & 0x3F);
    le = insert_lit(hpc, estimated_offset);
    if (le == nullptr)
      return;
  }

  uint64_t my_base =
      mem_addr.to<uint64_t>() - static_cast<uint64_t>(le->offset);

  if (is_recursive) {
    for (auto& h : lht)
      if (h.loaded_value == my_base)
        h.recursive_producer = true;
  }

  if (lht.size() >= LHT_CAPACITY) {
    update_lit_from_lht(lht.front());
    lht.pop_front();
  }
  lht.push_back({hpc, loaded_value, false});

  // LDS ID assignment
  if (le->lds_id == 0) {
    bool found_producer = false;
    for (std::size_t i = 0; i < lit.size(); ++i) {
      auto& producer = lit[i];
      if (!producer.valid || producer.hashed_pc == hpc)
        continue;
      if (producer.is_recursive() && producer.is_producer) {
        le->lds_id         = producer.lds_id;
        le->producer_index = static_cast<uint8_t>(i);
        le->is_producer    = false;
        found_producer     = true;
        break;
      }
    }
    if (!found_producer) {
      le->lds_id   = next_lds_id++;
      if (next_lds_id > MAX_LDS_ID) next_lds_id = 1;
      le->is_producer = true;
    }
  }
}

// =============================================================================
// AHQ insertion & 2-successor metadata recording  (KEY CHANGE vs Augur)
//
// When a node address is evicted from the AHQ the evicted address is the
// trigger and the newly inserted address is ONE of its successors.
// We call metadata_entry::observe() which:
//   - increments confidence if this successor is already recorded, OR
//   - fills an empty slot, OR
//   - evicts the lowest-confidence slot if both are taken.
// Over time, a binary-tree node that always branches to two distinct children
// will naturally have both children sitting in slot 0 and slot 1 with high
// confidence, while a list node (single successor) will fill only slot 0.
// =============================================================================
void augurX::ahq_insert(uint64_t node_addr)
{
  // Deduplicate: don't record a self-correlation
  for (auto& a : ahq)
    if (a == node_addr)
      return;

  if (ahq.size() >= AHQ_SIZE) {
    uint64_t trigger = ahq.front();
    ahq.pop_front();

    // Record this successor observation in the 2-slot metadata entry
    metadata[trigger].observe(node_addr);
  }

  ahq.push_back(node_addr);
}

// =============================================================================
// Issue prefetches for ALL confident successors of a trigger  (KEY CHANGE)
//
// For each successor slot with confidence >= CONF_THRESHOLD we compute the
// concrete prefetch addresses by adding LIT offsets to the target node base,
// exactly as in vanilla Augur §5.3, but we do it for BOTH slots.
//
// Tree example:
//   trigger = node A  →  slot0: node B (conf 5), slot1: node C (conf 3)
//   Both B and C are above CONF_THRESHOLD=1, so we prefetch:
//     B+offset0, B+offset1, C+offset0, C+offset1
// =============================================================================
void augurX::issue_prefetches(const metadata_entry& entry)
{
  // Collect LDS ID from the first recursive producer in the LIT
  uint8_t lds_id = 0;
  for (auto& e : lit)
    if (e.valid && e.is_recursive() && e.is_producer) {
      lds_id = e.lds_id;
      break;
    }

  bool issued_any = false;
  bool mshr_ok    = intern_->get_mshr_occupancy_ratio() < 0.5;

  for (int s = 0; s < NUM_SUCCESSORS; ++s) {
    const auto& slot = entry.slots[s];

    // Skip empty slots or slots below confidence threshold
    if (slot.confidence < CONF_THRESHOLD)
      continue;

    uint64_t target_node = slot.target_node_addr;

    // Issue one prefetch per direct LIT element (same as vanilla Augur §5.3)
    bool issued_for_this_slot = false;
    for (auto& e : lit) {
      if (!e.valid)
        continue;
      if (lds_id != 0 && e.lds_id != lds_id)
        continue;
      if (!e.is_direct)
        continue;

      uint64_t pf_addr = target_node + static_cast<uint64_t>(e.offset);

      // Discard cross-page prefetches (paper §5.3)
      if ((pf_addr >> 12) != (target_node >> 12))
        continue;

      prefetch_line(champsim::address{pf_addr}, mshr_ok, 0);
      issued_for_this_slot = true;
      issued_any           = true;
    }

    // Fallback: no LIT offsets yet → prefetch the node base directly
    if (!issued_for_this_slot) {
      if ((target_node >> 12) != (target_node >> 12)) // page guard (always true here)
        continue;
      prefetch_line(champsim::address{target_node}, mshr_ok, 0);
      issued_any = true;
    }
  }

  (void)issued_any; // suppress unused-variable warning
}

// =============================================================================
// prefetcher_cache_operate
// =============================================================================
uint32_t augurX::prefetcher_cache_operate(champsim::address addr,
                                          champsim::address ip,
                                          uint8_t           cache_hit,
                                          bool              useful_prefetch,
                                          access_type       type,
                                          uint32_t          metadata_in)
{
  if (type != access_type::LOAD)
    return metadata_in;

  // ── Training path ──────────────────────────────────────────────────────────
  if (!cache_hit || useful_prefetch) {
    uint64_t approx_loaded_value = addr.to<uint64_t>(); // placeholder (see note)

    recursive_load_identification(ip, approx_loaded_value, addr);

    uint32_t hpc    = hash_pc(ip);
    auto*    le     = find_lit(hpc);
    int64_t  offset = (le != nullptr) ? le->offset : 0;
    bool     is_rec = (le == nullptr) ? true : le->is_recursive();

    if (is_rec) {
      uint64_t node_addr = to_node_addr(addr.to<uint64_t>(), offset);
      ahq_insert(node_addr);
    }
  }

  // ── Prediction path ────────────────────────────────────────────────────────
  uint32_t hpc = hash_pc(ip);
  auto*    le  = find_lit(hpc);
  if (le != nullptr && le->is_recursive()) {
    uint64_t node_addr = to_node_addr(addr.to<uint64_t>(), le->offset);
    auto it = metadata.find(node_addr);
    if (it != metadata.end()) {
      // AugurX: pass the whole entry so we can prefetch both successors
      issue_prefetches(it->second);
    }
  }

  return metadata_in;
}

// =============================================================================
// prefetcher_cache_fill  (unchanged)
// =============================================================================
uint32_t augurX::prefetcher_cache_fill(champsim::address addr, long set, long way,
                                       uint8_t prefetch,
                                       champsim::address evicted_addr,
                                       uint32_t          metadata_in)
{
  return metadata_in;
}

// =============================================================================
// prefetcher_cycle_operate  (unchanged)
// =============================================================================
void augurX::prefetcher_cycle_operate() {}