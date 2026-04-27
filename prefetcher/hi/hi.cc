#include "hi.h"

#include <algorithm>
#include <cassert>

#include "cache.h"

// =============================================================================
// LIT helpers
// =============================================================================

hi::lit_entry* hi::find_lit(uint32_t hpc)
{
  for (auto& e : lit)
    if (e.valid && e.hashed_pc == hpc)
      return &e;
  return nullptr;
}

hi::lit_entry* hi::insert_lit(uint32_t hpc, int64_t offset)
{
  // Already present?
  if (auto* e = find_lit(hpc); e != nullptr)
    return e;

  // Find an empty slot
  for (auto& e : lit) {
    if (!e.valid) {
      e = {};
      e.hashed_pc  = hpc;
      e.offset     = offset;
      e.recursive_confidence = MAX_CONF;
      e.valid      = true;
      return &e;
    }
  }

  // LIT is full: evict the first entry (simple FIFO; paper uses LRU)
  if (lit.size() >= LIT_CAPACITY) {
    lit.erase(lit.begin());
  }

  lit.push_back({});
  auto& e = lit.back();
  e.hashed_pc  = hpc;
  e.offset     = offset;
  e.recursive_confidence = MAX_CONF;
  e.valid      = true;
  return &e;
}

// =============================================================================
// Algorithm 2 – UpdateLIT (called when an LHT entry is evicted)
// =============================================================================
void hi::update_lit_from_lht(const lht_entry& evicted)
{
  auto* le = find_lit(evicted.hashed_pc);
  if (le == nullptr)
    return;

  if (evicted.recursive_producer) {
    // This load produced the address of a recursive load → it is itself recursive
    if (le->recursive_confidence < MAX_CONF)
      ++le->recursive_confidence;
  } else {
    // No recursive consumer observed → prune
    if (le->recursive_confidence > 0)
      --le->recursive_confidence;
  }
}

// =============================================================================
// Algorithm 2 – RecursiveLoadIdentification
// Called at the write-back stage for every load instruction.
// =============================================================================
void hi::recursive_load_identification(champsim::address ip,
                                          uint64_t loaded_value,
                                          champsim::address mem_addr)
{
  // Skip null pointers and (implicitly) stack loads – paper §5.1
  if (loaded_value == 0)
    return;

  uint32_t hpc = hash_pc(ip);

  // --- Step 1: CheckLIT (Algorithm 2, line 2) ---
  auto* le = find_lit(hpc);
  bool  is_recursive = true; // default: assume recursive on first sight

  if (le != nullptr) {
    is_recursive = le->is_recursive();
  } else {
    // LIT miss: register new entry (line 24) with offset derived from address
    // We estimate offset as the lower 6 bits of the address (heuristic).
    // In a real RTL implementation the ISA supplies the immediate directly.
    int64_t estimated_offset = static_cast<int64_t>(mem_addr.to<uint64_t>() & 0x3F);
    le = insert_lit(hpc, estimated_offset);
    if (le == nullptr)
      return;
  }

  // --- Step 2: scan LHT for an entry whose loaded_value matches our base addr ---
  // base_addr = load.addr - load.offset  (the node address that produced this load)
  uint64_t my_base = mem_addr.to<uint64_t>() - static_cast<uint64_t>(le->offset);

  if (is_recursive) {
    for (auto& h : lht) {
      if (h.loaded_value == my_base) {
        h.recursive_producer = true; // line 5
      }
    }
  }

  // --- Step 3: enqueue this load into the LHT (line 10) ---
  if (lht.size() >= LHT_CAPACITY) {
    // Evict the head (FIFO), update LIT (line 7-8)
    update_lit_from_lht(lht.front());
    lht.pop_front();
  }
  lht.push_back({hpc, loaded_value, false});

  // --- Step 4: update LDS layout (LDS ID / Producer Index) in LIT ---
  // Assign LDS ID: if this load depends on another recursive load that shares
  // its loaded value with our base address, inherit its LDS ID.
  if (le->lds_id == 0) {
    // Find a producer in LIT whose loaded value could match
    bool found_producer = false;
    for (std::size_t i = 0; i < lit.size(); ++i) {
      auto& producer = lit[i];
      if (!producer.valid || producer.hashed_pc == hpc)
        continue;
      if (producer.is_recursive() && producer.is_producer) {
        // Heuristic: share LDS ID with any existing recursive producer
        le->lds_id       = producer.lds_id;
        le->producer_index = static_cast<uint8_t>(i);
        le->is_producer  = false;
        found_producer   = true;
        break;
      }
    }
    if (!found_producer) {
      // New LDS root
      le->lds_id      = next_lds_id++;
      if (next_lds_id > MAX_LDS_ID) next_lds_id = 1;
      le->is_producer = true;
    }
  }
}

// =============================================================================
// AHQ insertion & metadata recording (Section 5.2)
// =============================================================================
void hi::ahq_insert(uint64_t node_addr)
{
  // If already in AHQ, do not add duplicate
  for (auto& a : ahq)
    if (a == node_addr)
      return;

  if (ahq.size() >= AHQ_SIZE) {
    // The evicted (oldest) node is the trigger; the new node is the target
    uint64_t trigger = ahq.front();
    ahq.pop_front();

    // Record / update correlation trigger → node_addr
    auto it = metadata.find(trigger);
    if (it == metadata.end()) {
      metadata[trigger] = {node_addr, true};
    } else {
      // Confidence-based replacement (1-bit, §5.4.1)
      if (it->second.target_node_addr == node_addr) {
        it->second.confident = true;
      } else {
        if (it->second.confident) {
          it->second.confident = false; // lose confidence first
        } else {
          it->second.target_node_addr = node_addr; // replace
          it->second.confident = true;
        }
      }
    }
  }

  ahq.push_back(node_addr);
}

// =============================================================================
// Issue prefetch requests for every direct element of target_node_addr (§5.3)
// =============================================================================
void hi::issue_prefetches(uint64_t target_node_addr)
{
  // Collect the LDS ID that owns the trigger (any recursive entry will do)
  // We prefetch target_node + offset for every direct LIT entry with is_direct.
  // For simplicity, use the LDS ID of the first recursive producer we find.
  uint8_t lds_id = 0;
  for (auto& e : lit) {
    if (e.valid && e.is_recursive() && e.is_producer) {
      lds_id = e.lds_id;
      break;
    }
  }

  bool issued_any = false;

  for (auto& e : lit) {
    if (!e.valid)
      continue;
    if (lds_id != 0 && e.lds_id != lds_id)
      continue;
    if (!e.is_direct)
      continue; // paper §5.3: only prefetch direct elements

    uint64_t pf_addr = target_node_addr + static_cast<uint64_t>(e.offset);

    // Discard if crossing 4 KB page boundary (paper §5.3)
    if ((pf_addr >> 12) != (target_node_addr >> 12))
      continue;

    // Check MSHR load and fill into L1 dcache (recursive loads are critical)
    bool fill_l1 = true;
    bool mshr_ok = intern_->get_mshr_occupancy_ratio() < 0.5;
    prefetch_line(champsim::address{pf_addr}, fill_l1 && mshr_ok, 0);
    issued_any = true;
  }

  // Fallback: if LIT has no usable entries yet, prefetch the node base itself
  if (!issued_any) {
    bool mshr_ok = intern_->get_mshr_occupancy_ratio() < 0.5;
    prefetch_line(champsim::address{target_node_addr}, mshr_ok, 0);
  }
}

// =============================================================================
// prefetcher_cache_operate – called on every L1-D access (hit or miss)
// =============================================================================
uint32_t hi::prefetcher_cache_operate(champsim::address addr,
                                         champsim::address ip,
                                         uint8_t cache_hit,
                                         bool useful_prefetch,
                                         access_type type,
                                         uint32_t metadata_in)
{
  // Only instrument loads
  if (type != access_type::LOAD)
    return metadata_in;

  // -----------------------------------------------------------------
  // Training path (§5.1 + §5.2):
  // We observe the miss / prefetch-hit stream to extract correlations.
  // On a miss or useful prefetch hit, run the pruning logic.
  // -----------------------------------------------------------------
  if (!cache_hit || useful_prefetch) {
    // In ChampSim we don't have the loaded value at this call site.
    // We approximate: use the address itself shifted right by one cache-line
    // as a proxy for a pointer value (common for linked-list nodes allocated
    // sequentially). For production use, a memory-value shadow structure
    // (as noted in §6.1.1 of the paper) should be used.
    uint64_t approx_loaded_value = addr.to<uint64_t>(); // placeholder

    recursive_load_identification(ip, approx_loaded_value, addr);

    // Compute node address for this access using LIT offset
    uint32_t hpc = hash_pc(ip);
    auto* le = find_lit(hpc);
    int64_t offset = (le != nullptr) ? le->offset : 0;

    // Only feed the AHQ with addresses from recursive loads (§5.2)
    bool is_rec = (le == nullptr) ? true : le->is_recursive();
    if (is_rec) {
      uint64_t node_addr = to_node_addr(addr.to<uint64_t>(), offset);
      ahq_insert(node_addr);
    }
  }

  // -----------------------------------------------------------------
  // Prediction path (§5.3):
  // On arrival of a recursive load, look up metadata and prefetch.
  // -----------------------------------------------------------------
  uint32_t hpc = hash_pc(ip);
  auto* le = find_lit(hpc);
  if (le != nullptr && le->is_recursive()) {
    uint64_t node_addr = to_node_addr(addr.to<uint64_t>(), le->offset);
    auto it = metadata.find(node_addr);
    if (it != metadata.end()) {
      issue_prefetches(it->second.target_node_addr);
    }
  }

  return metadata_in;
}

// =============================================================================
// prefetcher_cache_fill – called when a line is filled into the cache
// =============================================================================
uint32_t hi::prefetcher_cache_fill(champsim::address addr, long set, long way,
                                      uint8_t prefetch,
                                      champsim::address evicted_addr,
                                      uint32_t metadata_in)
{
  return metadata_in;
}

// =============================================================================
// prefetcher_cycle_operate – called every cycle
// =============================================================================
void hi::prefetcher_cycle_operate()
{
  // hi does not use a lookahead pipeline like ip_stride; all prefetching
  // happens synchronously in prefetcher_cache_operate.
}