/*
 * triangel.cc  –  ChampSim (modern JSON-config API) port of the Triangel
 *                 temporal prefetcher.
 *
 * Reference:
 *   Sam Ainsworth and Lev Mukhanov,
 *   "Triangel: A High-Performance, Accurate, Timely On-Chip Temporal Prefetcher"
 *   ISCA 2024.
 *
 * Original gem5 implementation: https://github.com/SamAinsworth/gem5-triangel
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *  Modern ChampSim module pattern (JSON-config / modular build):
 *
 *  class triangel : champsim::modules::prefetcher {
 *      TriangelEngine engine;
 *  public:
 *      using champsim::modules::prefetcher::prefetcher;
 *      uint32_t prefetcher_cache_operate(...);
 *      uint32_t prefetcher_cache_fill(...);
 *      void     prefetcher_final_stats();
 *  };
 *
 *  ChampSim discovers the class by filename: this file is "triangel.cc" so the
 *  class must be named "triangel".
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "triangel.h"
#include "cache.h"

#include <algorithm>
#include <cassert>
#include <iostream>

uint32_t triangel::prefetcher_cache_operate(champsim::address addr, champsim::address ip,
                                            bool cache_hit, bool useful_prefetch,
                                            access_type /* type */, uint32_t metadata_in)
{
    // Train and prefetch only on:
    //   (a) demand misses        -> !cache_hit
    //   (b) tagged prefetch hits -> cache_hit && useful_prefetch
    // Plain demand hits carry no new miss-stream information.
    if (cache_hit && !useful_prefetch)
        return metadata_in;

    auto candidates = engine.operate(addr.to<uint64_t>(), ip.to<uint64_t>(), cache_hit, useful_prefetch);

    for (uint64_t pf_addr : candidates) {
        // Fill into the current cache level.
        prefetch_line(champsim::address{pf_addr}, true, 0);
    }

    return metadata_in;
}

uint32_t triangel::prefetcher_cache_fill(champsim::address addr, long /*set*/,
                                         long /*way*/, bool /*prefetch*/,
                                         champsim::address /*evicted_addr*/,
                                         uint32_t metadata_in)
{
    engine.on_fill(addr.to<uint64_t>());
    return metadata_in;
}

void triangel::prefetcher_final_stats()
{
    std::cout << "[Triangel] prefetches issued = "
              << engine.stat_pf_issued << '\n';
}

// ═════════════════════════════════════════════════════════════════════════════
//  TriangelEngine implementation
// ═════════════════════════════════════════════════════════════════════════════

// ── Construction ─────────────────────────────────────────────────────────────

TriangelEngine::TriangelEngine() : rng(42) {
    // Tables are zero-/value-initialised by their aggregate constructors.
    // Set all Markov RRIP fields to the "distant" value.
    for (auto& set : markov)
        for (auto& e : set) e.rrip = 7;
}

// ── Index / hash helpers ──────────────────────────────────────────────────────

uint32_t TriangelEngine::pc_to_set(uint64_t pc) {
    uint64_t h = pc ^ (pc >> 16) ^ (pc >> 32) ^ (pc >> 48);
    return static_cast<uint32_t>(h % TRAIN_SETS);
}

uint64_t TriangelEngine::pc_to_tag(uint64_t pc) {
    // 10-bit hashed tag (matching TrainEntry::pc_tag width)
    uint64_t h = pc ^ (pc >> 10) ^ (pc >> 20) ^ (pc >> 30);
    return h & 0x3FFull;
}

uint32_t TriangelEngine::cl_to_markov_set(uint64_t cl) {
    uint64_t h = cl ^ (cl >> 14) ^ (cl >> 28);
    return static_cast<uint32_t>(h % MARKOV_SETS);
}

uint32_t TriangelEngine::cl_to_hs_set(uint64_t cl) {
    uint64_t h = cl ^ (cl >> 8) ^ (cl >> 16);
    return static_cast<uint32_t>(h % HS_SETS);
}

uint32_t TriangelEngine::cl_to_mrb_set(uint64_t cl) {
    uint64_t h = cl ^ (cl >> 7);
    return static_cast<uint32_t>(h % MRB_SETS);
}

// ── Confidence helpers ────────────────────────────────────────────────────────

void TriangelEngine::sat_inc(int8_t& v) {
    if (v < CONF_MAX) v++;
}

void TriangelEngine::sat_dec(int8_t& v, int8_t amount) {
    v = (v > amount) ? (int8_t)(v - amount) : (int8_t)0;
}

void TriangelEngine::upd_reuse(TrainEntry* te, bool inc) {
    if (inc) sat_inc(te->reuse_conf);
    else     sat_dec(te->reuse_conf, 1);
}

// BasePatternConf: +1 / -2  (≥66% threshold)
void TriangelEngine::upd_base_pconf(TrainEntry* te, bool inc) {
    if (inc) sat_inc(te->base_pconf);
    else     sat_dec(te->base_pconf, 2);
}

// HighPatternConf: +1 / -5  (≥83% = 5/6 threshold)
void TriangelEngine::upd_high_pconf(TrainEntry* te, bool inc) {
    if (inc) sat_inc(te->high_pconf);
    else     sat_dec(te->high_pconf, 5);
}

// Lookahead bit (paper §IV-E): set when HighPatternConf saturates at 15,
// clear when BasePatternConf falls back to its initial value (8).
void TriangelEngine::upd_lookahead(TrainEntry* te) {
    if (!te->lookahead && te->high_pconf >= HIGH_THRESH_SET_LA2)
        te->lookahead = true;
    else if (te->lookahead && te->base_pconf < BASE_THRESH_CLR_LA2)
        te->lookahead = false;
}

// ── Aggression predicates (paper §IV-E) ──────────────────────────────────────

// Allow metadata storage and prefetch issuance only when both confidence
// counters are strictly above their initial (half-way) value.
bool TriangelEngine::should_act(const TrainEntry* te) const {
    return te->reuse_conf > CONF_INIT && te->base_pconf > CONF_INIT;
}

// Issue up to MAX_DEGREE (4) chained prefetches when HighPatternConf is
// above its initial value; otherwise issue just 1.
uint32_t TriangelEngine::get_degree(const TrainEntry* te) const {
    return (te->high_pconf > CONF_INIT) ? MAX_DEGREE : 1;
}

// ── Training table ────────────────────────────────────────────────────────────

uint16_t TriangelEngine::train_id_of(const TrainEntry* te) const
{
    const auto* base = &train_table[0][0];
    return static_cast<uint16_t>(te - base);
}

TrainEntry* TriangelEngine::train_by_id(uint16_t train_id)
{
    if (train_id >= TRAIN_SETS * TRAIN_WAYS)
        return nullptr;

    const uint32_t set = train_id / TRAIN_WAYS;
    const uint32_t way = train_id % TRAIN_WAYS;
    return &train_table[set][way];
}

TrainEntry* TriangelEngine::train_lookup(uint64_t pc) {
    uint32_t set = pc_to_set(pc);
    uint64_t tag = pc_to_tag(pc);
    for (auto& e : train_table[set])
        if (e.valid && e.pc_tag == tag) return &e;
    return nullptr;
}

TrainEntry* TriangelEngine::train_get_or_alloc(uint64_t pc) {
    uint32_t set = pc_to_set(pc);
    uint64_t tag = pc_to_tag(pc);

    // Hit
    for (auto& e : train_table[set])
        if (e.valid && e.pc_tag == tag) return &e;

    // Invalid slot
    for (auto& e : train_table[set]) {
        if (!e.valid) {
            e = TrainEntry{};
            e.valid = true;
            e.pc_tag = tag;
            return &e;
        }
    }

    // LRU eviction
    TrainEntry* victim = &train_table[set][0];
    for (auto& e : train_table[set])
        if (e.lru_age > victim->lru_age) victim = &e;
    *victim = TrainEntry{};
    victim->valid  = true;
    victim->pc_tag = tag;
    return victim;
}

// ── History Sampler ───────────────────────────────────────────────────────────

HSEntry* TriangelEngine::hs_find(uint64_t cl_addr, uint16_t train_id) {
    uint32_t set = cl_to_hs_set(cl_addr);
    for (auto& e : hs_table[set])
        if (e.valid && e.lookup_addr == cl_addr && e.train_id == train_id)
            return &e;
    return nullptr;
}

// Full History-Sampler update (paper §IV-D, Fig. 7).
// - Checks if LastAddr[0] appears in the sampler → ReuseConf update
// - Checks if the sequence (LastAddr[0] → cur_cl) repeats → PatternConf update
// - May insert a new entry for LastAddr[0] with probability proportional to
//   SamplerSize / (MaxSize × 2^(SampleRate-8))
void TriangelEngine::scs_invalidate_and_penalize(SCSEntry& entry)
{
    if (entry.valid) {
        if (TrainEntry* owner = train_by_id(entry.train_id); owner != nullptr && owner->valid) {
            upd_base_pconf(owner, false);
            upd_high_pconf(owner, false);
        }
        entry.valid = false;
    }
}

void TriangelEngine::hs_update(TrainEntry* te, uint64_t cur_cl) {
    const uint16_t train_id = train_id_of(te);
    const bool scs_hit = scs_check(cur_cl, train_id);
    bool exact_hs_match = false;

    // ── Step 1: check for a HS hit on LastAddr[0] ──────────────────────────
    if (te->last_addr0 != 0) {
        HSEntry* hs = hs_find(te->last_addr0, train_id);

        if (hs != nullptr) {
            hs->used = true;

            // ReuseConf: is the reuse distance short enough for the Markov table?
            uint32_t dist = te->timestamp - hs->timestamp;
            upd_reuse(te, dist < MARKOV_TOTAL);

            // PatternConf: did the sequence repeat exactly?
            if (cur_cl == hs->target_addr) {
                exact_hs_match = true;
                upd_base_pconf(te, true);
                upd_high_pconf(te, true);
            } else {
                // Defer judgement to the Second-Chance Sampler. If this exact
                // mismatch still leads to a timely use later, SCS will reward
                // it. Otherwise the entry is penalized when it expires/evicts.
                scs_insert(hs->target_addr, train_id);
            }
        }
    }

    if (scs_hit && !exact_hs_match) {
        upd_base_pconf(te, true);
        upd_high_pconf(te, true);
    }

    // ── Step 2: update lookahead bit ──────────────────────────────────────
    upd_lookahead(te);

    // ── Step 3: probabilistic insertion into HS ───────────────────────────
    // Probability = SamplerSize / (MaxSize × 2^(SampleRate-8))
    // SamplerSize = HS_SETS × HS_WAYS = 512
    // MaxSize     = MARKOV_TOTAL      = 196 608
    // At default SampleRate=8: p ≈ 512/196608 ≈ 1/384
    if (te->last_addr0 != 0) {
        uint32_t shift = (te->sample_rate >= 8) ? (te->sample_rate - 8) : 0;
        // Avoid overflow: cap shift at 23 (2^23 × 196608 > 2^32)
        if (shift > 23) shift = 23;
        uint32_t denom = (MARKOV_TOTAL << shift);
        uint32_t numer = (HS_SETS * HS_WAYS);
        bool do_insert = ((rng() % denom) < numer);

        if (do_insert) {
            uint32_t set = cl_to_hs_set(te->last_addr0);

            // Find victim: prefer invalid slot, then LRU
            HSEntry* victim = nullptr;
            for (auto& e : hs_table[set])
                if (!e.valid) { victim = &e; break; }
            if (!victim) {
                victim = &hs_table[set][0];
                for (auto& e : hs_table[set])
                    if (e.lru_age > victim->lru_age) victim = &e;

                // Adjust sample rate based on victim freshness (paper §IV-D3)
                if (!victim->used) {
                    uint32_t v_dist = MARKOV_TOTAL;
                    if (TrainEntry* owner = train_by_id(victim->train_id); owner != nullptr && owner->valid)
                        v_dist = owner->timestamp - victim->timestamp;
                    if (v_dist >= MARKOV_TOTAL) {
                        // Stale victim: safe; boost this PC's insertion rate and
                        // reduce the stale victim owner's reuse confidence.
                        if (TrainEntry* owner = train_by_id(victim->train_id); owner != nullptr && owner->valid)
                            upd_reuse(owner, false);
                        if (te->sample_rate < 15) te->sample_rate++;
                    } else {
                        // Evicting potentially useful entry: slow down
                        if (te->sample_rate > 0) te->sample_rate--;
                    }
                }
            }

            // Insert
            victim->lookup_addr = te->last_addr0;
            victim->target_addr = cur_cl;
            victim->train_id    = train_id;
            victim->timestamp   = te->timestamp;
            victim->valid       = true;
            victim->used        = false;
            victim->lru_age     = 0;
            // Age all other entries in set
            for (auto& e : hs_table[set])
                if (&e != victim && e.valid) e.lru_age++;
        }
    }
}

// ── Second-Chance Sampler ─────────────────────────────────────────────────────

void TriangelEngine::scs_insert(uint64_t target_cl, uint16_t train_id) {
    for (auto& entry : scs) {
        if (entry.valid && entry.target_addr == target_cl && entry.train_id == train_id) {
            entry.fill_count = global_fill_count;
            return;
        }
    }

    SCSEntry& slot    = scs[scs_hand];
    if (slot.valid)
        scs_invalidate_and_penalize(slot);
    slot.target_addr  = target_cl;
    slot.train_id     = train_id;
    slot.fill_count   = global_fill_count;
    slot.valid        = true;
    scs_hand          = (scs_hand + 1) % SCS_SIZE;
}

bool TriangelEngine::scs_check(uint64_t cur_cl, uint16_t train_id) {
    for (auto& e : scs) {
        if (!e.valid) continue;
        uint32_t age = global_fill_count - e.fill_count;

        if (age > SCS_WINDOW) {
            scs_invalidate_and_penalize(e);
            continue;
        }

        if (e.target_addr == cur_cl && e.train_id == train_id) {
            e.valid = false;  // consume
            return true;
        }
    }
    return false;
}

// ── Markov table ─────────────────────────────────────────────────────────────

MarkovEntry* TriangelEngine::markov_find(uint64_t cl_addr) {
    uint32_t set = cl_to_markov_set(cl_addr);
    for (auto& e : markov[set])
        if (e.valid && e.lookup_addr == cl_addr) return &e;
    return nullptr;
}

// SRRIP victim selection: scan for RRIP==7; if none found, age all entries.
MarkovEntry* TriangelEngine::markov_victim(uint32_t set) {
    for (auto& e : markov[set])
        if (!e.valid) return &e;

    for (;;) {
        for (auto& e : markov[set])
            if (e.rrip >= 7) return &e;
        for (auto& e : markov[set])
            if (e.rrip < 7) e.rrip++;
    }
}

// Train Markov table: paper §II, §IV-C, confidence bit rule §III-D.
void TriangelEngine::markov_train(uint64_t index_cl, uint64_t target_cl) {
    uint32_t set = cl_to_markov_set(index_cl);
    MarkovEntry* e = nullptr;
    for (auto& me : markov[set])
        if (me.valid && me.lookup_addr == index_cl) { e = &me; break; }

    if (e) {
        // Hit: update confidence bit and optionally replace target
        if (e->prefetch_tgt == target_cl) {
            e->confidence = true;       // same target → gain confidence
        } else if (!e->confidence) {
            e->prefetch_tgt = target_cl; // different target, low confidence → replace
        }
        e->rrip = 0;   // mark as recently used
    } else {
        // Miss: insert new entry (SRRIP: insert at distant position)
        MarkovEntry* v = markov_victim(set);
        v->lookup_addr  = index_cl;
        v->prefetch_tgt = target_cl;
        v->confidence   = false;
        v->valid        = true;
        v->rrip         = 2;  // SRRIP default: not MRU, not LRU
    }
}

// ── Metadata Reuse Buffer ─────────────────────────────────────────────────────

MRBEntry* TriangelEngine::mrb_find(uint64_t cl_addr) {
    uint32_t set = cl_to_mrb_set(cl_addr);
    for (auto& e : mrb[set])
        if (e.valid && e.lookup_addr == cl_addr) return &e;
    return nullptr;
}

void TriangelEngine::mrb_insert(uint64_t lookup_cl, uint64_t target_cl) {
    uint32_t set = cl_to_mrb_set(lookup_cl);

    // Update payload if already present, but do not reorder on hit: FIFO.
    for (auto& e : mrb[set]) {
        if (e.valid && e.lookup_addr == lookup_cl) {
            e.prefetch_tgt = target_cl;
            return;
        }
    }

    // Find FIFO victim
    MRBEntry* v = nullptr;
    for (auto& e : mrb[set]) if (!e.valid) { v = &e; break; }
    if (!v) {
        v = &mrb[set][0];
        for (auto& e : mrb[set]) if (e.fifo_stamp < v->fifo_stamp) v = &e;
    }

    v->lookup_addr  = lookup_cl;
    v->prefetch_tgt = target_cl;
    v->valid        = true;
    v->fifo_stamp   = mrb_insertion_stamp++;
}

// ── on_fill ───────────────────────────────────────────────────────────────────

void TriangelEngine::on_fill(uint64_t /*addr*/) {
    global_fill_count++;
}

// ── Main operate() ────────────────────────────────────────────────────────────
//
// Algorithm (paper §IV-A, Fig. 3):
//  1. Retrieve / create training-table entry for this PC.
//  2. Run History Sampler update → confidence counter updates.
//  3. If should_act(): train the Markov table.
//  4. If should_act(): issue up to get_degree() chained prefetches via MRB.
//  5. Update training-table shift register and timestamp.

std::vector<uint64_t> TriangelEngine::operate(uint64_t addr, uint64_t pc,
                                               bool cache_hit,
                                               bool useful_prefetch) {
    std::vector<uint64_t> prefetches;

    // Ignore plain demand hits; only train on misses and tagged prefetch hits.
    if (cache_hit && !useful_prefetch)
        return prefetches;

    uint64_t cl_addr = aln(addr);

    // ── 1. Training-table entry ──────────────────────────────────────────────
    TrainEntry* te = train_get_or_alloc(pc);

    // ── 2. History Sampler update ────────────────────────────────────────────
    hs_update(te, cl_addr);

    // Advance per-PC timestamp
    te->timestamp++;

    // ── 3. Markov table training ─────────────────────────────────────────────
    if (should_act(te) && te->last_addr0 != 0) {
        // With lookahead-2 we index by last_addr1 (two steps back) so that the
        // stored pair skips one position ahead in the stream, improving timeliness.
        uint64_t idx_cl = te->lookahead ? te->last_addr1 : te->last_addr0;
        if (idx_cl != 0) {
            // MRB optimisation (paper §IV-F, last paragraph):
            // If the MRB already holds (idx_cl → cl_addr) with the same target,
            // and the Markov confidence is already set, we can skip the write.
            MRBEntry* mrb_e = mrb_find(idx_cl);
            MarkovEntry* markov_e = markov_find(idx_cl);
            bool skip_write = (mrb_e != nullptr && mrb_e->prefetch_tgt == cl_addr &&
                               markov_e != nullptr && markov_e->prefetch_tgt == cl_addr &&
                               markov_e->confidence);
            if (!skip_write)
                markov_train(idx_cl, cl_addr);
        }
    }

    // ── 4. Prefetch generation ───────────────────────────────────────────────
    if (should_act(te)) {
        uint32_t degree = get_degree(te);
        uint64_t lookup = cl_addr;

        for (uint32_t d = 0; d < degree; d++) {
            // Check MRB first to avoid redundant L3 Markov accesses (paper §IV-F)
            uint64_t target = 0;
            MRBEntry* mrb_hit = mrb_find(lookup);
            if (mrb_hit) {
                target = mrb_hit->prefetch_tgt;
            } else {
                MarkovEntry* me = markov_find(lookup);
                if (!me) break;
                target = me->prefetch_tgt;
                // Cache this result in MRB for the next degree step
                mrb_insert(lookup, target);
                // Mark Markov entry recently used
                me->rrip = 0;
            }

            if (target == 0 || target == lookup) break;

            prefetches.push_back(target);
            stat_pf_issued++;
            lookup = target;
        }
    }

    // ── 5. Update shift register and LRU ────────────────────────────────────
    te->last_addr1 = te->last_addr0;
    te->last_addr0 = cl_addr;

    // Update LRU ages in this training-table set
    uint32_t tset = pc_to_set(pc);
    for (auto& e : train_table[tset])
        if (e.valid) e.lru_age++;
    te->lru_age = 0;

    return prefetches;
}
