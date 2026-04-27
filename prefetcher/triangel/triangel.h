#ifndef TRIANGEL_H
#define TRIANGEL_H

/*
 * triangel.h  –  ChampSim (modern JSON-config API) port of the Triangel
 *                temporal prefetcher.
 *
 * Reference:
 *   Sam Ainsworth and Lev Mukhanov,
 *   "Triangel: A High-Performance, Accurate, Timely On-Chip Temporal Prefetcher"
 *   ISCA 2024.
 *
 * Original gem5 implementation: https://github.com/SamAinsworth/gem5-triangel
 *
 * ── Integration ────────────────────────────────────────────────────────────
 * 1. Create directory   ChampSim/prefetcher/triangel/
 * 2. Copy triangel.h and triangel.cc into it.
 * 3. In your JSON config set:
 *        "L2C": { "prefetcher": "triangel" }
 *
 * ── Design notes ───────────────────────────────────────────────────────────
 * The paper stores the Markov table in a *partition of the L3 cache*.
 * The modern ChampSim API gives no direct access to cache internals, so the
 * Markov table is modelled as an independent SRAM of equivalent capacity
 * (paper default 1 MiB → 196 608 × 42-bit entries, 8-way SA with SRRIP).
 * All other structures match the paper exactly.
 */

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "access_type.h"
#include "address.h"
#include "champsim.h"
#include "modules.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Sizing constants  (paper §IV-H / Table I)
// ─────────────────────────────────────────────────────────────────────────────

// Training table: 512 entries, 2-way SA (paper Fig. 5)
static constexpr uint32_t TRAIN_SETS  = 256;
static constexpr uint32_t TRAIN_WAYS  = 2;

// History Sampler: 512 entries, 2-way SA (paper Fig. 7)
static constexpr uint32_t HS_SETS     = 256;
static constexpr uint32_t HS_WAYS     = 2;

// Second-Chance Sampler: 64 entries, FIFO (paper Fig. 8)
static constexpr uint32_t SCS_SIZE    = 64;

// Metadata Reuse Buffer: 256 entries, 2-way SA (paper §IV-F)
static constexpr uint32_t MRB_SETS    = 128;
static constexpr uint32_t MRB_WAYS    = 2;

// Markov table: 196 608 entries ≈ 1 MiB of 42-bit entries, 8-way SA
static constexpr uint32_t MARKOV_WAYS  = 8;
static constexpr uint32_t MARKOV_SETS  = 24576;             // 196608 / 8
static constexpr uint32_t MARKOV_TOTAL = MARKOV_SETS * MARKOV_WAYS;

// Confidence counter range: 4-bit saturating (0–15)
static constexpr int8_t CONF_MAX  = 15;
static constexpr int8_t CONF_INIT = 8;   // "initial value" = half-way point

// Lookahead control thresholds (paper §IV-E)
static constexpr int8_t HIGH_THRESH_SET_LA2 = 15;  // HighPatternConf → lookahead=2
static constexpr int8_t BASE_THRESH_CLR_LA2 =  8;  // BasePatternConf → back to LA=1

// SCS window: 512 L2 fills (paper §IV-D2)
static constexpr uint32_t SCS_WINDOW = 512;

// Maximum prefetch degree
static constexpr uint32_t MAX_DEGREE = 4;

// ─────────────────────────────────────────────────────────────────────────────
//  Entry types
// ─────────────────────────────────────────────────────────────────────────────

struct TrainEntry {
    uint64_t pc_tag      = 0;
    uint64_t last_addr0  = 0;   // most-recent miss/prefetch-hit (shift reg[0])
    uint64_t last_addr1  = 0;   // one older (shift reg[1], used for lookahead-2)
    uint32_t timestamp   = 0;   // per-PC local counter
    int8_t   reuse_conf  = CONF_INIT;  // ReuseConfidence
    int8_t   base_pconf  = CONF_INIT;  // BasePatternConf  (+1 / -2 bias)
    int8_t   high_pconf  = CONF_INIT;  // HighPatternConf  (+1 / -5 bias)
    uint8_t  sample_rate = 8;
    bool     lookahead   = false;
    bool     valid       = false;
    uint32_t lru_age     = 0;
};

struct HSEntry {
    uint64_t lookup_addr = 0;
    uint64_t target_addr = 0;
    uint16_t train_id    = 0;
    uint32_t timestamp   = 0;
    bool     valid       = false;
    bool     used        = false;
    uint32_t lru_age     = 0;
};

struct SCSEntry {
    uint64_t target_addr = 0;
    uint16_t train_id    = 0;
    uint32_t fill_count  = 0;
    bool     valid       = false;
};

struct MarkovEntry {
    uint64_t lookup_addr  = 0;
    uint64_t prefetch_tgt = 0;
    bool     confidence   = false;
    bool     valid        = false;
    uint8_t  rrip         = 7;
};

struct MRBEntry {
    uint64_t lookup_addr  = 0;
    uint64_t prefetch_tgt = 0;
    bool     valid        = false;
    uint32_t fifo_stamp   = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
//  TriangelEngine  –  framework-independent algorithmic core
//  The ChampSim module class owns one instance and delegates to it.
// ─────────────────────────────────────────────────────────────────────────────

class TriangelEngine {
public:
    TriangelEngine();

    // Called on every L2 demand miss or tagged prefetch hit (useful_prefetch).
    // Returns a list of cache-line-aligned addresses to prefetch.
    std::vector<uint64_t> operate(uint64_t addr, uint64_t pc,
                                  bool cache_hit, bool useful_prefetch);

    // Called on every L2 fill (increments global fill counter for SCS).
    void on_fill(uint64_t addr);

    uint64_t stat_pf_issued = 0;

private:
    std::array<std::array<TrainEntry,  TRAIN_WAYS>,  TRAIN_SETS>   train_table{};
    std::array<std::array<HSEntry,     HS_WAYS>,     HS_SETS>      hs_table{};
    std::array<SCSEntry,               SCS_SIZE>                   scs{};
    std::array<std::array<MarkovEntry, MARKOV_WAYS>, MARKOV_SETS>  markov{};
    std::array<std::array<MRBEntry,    MRB_WAYS>,    MRB_SETS>     mrb{};

    uint32_t  global_fill_count = 0;
    uint32_t  scs_hand          = 0;
    uint32_t  mrb_insertion_stamp = 0;
    std::mt19937 rng{42};

    // Training table
    TrainEntry* train_lookup(uint64_t pc);
    TrainEntry* train_get_or_alloc(uint64_t pc);
    TrainEntry* train_by_id(uint16_t train_id);
    uint16_t    train_id_of(const TrainEntry* te) const;

    // History Sampler
    void     hs_update(TrainEntry* te, uint64_t cur_cl);
    HSEntry* hs_find(uint64_t cl_addr, uint16_t train_id);

    // Second-Chance Sampler
    void scs_insert(uint64_t target_cl, uint16_t train_id);
    bool scs_check(uint64_t cur_cl, uint16_t train_id);
    void scs_invalidate_and_penalize(SCSEntry& entry);

    // Markov table
    MarkovEntry* markov_find(uint64_t cl_addr);
    void         markov_train(uint64_t index_cl, uint64_t target_cl);
    MarkovEntry* markov_victim(uint32_t set);

    // Metadata Reuse Buffer
    MRBEntry* mrb_find(uint64_t cl_addr);
    void      mrb_insert(uint64_t lookup_cl, uint64_t target_cl);

    // Confidence updates
    static void sat_inc(int8_t& v);
    static void sat_dec(int8_t& v, int8_t amount);
    void upd_reuse(TrainEntry* te, bool inc);
    void upd_base_pconf(TrainEntry* te, bool inc);
    void upd_high_pconf(TrainEntry* te, bool inc);
    void upd_lookahead(TrainEntry* te);

    // Aggression control
    bool     should_act(const TrainEntry* te) const;
    uint32_t get_degree(const TrainEntry* te) const;

    // Index / hash helpers
    static uint32_t pc_to_set(uint64_t pc);
    static uint64_t pc_to_tag(uint64_t pc);
    static uint32_t cl_to_markov_set(uint64_t cl);
    static uint32_t cl_to_hs_set(uint64_t cl);
    static uint32_t cl_to_mrb_set(uint64_t cl);

    // Cache-line alignment (64-byte lines, matching ChampSim default)
    static constexpr int CL_SHIFT = 6;
    static uint64_t aln(uint64_t a) { return (a >> CL_SHIFT) << CL_SHIFT; }
};

class triangel : public champsim::modules::prefetcher {
    TriangelEngine engine;

public:
    using champsim::modules::prefetcher::prefetcher;

    uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip,
                                      bool cache_hit, bool useful_prefetch,
                                      access_type type, uint32_t metadata_in);

    uint32_t prefetcher_cache_fill(champsim::address addr, long set,
                                   long way, bool prefetch,
                                   champsim::address evicted_addr,
                                   uint32_t metadata_in);

    void prefetcher_final_stats();
};

#endif // TRIANGEL_H
