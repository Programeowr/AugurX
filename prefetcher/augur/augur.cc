#include "augur.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>
#include <vector>

#include "cache.h"

namespace
{
constexpr std::size_t LHT_SIZE = 32;
constexpr std::size_t LIT_SIZE = 16;
constexpr std::size_t AHQ_SIZE = 8;
constexpr std::size_t CORRELATION_TABLE_SIZE = 128;
constexpr std::size_t PC_TABLE_SIZE = 64;
constexpr std::size_t RECENT_PREFETCH_SIZE = 32;

constexpr uint8_t MAX_RECURSIVE_CONF = 7;
constexpr uint8_t RECURSIVE_THRESHOLD = 2;
constexpr uint8_t MAX_CORRELATION_CONF = 3;
constexpr uint8_t PREFETCH_CONFIDENCE_THRESHOLD = 1;
constexpr uint8_t LOOKAHEAD_CONFIDENCE_THRESHOLD = 1;
constexpr int PREFETCH_DEGREE = 3;

constexpr uint64_t NODE_GRANULARITY = 32;
constexpr uint64_t NODE_MASK = ~(NODE_GRANULARITY - 1);
constexpr uint64_t CACHELINE_GRANULARITY = 64;
constexpr uint64_t CACHELINE_MASK = ~(CACHELINE_GRANULARITY - 1);
constexpr uint64_t PAGE_MASK = ~0xfffULL;

struct lht_entry {
  uint64_t pc;
  uint64_t node_addr;
  bool recursive_producer = false;
};

struct lit_entry {
  uint64_t pc = 0;
  uint8_t recursive_conf = 0;
  uint64_t last_touch = 0;
};

struct correlation_entry {
  uint64_t target_node = 0;
  uint8_t confidence = 0;
  uint64_t last_touch = 0;
};

struct pc_entry {
  uint64_t pc = 0;
  uint64_t last_line = 0;
  int64_t delta = 0;
  uint8_t confidence = 0;
  uint64_t last_touch = 0;
};

std::deque<lht_entry> lht;
std::vector<lit_entry> lit;
std::deque<uint64_t> ahq;
std::deque<uint64_t> recent_prefetches;
std::unordered_map<uint64_t, correlation_entry> correlations;
std::vector<pc_entry> pc_table;
uint64_t lit_timestamp = 0;
uint64_t correlation_timestamp = 0;
uint64_t pc_timestamp = 0;

uint64_t align_node(uint64_t addr)
{
  return addr & NODE_MASK;
}

uint64_t align_cacheline(uint64_t addr)
{
  return addr & CACHELINE_MASK;
}

lit_entry& get_lit_entry(uint64_t pc)
{
  ++lit_timestamp;

  auto hit = std::find_if(std::begin(lit), std::end(lit), [pc](const auto& entry) { return entry.pc == pc; });
  if (hit != std::end(lit)) {
    hit->last_touch = lit_timestamp;
    return *hit;
  }

  if (lit.size() < LIT_SIZE) {
    lit.push_back({pc, MAX_RECURSIVE_CONF, lit_timestamp});
    return lit.back();
  }

  auto victim = std::min_element(std::begin(lit), std::end(lit), [](const auto& lhs, const auto& rhs) { return lhs.last_touch < rhs.last_touch; });
  *victim = {pc, MAX_RECURSIVE_CONF, lit_timestamp};
  return *victim;
}

void update_lit_on_lht_eviction(const lht_entry& evicted)
{
  auto hit = std::find_if(std::begin(lit), std::end(lit), [pc = evicted.pc](const auto& entry) { return entry.pc == pc; });
  if (hit == std::end(lit))
    return;

  if (evicted.recursive_producer) {
    hit->recursive_conf = std::min<uint8_t>(MAX_RECURSIVE_CONF, hit->recursive_conf + 1);
  } else {
    hit->recursive_conf = (hit->recursive_conf > 0) ? hit->recursive_conf - 1 : 0;
  }
}

void enqueue_lht(uint64_t pc, uint64_t node_addr)
{
  if (lht.size() >= LHT_SIZE) {
    update_lit_on_lht_eviction(lht.front());
    lht.pop_front();
  }

  lht.push_back({pc, node_addr, false});
}

void update_correlation(uint64_t trigger_node, uint64_t target_node)
{
  ++correlation_timestamp;

  auto hit = correlations.find(trigger_node);
  if (hit == correlations.end()) {
    if (correlations.size() >= CORRELATION_TABLE_SIZE) {
      auto victim = std::min_element(std::begin(correlations), std::end(correlations), [](const auto& lhs, const auto& rhs) {
        return lhs.second.last_touch < rhs.second.last_touch;
      });
      if (victim != std::end(correlations))
        correlations.erase(victim);
    }

    correlations.emplace(trigger_node, correlation_entry{target_node, 1, correlation_timestamp});
    return;
  }

  hit->second.last_touch = correlation_timestamp;
  if (hit->second.target_node == target_node) {
    hit->second.confidence = std::min<uint8_t>(MAX_CORRELATION_CONF, hit->second.confidence + 1);
    return;
  }

  if (hit->second.confidence > 0) {
    hit->second.confidence--;
  } else {
    hit->second.target_node = target_node;
    hit->second.confidence = 1;
  }
}

void update_ahq(uint64_t node_addr)
{
  auto duplicate = std::find(std::begin(ahq), std::end(ahq), node_addr);
  if (duplicate != std::end(ahq))
    return;

  if (ahq.size() >= AHQ_SIZE) {
    const auto trigger_node = ahq.front();
    ahq.pop_front();
    update_correlation(trigger_node, node_addr);
  }

  ahq.push_back(node_addr);
}

bool recently_prefetched(uint64_t line_addr)
{
  return std::find(std::begin(recent_prefetches), std::end(recent_prefetches), line_addr) != std::end(recent_prefetches);
}

void remember_prefetch(uint64_t line_addr)
{
  if (recently_prefetched(line_addr))
    return;

  if (recent_prefetches.size() >= RECENT_PREFETCH_SIZE)
    recent_prefetches.pop_front();

  recent_prefetches.push_back(line_addr);
}

pc_entry& get_pc_entry(uint64_t pc)
{
  ++pc_timestamp;

  auto hit = std::find_if(std::begin(pc_table), std::end(pc_table), [pc](const auto& entry) { return entry.pc == pc; });
  if (hit != std::end(pc_table)) {
    hit->last_touch = pc_timestamp;
    return *hit;
  }

  if (pc_table.size() < PC_TABLE_SIZE) {
    pc_table.push_back({pc, 0, 0, 0, pc_timestamp});
    return pc_table.back();
  }

  auto victim = std::min_element(std::begin(pc_table), std::end(pc_table), [](const auto& lhs, const auto& rhs) { return lhs.last_touch < rhs.last_touch; });
  *victim = {pc, 0, 0, 0, pc_timestamp};
  return *victim;
}

void update_pc_delta(pc_entry& entry, uint64_t current_line)
{
  if (entry.last_line == 0) {
    entry.last_line = current_line;
    return;
  }

  const int64_t observed_delta = static_cast<int64_t>(current_line) - static_cast<int64_t>(entry.last_line);
  if (observed_delta == 0) {
    entry.last_line = current_line;
    return;
  }

  if (entry.confidence == 0) {
    entry.delta = observed_delta;
    entry.confidence = 1;
  } else if (entry.delta == observed_delta) {
    entry.confidence = std::min<uint8_t>(MAX_CORRELATION_CONF, entry.confidence + 1);
  } else {
    entry.confidence--;
    if (entry.confidence == 0) {
      entry.delta = observed_delta;
      entry.confidence = 1;
    }
  }

  entry.last_line = current_line;
}

bool try_issue_prefetch(const augur* self, uint64_t current_line, uint64_t pf_line, uint32_t metadata_in)
{
  if (pf_line == 0 || pf_line == current_line || recently_prefetched(pf_line))
    return false;

  if ((pf_line & PAGE_MASK) != (current_line & PAGE_MASK))
    return false;

  const bool fill_l1 = self->intern_->get_mshr_occupancy_ratio() < 0.5;
  if (!self->prefetch_line(champsim::address{pf_line}, fill_l1, metadata_in))
    return false;

  remember_prefetch(pf_line);
  return true;
}

void start_lookahead(augur* self, uint64_t base_line, int64_t delta)
{
  if (delta == 0)
    return;

  const auto next_line = static_cast<uint64_t>(static_cast<int64_t>(base_line) + delta);
  if ((next_line & PAGE_MASK) != (base_line & PAGE_MASK))
    return;

  self->active_lookahead = {champsim::address{next_line}, delta, PREFETCH_DEGREE};
}
} // namespace

uint32_t augur::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                         uint32_t metadata_in)
{
  (void)useful_prefetch;

  if (type != access_type::LOAD)
    return metadata_in;

  const uint64_t pc = ip.to<uint64_t>();
  const uint64_t raw_addr = addr.to<uint64_t>();
  const uint64_t node_addr = align_node(raw_addr);
  const uint64_t current_line = align_cacheline(raw_addr);

  auto& lit_state = get_lit_entry(pc);
  const bool is_recursive = lit_state.recursive_conf >= RECURSIVE_THRESHOLD;
  const bool demand_miss = (cache_hit == 0);

  if (is_recursive && demand_miss) {
    for (auto& entry : lht) {
      // ChampSim's prefetcher API does not expose the loaded pointer value, so we
      // conservatively approximate producer detection using repeated line visits.
      if (align_cacheline(entry.node_addr) == current_line)
        entry.recursive_producer = true;
    }
  }

  enqueue_lht(pc, node_addr);

  auto& pc_state = get_pc_entry(pc);

  bool issued = false;
  if (demand_miss) {
    auto candidate = correlations.find(current_line);
    if (candidate != std::end(correlations) && candidate->second.confidence >= PREFETCH_CONFIDENCE_THRESHOLD) {
      issued = try_issue_prefetch(this, current_line, align_cacheline(candidate->second.target_node), metadata_in);
    }
  }

  if (pc_state.confidence >= PREFETCH_CONFIDENCE_THRESHOLD) {
    const uint64_t pf_line = static_cast<uint64_t>(static_cast<int64_t>(current_line) + pc_state.delta);
    issued = try_issue_prefetch(this, current_line, pf_line, metadata_in) || issued;
  }

  if (pc_state.confidence >= LOOKAHEAD_CONFIDENCE_THRESHOLD) {
    start_lookahead(this, current_line, pc_state.delta);
  }

  if (demand_miss)
    update_ahq(current_line);
  update_pc_delta(pc_state, current_line);
  return metadata_in;
}

uint32_t augur::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in)
{
  (void)addr;
  (void)set;
  (void)way;
  (void)prefetch;
  (void)evicted_addr;
  return metadata_in;
}

void augur::prefetcher_cycle_operate()
{
  if (!active_lookahead.has_value())
    return;

  auto [old_pf_address, stride, degree] = active_lookahead.value();
  if (degree <= 0 || stride == 0) {
    active_lookahead.reset();
    return;
  }

  const auto old_line = align_cacheline(old_pf_address.to<uint64_t>());
  const auto next_line = static_cast<uint64_t>(static_cast<int64_t>(old_line) + stride);
  if ((next_line & PAGE_MASK) != (old_line & PAGE_MASK)) {
    active_lookahead.reset();
    return;
  }

  const bool issued = try_issue_prefetch(this, old_line, next_line, 0);
  if (!issued)
    return;

  active_lookahead = {champsim::address{next_line}, stride, degree - 1};
  if (active_lookahead->degree == 0)
    active_lookahead.reset();
}
