/*
 *    Copyright 2023 The ChampSim Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <array>
#include <map>
#include <numeric>
#include <utility>
#include <nlohmann/json.hpp>

#include "stats_printer.h"

namespace
{
constexpr std::array all_access_types{access_type::LOAD, access_type::RFO, access_type::PREFETCH, access_type::WRITE, access_type::TRANSLATION};
constexpr std::array demand_access_types{access_type::LOAD, access_type::RFO, access_type::WRITE, access_type::TRANSLATION};

template <typename CounterT, typename TypesT>
uint64_t sum_counter_for_cpu(const CounterT& counter, std::size_t cpu, const TypesT& types)
{
  return std::accumulate(std::begin(types), std::end(types), uint64_t{0},
                         [&counter, cpu](uint64_t acc, auto type) { return acc + counter.value_or(std::pair{type, cpu}, uint64_t{}); });
}
} // namespace

void to_json(nlohmann::json& j, const O3_CPU::stats_type& stats)
{
  constexpr std::array types{branch_type::BRANCH_DIRECT_JUMP, branch_type::BRANCH_INDIRECT,      branch_type::BRANCH_CONDITIONAL,
                             branch_type::BRANCH_DIRECT_CALL, branch_type::BRANCH_INDIRECT_CALL, branch_type::BRANCH_RETURN};

  auto total_mispredictions = std::ceil(
      std::accumulate(std::begin(types), std::end(types), 0LL, [btm = stats.branch_type_misses](auto acc, auto next) { return acc + btm.value_or(next, 0); }));

  std::map<std::string, std::size_t> mpki{};
  for (auto type : types) {
    mpki.emplace(branch_type_names.at(champsim::to_underlying(type)), stats.branch_type_misses.value_or(type, 0));
  }

  j = nlohmann::json{{"instructions", stats.instrs()},
                     {"cycles", stats.cycles()},
                     {"IPC", std::ceil(stats.instrs()) / std::ceil(stats.cycles())},
                     {"CPI", std::ceil(stats.cycles()) / std::ceil(stats.instrs())},
                     {"Avg ROB occupancy at mispredict", std::ceil(stats.total_rob_occupancy_at_branch_mispredict) / std::ceil(total_mispredictions)},
                     {"mispredict", mpki}};
}

void to_json(nlohmann::json& j, const CACHE::stats_type& stats)
{
  using hits_value_type = typename decltype(stats.hits)::value_type;
  using misses_value_type = typename decltype(stats.misses)::value_type;
  using mshr_merge_value_type = typename decltype(stats.mshr_merge)::value_type;
  using mshr_return_value_type = typename decltype(stats.mshr_return)::value_type;

  std::map<std::string, nlohmann::json> statsmap;
  statsmap.emplace("prefetch requested", stats.pf_requested);
  statsmap.emplace("prefetch issued", stats.pf_issued);
  statsmap.emplace("useful prefetch", stats.pf_useful);
  statsmap.emplace("useless prefetch", stats.pf_useless);

  uint64_t total_downstream_demands = stats.mshr_return.total();
  for (std::size_t cpu = 0; cpu < NUM_CPUS; ++cpu)
    total_downstream_demands -= stats.mshr_return.value_or(std::pair{access_type::PREFETCH, cpu}, mshr_return_value_type{});

  statsmap.emplace("miss latency", std::ceil(stats.total_miss_latency_cycles) / std::ceil(total_downstream_demands));
  for (const auto type : {access_type::LOAD, access_type::RFO, access_type::PREFETCH, access_type::WRITE, access_type::TRANSLATION}) {
    std::vector<hits_value_type> hits;
    std::vector<misses_value_type> misses;
    std::vector<mshr_merge_value_type> mshr_merges;

    for (std::size_t cpu = 0; cpu < NUM_CPUS; ++cpu) {
      hits.push_back(stats.hits.value_or(std::pair{type, cpu}, hits_value_type{}));
      misses.push_back(stats.misses.value_or(std::pair{type, cpu}, misses_value_type{}));
      mshr_merges.push_back(stats.mshr_merge.value_or(std::pair{type, cpu}, mshr_merge_value_type{}));
    }

    statsmap.emplace(access_type_names.at(champsim::to_underlying(type)), nlohmann::json{{"hit", hits}, {"miss", misses}, {"mshr_merge", mshr_merges}});
  }

  j = statsmap;
}

void to_json(nlohmann::json& j, const DRAM_CHANNEL::stats_type stats)
{
  j = nlohmann::json{{"RQ ROW_BUFFER_HIT", stats.RQ_ROW_BUFFER_HIT},
                     {"RQ ROW_BUFFER_MISS", stats.RQ_ROW_BUFFER_MISS},
                     {"WQ ROW_BUFFER_HIT", stats.WQ_ROW_BUFFER_HIT},
                     {"WQ ROW_BUFFER_MISS", stats.WQ_ROW_BUFFER_MISS},
                     {"AVG DBUS CONGESTED CYCLE", (std::ceil(stats.dbus_cycle_congested) / std::ceil(stats.dbus_count_congested))},
                     {"REFRESHES ISSUED", stats.refresh_cycles}};
}

namespace champsim
{
void to_json(nlohmann::json& j, const champsim::phase_stats stats)
{
  std::map<std::size_t, long long> roi_instrs_by_cpu;
  for (std::size_t cpu = 0; cpu < std::size(stats.roi_cpu_stats); ++cpu)
    roi_instrs_by_cpu.emplace(cpu, stats.roi_cpu_stats.at(cpu).instrs());

  std::map<std::string, nlohmann::json> roi_stats;
  roi_stats.emplace("cores", stats.roi_cpu_stats);
  roi_stats.emplace("DRAM", stats.roi_dram_stats);
  for (auto x : stats.roi_cache_stats) {
    roi_stats.emplace(x.name, x);
  }

  nlohmann::json evaluation = nlohmann::json::object();
  for (std::size_t cpu = 0; cpu < std::size(stats.roi_cpu_stats); ++cpu) {
    const auto& cpu_stat = stats.roi_cpu_stats.at(cpu);
    evaluation[fmt::format("cpu{}", cpu)] = nlohmann::json{{"IPC", std::ceil(cpu_stat.instrs()) / std::ceil(cpu_stat.cycles())},
                                                           {"CPI", std::ceil(cpu_stat.cycles()) / std::ceil(cpu_stat.instrs())},
                                                           {"instructions", cpu_stat.instrs()},
                                                           {"cycles", cpu_stat.cycles()}};
  }

  for (const auto& cache_stat : stats.roi_cache_stats) {
    nlohmann::json cache_eval = nlohmann::json::object();
    for (std::size_t cpu = 0; cpu < std::size(stats.roi_cpu_stats); ++cpu) {
      const auto instrs = roi_instrs_by_cpu.at(cpu);
      const auto total_access = sum_counter_for_cpu(cache_stat.hits, cpu, all_access_types) + sum_counter_for_cpu(cache_stat.misses, cpu, all_access_types);
      const auto total_miss = sum_counter_for_cpu(cache_stat.misses, cpu, all_access_types);
      const auto demand_access =
          sum_counter_for_cpu(cache_stat.hits, cpu, demand_access_types) + sum_counter_for_cpu(cache_stat.misses, cpu, demand_access_types);
      const auto demand_miss = sum_counter_for_cpu(cache_stat.misses, cpu, demand_access_types);
      const auto prefetch_access =
          cache_stat.hits.value_or(std::pair{access_type::PREFETCH, cpu}, uint64_t{}) + cache_stat.misses.value_or(std::pair{access_type::PREFETCH, cpu}, uint64_t{});

      cache_eval[fmt::format("cpu{}", cpu)] = nlohmann::json{
          {"total_miss", total_miss},
          {"demand_miss", demand_miss},
          {"total_miss_rate", total_access > 0 ? std::ceil(100 * total_miss) / std::ceil(total_access) : 0.0},
          {"demand_miss_rate", demand_access > 0 ? std::ceil(100 * demand_miss) / std::ceil(demand_access) : 0.0},
          {"MPKI", instrs > 0 ? std::ceil(std::kilo::num * demand_miss) / std::ceil(instrs) : 0.0},
          {"avg_miss_latency_cycles", demand_miss > 0 ? std::ceil(cache_stat.total_miss_latency_cycles) / std::ceil(demand_miss) : 0.0},
          {"prefetch_accuracy", cache_stat.pf_issued > 0 ? std::ceil(100 * cache_stat.pf_useful) / std::ceil(cache_stat.pf_issued) : 0.0},
          {"prefetch_coverage_proxy", demand_miss > 0 ? std::ceil(100 * cache_stat.pf_useful) / std::ceil(demand_miss) : 0.0},
          {"prefetch_pollution_ratio", cache_stat.pf_issued > 0 ? std::ceil(100 * cache_stat.pf_useless) / std::ceil(cache_stat.pf_issued) : 0.0},
          {"useful_per_kilo_instr", instrs > 0 ? std::ceil(std::kilo::num * cache_stat.pf_useful) / std::ceil(instrs) : 0.0},
          {"useless_per_kilo_instr", instrs > 0 ? std::ceil(std::kilo::num * cache_stat.pf_useless) / std::ceil(instrs) : 0.0},
          {"prefetch_traffic_ratio", demand_access > 0 ? std::ceil(100 * prefetch_access) / std::ceil(demand_access) : 0.0}};
    }
    evaluation[cache_stat.name] = cache_eval;
  }

  std::map<std::string, nlohmann::json> sim_stats;
  sim_stats.emplace("cores", stats.sim_cpu_stats);
  sim_stats.emplace("DRAM", stats.sim_dram_stats);
  for (auto x : stats.sim_cache_stats) {
    sim_stats.emplace(x.name, x);
  }

  std::map<std::string, nlohmann::json> statsmap{{"name", stats.name}, {"traces", stats.trace_names}};
  statsmap.emplace("roi", roi_stats);
  statsmap.emplace("sim", sim_stats);
  statsmap.emplace("evaluation", evaluation);
  j = statsmap;
}
} // namespace champsim

void champsim::json_printer::print(std::vector<phase_stats>& stats) { stream << nlohmann::json::array_t{std::begin(stats), std::end(stats)}; }
