#include "augur_prefetcher.h"

// temporary empty implementation

uint32_t augur_prefetcher::prefetcher_cache_operate(
    champsim::address addr,
    champsim::address ip,
    uint8_t cache_hit,
    bool useful_prefetch,
    access_type type,
    uint32_t metadata_in)
{
    return metadata_in;
}

uint32_t augur_prefetcher::prefetcher_cache_fill(
    champsim::address addr,
    long set,
    long way,
    uint8_t prefetch,
    champsim::address evicted_addr,
    uint32_t metadata_in)
{
    return metadata_in;
}