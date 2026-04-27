#include "augur_memory.h"
#include <unordered_map>

static std::unordered_map<uint64_t, uint64_t> memory_values;

uint64_t augur_get_value(uint64_t addr)
{
    auto it = memory_values.find(addr);
    if (it == memory_values.end())
        return 0;

    return it->second;
}

void augur_set_value(uint64_t addr, uint64_t value)
{
    memory_values[addr] = value;
}