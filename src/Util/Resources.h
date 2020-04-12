#pragma once

#include <Util/Memory.h>
#include <Math/RNG.h>

struct Resources
{
    uint thread_id;
    Memory memory;
    RNG rng;
};
