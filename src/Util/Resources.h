#pragma once

#include "Memory.h"
#include "../Math/RNG.h"

struct Resources
{
    uint thread_id;
    Memory memory;
    RNG rng;
};
