#pragma once

#include <Util/Memory.h>
#include <Math/RNG.h>
#include <intersection/Intersector.h>

struct Resources
{
    uint thread_id;
    Memory memory;
    RNG rng;
    Intersector * intersector;
};
