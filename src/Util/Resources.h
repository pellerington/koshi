#pragma once

#include <Util/Memory.h>
#include <Math/RNG.h>
#include <intersection/Intersector.h>
#include <base/Settings.h>

struct Resources
{
    uint thread_id;
    const Settings * settings;
    Intersector * intersector;
    Memory memory;
    RNG rng;
};
