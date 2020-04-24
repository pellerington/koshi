#pragma once

#include <Util/Memory.h>
#include <Math/RNG.h>
#include <intersection/Intersector.h>
#include <Scene/Settings.h>

struct Resources
{
    const Settings * settings;
    uint thread_id;
    Memory memory;
    RNG rng;
    Intersector * intersector;
};
