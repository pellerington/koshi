#pragma once

#include <Util/Memory.h>
#include <Math/Random.h>
#include <intersection/Intersector.h>
#include <base/Settings.h>

struct Resources
{
    uint thread_id;
    const Settings * settings;
    Intersector * intersector;
    Memory memory;
    RandomNumberService random_number_service;
};
