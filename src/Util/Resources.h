#pragma once

#include <Util/Memory.h>
#include <Math/Random.h>
#include <base/Settings.h>
class Intersector;

struct Resources
{
    uint thread_id;
    const Settings * settings;
    Intersector * intersector;
    Memory memory;
    RandomNumberService random_number_service;
};
