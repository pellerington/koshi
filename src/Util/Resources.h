#pragma once

#include <Util/Memory.h>
#include <Math/Random.h>
#include <base/Settings.h>
class Intersector;
class Scene;

struct Resources
{
    // TODO: Maket this const
    uint thread_id;

    const Settings * settings;
    Scene * scene;
    Intersector * intersector;

    RandomService * random_service;
    Memory * memory;
};
