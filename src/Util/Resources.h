#pragma once

#include <Util/Memory.h>
#include <base/Settings.h>
class Intersector;
class Scene;
class RandomService;

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
