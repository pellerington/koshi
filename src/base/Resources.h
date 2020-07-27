#pragma once

#include <koshi/base/Memory.h>
#include <koshi/base/Settings.h>
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
