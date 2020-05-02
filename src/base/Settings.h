#pragma once

#include <Math/Types.h>

// TODO: Make this more user customizable.
struct Settings
{
    uint num_threads = 1;
    uint max_depth = 2;
    uint depth = 32;
    float sampling_quality = 1;

    // TODO: Put these as options on the integrators themselves.
    bool sample_lights = true;
    bool sample_material = true;
};