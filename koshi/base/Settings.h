#pragma once

#include <koshi/math/Types.h>

// TODO: Make this more user customizable.
struct Settings
{
    uint num_threads = 1;
    uint max_depth = 2;
    uint depth = 32;
    float sampling_quality = 1;
    uint max_samples_per_pixel = 1;
    uint min_samples_per_pixel = 1;
};