#pragma once

#include <cfloat>
#include "../Math/Types.h"
#include "../Objects/Object.h"
class Object;

#define SAMPLES_PER_SA 64

struct Surface
{
    Vec3f position;
    Vec3f wi;
    Vec3f normal; // + Geometric normal?
    float u = 0.f;
    float v = 0.f;
    bool enter = true;
    Object * object = nullptr;
    //TODO: Surface should store a normal transform?
};
