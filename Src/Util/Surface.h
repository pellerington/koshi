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
    Vec3f normal; // TODO: add geometric normal
    float u = 0.f;
    float v = 0.f;
    Object * object = nullptr;
    //TODO: Surface should store transform
};

struct SrfSample
{
    Vec3f wo;
    Vec3f fr;
    float pdf = 0.f;
    Vec3f color = 0.f;

    enum Type {
        Material,
        Light
    };
    Type type;
};
