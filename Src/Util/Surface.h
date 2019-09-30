#pragma once

#include <cfloat>
#include "../Math/Types.h"
#include "../Objects/Object.h"
class Object;

#define SAMPLES_PER_SA 64

struct Surface
{
    Surface() : object(nullptr) {}

    Surface(Object * object, const Vec3f &position, const Vec3f &normal,
            const Vec3f &wi = 0.f, const float u = 0.f, const float v = 0.f)
    : object(object), position(position), normal(normal), wi(wi), u(u), v(v),
      n_dot_wi(normal.dot(-wi)), enter(n_dot_wi >= 0.f), transform(Transform3f::normal_transform(normal))
    {}

    const Object * object;
    const Vec3f position;
    const Vec3f normal;
    const Vec3f wi;
    const float u = 0.f, v = 0.f;
    const float n_dot_wi = 0.f;
    const bool enter = true;
    const Transform3f transform;
};
