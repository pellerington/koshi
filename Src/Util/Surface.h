#pragma once

#include <cfloat>
#include "../Math/Types.h"
#include "../Objects/Object.h"
class Object;

#define SAMPLES_PER_SA 64

struct Surface
{
    Surface() : object(nullptr) {}

    Surface(Object * object, const Vec3f &position, const Vec3f &normal, const Vec3f &wi, const float u, const float v)
    : object(object), position(position), normal(normal), wi(wi), u(u), v(v), n_dot_wi(normal.dot(-wi)),
      enter(n_dot_wi >= 0.f), transform(Transform3f::normal_transform(normal)),
      front_position(position + normal * EPSILON_F),
      back_position(position + normal * -EPSILON_F)
    {}

    const Object * object;
    const Vec3f position;
    const Vec3f normal;
    const Vec3f wi;
    const float u = 0.f, v = 0.f;
    const float n_dot_wi = 0.f;
    const bool enter = true;
    const Transform3f transform;
    const Vec3f front_position;
    const Vec3f back_position;
};
