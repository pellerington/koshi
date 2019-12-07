#pragma once

#include <cfloat>
#include <unordered_set>
#include "../Math/Types.h"
#include "../Math/Transform3f.h"
#include "IorStack.h"

#define SAMPLES_PER_SA 64

struct Surface
{
    Surface() {}

    Surface(const Vec3f &position, const Vec3f &normal, const Vec3f &geometric_normal, const Vec3f &wi, const float u, const float v, const IorStack ior = IorStack())
    : position(position), normal(normal), geometric_normal(normal), wi(wi), u(u), v(v), ior(ior),
      n_dot_wi(normal.dot(-wi)), front(n_dot_wi >= 0.f), transform(Transform3f::basis_transform(normal)),
      front_position(position + geometric_normal * EPSILON_F), back_position(position + geometric_normal * -EPSILON_F) {}

    Surface(const Vec3f &wi) : wi(wi), distant(true) {}

    const Vec3f position;
    const Vec3f normal;
    const Vec3f geometric_normal;
    const Vec3f wi;
    const float u = 0.f, v = 0.f;
    const IorStack ior;
    const float n_dot_wi = 0.f;
    const bool front = true;
    const Transform3f transform;
    const Vec3f front_position;
    const Vec3f back_position;
    const bool distant = false;

};
