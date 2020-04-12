#pragma once

// TODO MOVE THIS INTO AN INTERSECT FOLDER

#include <cfloat>
#include <unordered_set>
#include <Math/Types.h>
#include <Math/Transform3f.h>
#include <Util/IorStack.h>

#define SAMPLES_PER_SA 64

struct Surface
{
    Surface() {}

    Surface(const Vec3f &wi, const float &curr_ior = 1.f, const float &prev_ior = 1.f)
    : wi(wi), curr_ior(curr_ior), prev_ior(prev_ior) {}

    void set_hit(const Vec3f &_position, const Vec3f &_normal, const Vec3f &_geometric_normal, const float _u, const float _v)
    {
        hit = true;

        position = _position;
        normal = _normal;
        geometric_normal = _geometric_normal;
        u = _u; v = _v;

        n_dot_wi = normal.dot(-wi);
        front = (n_dot_wi >= 0.f);

        transform = Transform3f::basis_transform(normal);

        front_position = position + geometric_normal *  EPSILON_F;
        back_position  = position + geometric_normal * -EPSILON_F;
    }

    bool hit = false;

    Vec3f position;
    Vec3f normal;
    Vec3f geometric_normal;
    Vec3f wi;
    float u, v;

    float curr_ior;
    float prev_ior;

    float n_dot_wi;
    bool front;
    Transform3f transform;

    Vec3f front_position;
    Vec3f back_position;
};
