#pragma once

#include <cfloat>
#include <unordered_set>
#include <Math/Types.h>
#include <Math/Transform3f.h>

#define SAMPLES_PER_SA 64

struct GeometrySurface /* : public GeometryData */
{
    void set(const Vec3f& _position, const Vec3f& _normal, const float& _u, const float& _v, const Vec3f& _wi = VEC3F_ZERO)
    {
        position = _position;
        wi = _wi;
        set_normal(_normal);
        geometric_normal = _normal;
        u = _u; v = _v;
        front_position = position + geometric_normal *  EPSILON_F;
        back_position  = position + geometric_normal * -EPSILON_F;
    }

    void set_normal(const Vec3f& _normal)
    {
        normal = _normal;
        n_dot_wi = normal.dot(-wi);
        facing = (n_dot_wi >= 0.f);
        transform = Transform3f::basis_transform(normal);
    }

    Vec3f position;
        
    Vec3f front_position;
    Vec3f back_position;

    Vec3f normal;
    Vec3f geometric_normal;
    Transform3f transform;
    bool facing;

    float u, v;

    Vec3f wi;
    float n_dot_wi;
};
