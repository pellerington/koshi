#pragma once

#include <cuda_runtime.h>

#include <koshi/math/Vec2.h>
#include <koshi/geometry/Geometry.h>

KOSHI_OPEN_NAMESPACE


class GeometryQuad : public Geometry
{
public:
    GeometryQuad();

    // void setTransform(const Transform& _obj_to_world) override;

    // DEVICE_FUNCTION const Vec3f& get_du() { return du; }
    // DEVICE_FUNCTION const Vec3f& get_dv() { return dv; }
    // DEVICE_FUNCTION const Vec3f& get_normal() { return normal; }
    // DEVICE_FUNCTION const Vec2u& get_size() { return size; }

    Vec3f temp_light;

    static const float vertices[4*3];
    static const uint32_t indices[2*3];

private:

    // // Should be on the light sampler? Will be incorrect if we have instancer or scatterer.
    // Vec3f du;
    // Vec3f dv;
    // Vec3f normal;
    // Vec2u size;
};

KOSHI_CLOSE_NAMESPACE