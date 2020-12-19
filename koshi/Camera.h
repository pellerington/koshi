#pragma once

#include <koshi/Vec2.h>
#include <koshi/Ray.h>
#include <koshi/Transform.h>

KOSHI_OPEN_NAMESPACE

class Camera
{
public:
    Camera(const Vec2u& resolution, const Transform& transform, const Transform& projection)
    : resolution(resolution), transform(transform), inv_transform(transform.inverse()), projection(projection), inv_projection(projection.inverse())
    {
    }

    DEVICE_FUNCTION const Vec2u& getResolution() { return resolution; }

    DEVICE_FUNCTION Ray sample(const uint& x, const uint& y)
    {

        // // Un-transform the pixel's NDC coordinates through the
        // // projection matrix to get the trace of the camera ray in the
        // // near plane.
        Vec3f ndc(2.f * ((float)x / resolution.x) - 1.f, 2.f * ((float)y / resolution.y) - 1.f, -1.f);
        Vec3f nearPlane = inv_projection * ndc;

        Ray ray;
        ray.origin = Vec3f(0.f);
        ray.direction = nearPlane; // normalized????

        ray *= inv_transform;

        return ray;
    }

private:
    Vec2u resolution;
    Transform transform;
    Transform inv_transform;
    Transform projection;
    Transform inv_projection;

};

KOSHI_CLOSE_NAMESPACE
              
