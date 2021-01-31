#pragma once

#include <cfloat>

#include <koshi/Vec2.h>
#include <koshi/Ray.h>
#include <koshi/Transform.h>

KOSHI_OPEN_NAMESPACE

class Camera
{
public:
    Camera(const Vec2u& resolution, const Transform& world_to_obj, const Transform& projection)
    : resolution(resolution), world_to_obj(world_to_obj), obj_to_world(world_to_obj.inverse()), projection(projection), inv_projection(projection.inverse())
    {
    }

    DEVICE_FUNCTION const Vec2u& getResolution() { return resolution; }

    DEVICE_FUNCTION Ray sample(const uint& x, const uint& y)
    {
        Vec3f ndc(2.f * ((float)x / resolution.x) - 1.f, 2.f * ((float)y / resolution.y) - 1.f, -1.f);
        
        Ray ray;
        ray.origin = Vec3f(0.f);
        ray.direction = inv_projection * ndc;
        ray *= obj_to_world;
        ray.direction.normalize();
        ray.tmin = 0.f;
        ray.tmax = FLT_MAX;
        return ray;
    }

private:
    Vec2u resolution;
    Transform world_to_obj;
    Transform obj_to_world;
    Transform projection;
    Transform inv_projection;

};

KOSHI_CLOSE_NAMESPACE
              
