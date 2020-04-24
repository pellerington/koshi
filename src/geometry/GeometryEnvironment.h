#pragma once

#include <geometry/Geometry.h>

class GeometryEnvironment : public Geometry
{
public:
    GeometryEnvironment()
    : Geometry(Transform3f())
    {
        intersection_cb = new IntersectionCallbacks;
        intersection_cb->null_intersection_cb = null_intersection_cb;
        set_attribute("intersection_callbacks", intersection_cb);
    }

    ~GeometryEnvironment()
    {
        delete intersection_cb;
    }

private:
    IntersectionCallbacks * intersection_cb;

    static void null_intersection_cb(IntersectList& intersects, Geometry * geometry)
    {
        // In the future add a IntersectList here and just append one

        // We only hit if we have infinite distance.
        if(intersects.ray.tmax != FLT_MAX)
            return;

        Intersect& intersect = intersects.push();

        intersect.t = FLT_MAX;
        intersect.geometry = geometry;

        float theta = acosf(intersect.ray.dir.y);
        float phi = atanf((intersect.ray.dir.z + EPSILON_F) / (intersect.ray.dir.x + EPSILON_F));
        const bool zd = intersect.ray.dir.z > 0, xd = intersect.ray.dir.x > 0;
        if(!zd) phi += PI;
        if(xd != zd) phi += PI;

        const float u = phi * INV_TWO_PI;
        const float v = theta * INV_PI;

        intersect.surface.set(intersect.ray.dir * FLT_MAX, -intersect.ray.dir.normalized(), u, v, intersect.ray.dir.normalized());
    }
};
