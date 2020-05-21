#pragma once

#include <geometry/Geometry.h>
#include <intersection/Opacity.h>
#include <integrators/Integrator.h>

class GeometryEnvironment : public Geometry
{
public:
    GeometryEnvironment(const Transform3f& transform)
    : Geometry(transform), integrator(nullptr)
    {
        intersection_cb = new IntersectionCallbacks;
        intersection_cb->null_intersection_cb = null_intersection_cb;
        set_attribute("intersection_callbacks", intersection_cb);
    }

    void pre_render(Scene * scene)
    {
        integrator = get_attribute<Integrator>("integrator");
        if(!integrator)
            integrator = dynamic_cast<Integrator*>(scene->get_object("default_integrator"));
    }

    ~GeometryEnvironment()
    {
        delete intersection_cb;
    }

private:
    Integrator * integrator;
    IntersectionCallbacks * intersection_cb;

    static void null_intersection_cb(IntersectList * intersects, Geometry * geometry, Resources& resources)
    {
        // In the future add a IntersectList here and just append one

        // We only hit if we have infinite distance.
        if(intersects->ray.tmax != FLT_MAX)
            return;

        GeometryEnvironment * environment_geometry = (GeometryEnvironment *)geometry;
        Intersect * intersect = intersects->push(resources);

        intersect->t = FLT_MAX;
        intersect->geometry = geometry;

        const Vec3f dir = geometry->get_world_to_obj().multiply(intersect->ray.dir, false);

        float theta = acosf(dir.y);
        float phi = atanf((dir.z + EPSILON_F) / (dir.x + EPSILON_F));
        const bool zd = dir.z > 0, xd = dir.x > 0;
        if(!zd) phi += PI;
        if(xd != zd) phi += PI;

        const float u = phi * INV_TWO_PI;
        const float v = theta * INV_PI;

        intersect->surface.set(intersect->ray.dir * FLT_MAX, -intersect->ray.dir.normalized(), u, v, intersect->ray.dir.normalized());
    
        Opacity * opacity = geometry->get_attribute<Opacity>("opacity");
        if(opacity)
        {
            intersect->opacity = opacity->get_opacity(intersect, resources);
            if(!intersect->opacity)
                intersects->pop();
        }

        intersect->integrator = environment_geometry->integrator;
    }
};
