#pragma once

#include <geometry/Geometry.h>
#include <geometry/GeometryDistant.h>
#include <intersection/Opacity.h>
#include <integrators/Integrator.h>
#include <integrators/DistantLightEvaluator.h>

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
        delete_integrator = false;
        if(!integrator)
        {
            integrator = new DistantLightEvaluator();
            delete_integrator = true;
        }
    }

    ~GeometryEnvironment()
    {
        if(delete_integrator)
            delete integrator;
        delete intersection_cb;
    }

private:
    bool delete_integrator;
    Integrator * integrator;
    IntersectionCallbacks * intersection_cb;

    static void null_intersection_cb(IntersectList * intersects, Geometry * geometry, Resources& resources)
    {
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

        GeometryDistant * geometry_distant = resources.memory.create<GeometryDistant>(u, v, intersect->ray.dir.normalized());
        intersect->geometry_data = geometry_distant;
    
        Opacity * opacity = geometry->get_attribute<Opacity>("opacity");
        if(opacity) geometry_distant->set_opacity(opacity->get_opacity(geometry_distant->u, geometry_distant->v, 0.f, intersect, resources));

        // TODO: Replace this with a specific distant integrator.
        intersect->integrator = environment_geometry->integrator;
    }
};
