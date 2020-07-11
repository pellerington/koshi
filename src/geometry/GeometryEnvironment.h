#pragma once

#include <geometry/Geometry.h>
#include <geometry/SurfaceDistant.h>
#include <intersection/Opacity.h>
#include <integrator/Integrator.h>
#include <integrator/DistantLightEvaluator.h>
#include <intersection/IntersectCallbacks.h>

class GeometryEnvironment : public Geometry
{
public:
    GeometryEnvironment(const Transform3f& transform)
    : Geometry(transform), integrator(nullptr)
    {
        intersection_cb = new IntersectionCallbacks;
        intersection_cb->post_intersection_cb = post_intersection_cb;
        intersection_cb->post_intersection_data = this;
        set_attribute("intersection_callbacks", intersection_cb);
    }

    void pre_render(Resources& resources)
    {
        integrator = new DistantLightEvaluator();
    }

    ~GeometryEnvironment()
    {
        delete integrator;
        delete intersection_cb;
    }

private:
    Integrator * integrator;
    IntersectionCallbacks * intersection_cb;

    static void post_intersection_cb(IntersectList * intersects, void * data, Resources& resources)
    {
        // We only hit if we have infinite distance.
        if(intersects->tend < FLT_MAX || intersects->ray.tmax != FLT_MAX)
            return;

        GeometryEnvironment * geometry = (GeometryEnvironment *)data;
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

        SurfaceDistant * distant = resources.memory->create<SurfaceDistant>(u, v);
        intersect->geometry_data = distant;
    
        Opacity * opacity = geometry->get_attribute<Opacity>("opacity");
        if(opacity) distant->opacity = opacity->get_opacity(distant->u, distant->v, 0.f, intersect, resources);

        intersect->integrator = geometry->integrator;
    }
};
