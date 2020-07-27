#pragma once

#include <koshi/geometry/Geometry.h>
#include <koshi/geometry/SurfaceDistant.h>
#include <koshi/integrator/Integrator.h>
#include <koshi/integrator/LightEvaluator.h>
#include <koshi/intersection/IntersectCallbacks.h>
#include <koshi/material/Material.h>

class GeometryEnvironment : public Geometry
{
public:
    GeometryEnvironment(const Transform3f& transform, const GeometryVisibility& visibility)
    : Geometry(transform, visibility)
    {
        intersection_cb = new IntersectionCallbacks;
        intersection_cb->post_intersection_cb = post_intersection_cb;
        intersection_cb->post_intersection_data = this;
        set_attribute("intersection_callbacks", intersection_cb);
        integrator = new LightEvaluator();
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

        // TODO: Do this in an object callback on the intersector
        // Check our visibility.
        if(intersects->path->depth == 0 && geometry->get_visibility().hide_camera)
            return;

        Intersect * intersect = intersects->push(resources);

        intersect->t = FLT_MAX;
        intersect->geometry = geometry;

        const Vec3f dir = geometry->get_world_to_obj().multiply(intersect->ray.dir, false);

        float theta = atanf((dir.z + EPSILON_F) / (dir.x + EPSILON_F));
        theta += ((dir.z < 0.f) ? PI : 0.f) + ((dir.z * dir.x < 0.f) ? PI : 0.f);
        const float u = theta * INV_TWO_PI;
        const float v = acosf(dir.y) * INV_PI;

        SurfaceDistant * distant = resources.memory->create<SurfaceDistant>(u, v, 0.f);
        intersect->geometry_data = distant;
    
        distant->material = geometry->get_attribute<Material>("material");
        distant->opacity = (distant->material) ? distant->material->opacity(distant->u, distant->v, 0.f, intersect, resources) : VEC3F_ONES;

        intersect->integrator = geometry->get_attribute<Integrator>("integrator");
        if(!intersect->integrator)
            intersect->integrator = geometry->integrator;
    }
};
