#pragma once

#include <geometry/Geometry.h>
#include <geometry/SurfaceDistant.h>
#include <integrator/Integrator.h>
#include <integrator/LightEvaluator.h>
#include <intersection/IntersectCallbacks.h>
#include <material/Material.h>

class GeometryEnvironment : public Geometry
{
public:
    GeometryEnvironment(const Transform3f& transform, const GeometryVisibility& visibility)
    : Geometry(transform, visibility), integrator(nullptr)
    {
        intersection_cb = new IntersectionCallbacks;
        intersection_cb->post_intersection_cb = post_intersection_cb;
        intersection_cb->post_intersection_data = this;
        set_attribute("intersection_callbacks", intersection_cb);
    }

    void pre_render(Resources& resources)
    {
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

        float theta = acosf(dir.y);
        float phi = atanf((dir.z + EPSILON_F) / (dir.x + EPSILON_F));
        const bool zd = dir.z > 0, xd = dir.x > 0;
        if(!zd) phi += PI;
        if(xd != zd) phi += PI;
        const float u = phi * INV_TWO_PI;
        const float v = theta * INV_PI;

        SurfaceDistant * distant = resources.memory->create<SurfaceDistant>(u, v, 0.f);
        intersect->geometry_data = distant;
    
        distant->material = geometry->get_attribute<Material>("material");
        distant->opacity = (distant->material) ? distant->material->opacity(distant->u, distant->v, 0.f, intersect, resources) : VEC3F_ONES;

        intersect->integrator = geometry->integrator;
    }
};
