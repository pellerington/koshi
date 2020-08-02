#pragma once

#include <koshi/geometry/Geometry.h>
#include <koshi/intersection/IntersectCallbacks.h>
#include <koshi/integrator/LightEvaluator.h>
#include <koshi/math/Helpers.h>

class GeometryDirectional : public Geometry
{
public:
    GeometryDirectional(const float& angle, const Transform3f& transform, const GeometryVisibility& visibility)
    : Geometry(transform, visibility), phi_max(clamp(angle, 0.f, HALF_PI - EPSILON_F)), cos_phi_max(cosf(phi_max)), area(TWO_PI * (1.f - cos_phi_max))
    {        
        intersection_cb = new IntersectionCallbacks;
        intersection_cb->post_intersection_cb = post_intersection_cb;
        intersection_cb->post_intersection_data = this;
        set_attribute("intersection_callbacks", intersection_cb);
        integrator = new LightEvaluator();
    }

    inline const float& get_phi_max() const { return phi_max; }
    inline const float& get_cos_phi_max() const { return cos_phi_max; }
    inline const float get_area() const { return std::max(area, EPSILON_F); }

    ~GeometryDirectional()
    {
        delete integrator;
        delete intersection_cb;
    }

private:
    const float phi_max, cos_phi_max, area;
    Integrator * integrator;
    IntersectionCallbacks * intersection_cb;

    static void post_intersection_cb(IntersectList * intersects, void * data, Resources& resources)
    {
        // We only hit if we have infinite distance.
        if(intersects->tend < FLT_MAX || intersects->ray.tmax != FLT_MAX)
            return;

        GeometryDirectional * geometry = (GeometryDirectional *)data;

        // Only intersect if we have an angle.
        if(geometry->phi_max < EPSILON_F)
            return;

        // TODO: Do this in an object callback on the intersector
        // Check our visibility.
        if(intersects->path->depth == 0 && geometry->get_visibility().hide_camera)
            return;

        const Vec3f dir = geometry->get_world_to_obj().multiply(intersects->ray.dir, false);
        if(dir.y < geometry->cos_phi_max)
            return;

        Intersect * intersect = intersects->push(resources);

        intersect->t = FLT_MAX;
        intersect->geometry = geometry;

        float theta = atanf((dir.z + EPSILON_F) / (dir.x + EPSILON_F));
        theta += ((dir.z < 0.f) ? PI : 0.f) + ((dir.z * dir.x < 0.f) ? PI : 0.f);
        const float u = theta * INV_TWO_PI;
        const float v = acosf(dir.y) / geometry->phi_max;

        SurfaceDistant * distant = resources.memory->create<SurfaceDistant>(u, v, 0.f);
        intersect->geometry_data = distant;
    
        distant->material = geometry->get_attribute<Material>("material");
        distant->opacity = (distant->material) ? distant->material->opacity(distant->u, distant->v, 0.f, intersect, resources) : VEC3F_ONES;

        intersect->integrator = geometry->get_attribute<Integrator>("integrator");
        if(!intersect->integrator)
            intersect->integrator = geometry->integrator;
    }
};