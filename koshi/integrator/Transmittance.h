#pragma once

#include <koshi/intersection/Intersect.h>
#include <koshi/intersection/IntersectCallbacks.h>
#include <koshi/base/Resources.h>

class Transmittance
{
public:
    Transmittance(const IntersectList * intersects, const Array<void*>& data) 
    : intersects(intersects), data(data)
    {
        intersect_callbacks.pre_intersection_cb = push_interiors;
        intersect_callbacks.pre_intersection_data = this;
    }

    Vec3f shadow(const float& t, Resources& resources)
    {
        Vec3f opacity = VEC3F_ONES;
        for(uint i = 0; i < intersects->size(); i++)
        {
            const Intersect * intersect = intersects->get(i);
            if(intersect->integrator && t > intersect->t)
                opacity *= intersect->integrator->shadow(t, intersect, data[i], resources);
        }
        return opacity;
    }

    enum NextInterior { NONE, PUSH, POP };
    IntersectionCallbacks * get_interiors_callback(const float& t, const NextInterior& interior = NONE, Geometry * geometry = nullptr, Integrator * integrator = nullptr)
    {
        interiors_t = t;
        next_interior = interior;
        next_interior_geometry = geometry;
        next_interior_integrator = integrator;
        return &intersect_callbacks;
    }

    static void push_interiors(IntersectList * intersects, void * data, Resources& resources)
    {
        Transmittance * transmittance = (Transmittance *)data;
        for(uint i = 0; i < transmittance->intersects->size(); i++)
        {
            const Intersect * prev_intersect = transmittance->intersects->get(i);
            if(prev_intersect->interior && transmittance->interiors_t > prev_intersect->t && transmittance->interiors_t < prev_intersect->t + prev_intersect->tlen)
            {
                if(transmittance->next_interior == POP && prev_intersect->geometry == transmittance->next_interior_geometry)
                    continue;

                Intersect * intersect = intersects->push(resources);
                intersect->geometry = prev_intersect->geometry;
                intersect->geometry_data = prev_intersect->geometry_data;
                intersect->integrator = prev_intersect->integrator;
                intersect->t = 0.f;
                intersect->tlen = FLT_MAX;
            }
        }

        if(transmittance->next_interior == PUSH && transmittance->next_interior_integrator)
        {
            Intersect * intersect = intersects->push(resources);
            intersect->geometry = transmittance->next_interior_geometry;
            // TODO: Add geometry_data
            intersect->integrator = transmittance->next_interior_integrator;
            intersect->t = 0.f;
            intersect->tlen = FLT_MAX;
        }
    }

private:
    const IntersectList * intersects;
    const Array<void*> data;

    float interiors_t;
    NextInterior next_interior;
    Geometry * next_interior_geometry;
    Integrator * next_interior_integrator;
    IntersectionCallbacks intersect_callbacks;
};