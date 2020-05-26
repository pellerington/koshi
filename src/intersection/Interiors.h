#pragma once

#include <intersection/Intersect.h>
// TODO: Do IorStack on the path data. We shouldn't bother handling it with opacity ect.
// OR Add IOR to intersects and access them from the interiors.

// TODO: Move this file somewhere else, should not be part of the core intersects.

class Interiors
{
public:
    Interiors(const float& t, const IntersectList * prev_intersects)
    : t(t), prev_intersects(prev_intersects), change(NONE), geometry(nullptr), integrator(nullptr) {}

    Interiors push(Geometry * geometry, Integrator * integrator)
    {
        Interiors interiors = *this;
        if(integrator)
        {
            interiors.change = PUSH;
            interiors.geometry = geometry;
            interiors.integrator = integrator;
        }
        return interiors;
    }

    Interiors pop(Geometry * geometry)
    {
        Interiors interiors = *this;
        interiors.change = POP;
        interiors.geometry = geometry;
        return interiors;
    }

    static void pre_intersect_callback(IntersectList * intersects, Resources& resources, void * data)
    {
        Interiors * interiors = (Interiors*)data;

        for(const Intersect * prev_intersect = interiors->prev_intersects->get(0); 
            prev_intersect; prev_intersect = prev_intersect->next)
        {
            if(prev_intersect->interior 
            && prev_intersect->t_len > 0.f
            && prev_intersect->t <= interiors->t 
            && interiors->t <= prev_intersect->t + prev_intersect->t_len)
            {
                if(interiors->change == POP && prev_intersect->geometry == interiors->geometry)
                    continue;

                Intersect * intersect = intersects->push(resources);
                intersect->geometry = prev_intersect->geometry;
                // TODO: In the future we might need a copy with the new intersect information for geom_data.
                intersect->geometry_data = prev_intersect->geometry_data;
                intersect->integrator = prev_intersect->integrator;
                intersect->t = 0.f;
                intersect->t_len = FLT_MAX;
                intersect->interior = true;
            }
        }

        if(interiors->change == PUSH && interiors->integrator)
        {
            Intersect * intersect = intersects->push(resources);
            intersect->geometry = interiors->geometry;
            intersect->integrator = interiors->integrator;
            intersect->t = 0.f;
            intersect->t_len = FLT_MAX;
            intersect->interior = true;
        }
    }

private:

    float t;
    const IntersectList * prev_intersects;

    enum Change { NONE, PUSH, POP } change;
    Geometry * geometry;
    Integrator * integrator;
};