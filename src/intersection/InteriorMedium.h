#pragma once

#include <intersection/Intersect.h>
// TODO: Do IorStack on the path data. We shouldn't bother handling it with opacity ect.
// OR Add IOR to intersects and access them from the interiors.

// TODO: Move this file somewhere else, should not be part of the core intersects.

class InteriorMedium
{
public:
    InteriorMedium(const float& t, const IntersectList * prev_intersects)
    : t(t), prev_intersects(prev_intersects), change(NONE), geometry(nullptr), integrator(nullptr) {}

    InteriorMedium push(Geometry * geometry, Integrator * integrator)
    {
        InteriorMedium interiors = *this;
        if(integrator)
        {
            interiors.change = PUSH;
            interiors.geometry = geometry;
            interiors.integrator = integrator;
        }
        return interiors;
    }

    InteriorMedium pop(Geometry * geometry)
    {
        InteriorMedium interiors = *this;
        interiors.change = POP;
        interiors.geometry = geometry;
        return interiors;
    }

    static void pre_intersect_callback(IntersectList * intersects, Resources& resources, void * data)
    {
        InteriorMedium * interiors = (InteriorMedium*)data;

        for(const Intersect * prev_intersect = interiors->prev_intersects->get(0); 
            prev_intersect; prev_intersect = prev_intersect->next)
        {
            if(prev_intersect->t_len > 0.f 
            && prev_intersect->t <= interiors->t 
            && interiors->t <= prev_intersect->t + prev_intersect->t_len)
            {
                if(interiors->change != POP || prev_intersect->geometry != interiors->geometry)
                {
                    Intersect * intersect = intersects->push(resources);
                    intersect->opacity = prev_intersect->opacity;
                    intersect->geometry = prev_intersect->geometry;
                    intersect->surface = prev_intersect->surface;
                    intersect->integrator = prev_intersect->integrator;

                    intersect->t = 0.f;
                    intersect->t_len = FLT_MAX;
                }
            }
        }

        if(interiors->change == PUSH && interiors->integrator)
        {
            Intersect * intersect = intersects->push(resources);
            intersect->geometry = interiors->geometry;
            intersect->integrator = interiors->integrator;
            intersect->t = 0.f;
            intersect->t_len = FLT_MAX;
        }
    }

private:

    float t;
    const IntersectList * prev_intersects;

    enum Change { NONE, PUSH, POP } change;
    Geometry * geometry;
    Integrator * integrator;
};