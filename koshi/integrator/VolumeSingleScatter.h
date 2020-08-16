#pragma once

#include <koshi/integrator/Integrator.h>

class VolumeSingleScatter : public Integrator
{
public:
    // virtual void * pre_integrate(const Intersect * intersect) = 0;
    Vec3f integrate(const Intersect * intersect, void * data, Transmittance& transmittance, Resources& resources) const;
    Vec3f shadow(const float& t, const Intersect * intersect, void * data, Resources& resources) const;
};