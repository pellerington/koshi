#pragma once

#include <integrators/Integrator.h>

class VolumeSingleScatter : public Integrator
{
public:
    // virtual IntegratorData * pre_integrate(const Intersect * intersect) = 0;
    Vec3f integrate(const Intersect * intersect, Transmittance& transmittance, Resources &resources) const;
    Vec3f shadow(const float& t, const Intersect * intersect, Resources &resources) const;
};