#pragma once

#include <koshi/integrator/Integrator.h>

class AbsorbtionMedium : public Integrator
{
public:
    AbsorbtionMedium(const Vec3f& density) : density(density) {}

    AbsorbtionMedium(const Vec3f& color, const float& depth)
    : density(-Vec3f::log(Vec3f::max(color, Vec3f(EPSILON_F))) / std::max(depth, EPSILON_F)) {}

    Vec3f integrate(const Intersect * intersect, IntegratorData * data, Transmittance& transmittance, Resources& resources) const
    {
        return VEC3F_ZERO;
    }

    // Given a point on the intersect t, how much shadowing should be applied.
    virtual Vec3f shadow(const float& t, const Intersect * intersect, IntegratorData * data, Resources& resources) const
    {
        if(t < intersect->t)
            return VEC3F_ONES;
        const float length = std::min(t - intersect->t, intersect->tlen);
        return Vec3f::exp(-length * density);
    }

private:
    const Vec3f density;
};