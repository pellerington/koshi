#pragma once

#include <koshi/math/Types.h>
#include <koshi/intersection/Intersect.h>
#include <koshi/base/Resources.h>
#include <koshi/geometry/Geometry.h>
class Transmittance;

class Integrator : public Object
{
public:
    // Any data that might be needed for shadowing and integrating.
    virtual void * pre_integrate(const Intersect * intersect, Resources& resources) { return nullptr; }

    // Perform the integration of an intersection. Eg. Direct sampling would generate light paths here.
    virtual Vec3f integrate(const Intersect * intersect, void * data, Transmittance& transmittance,  Resources& resources) const = 0;

    // Given a point on the intersect t, how much shadowing should be applied.
    virtual Vec3f shadow(const float& t, const Intersect * intersect, void * data, Resources& resources) const = 0;

    // Todo: move this function somewhere else. Shader::shade ???
    static Vec3f shade(const IntersectList * intersects, Resources& resources);
    static Transmittance shadow(const IntersectList * intersects, Resources& resources);
};