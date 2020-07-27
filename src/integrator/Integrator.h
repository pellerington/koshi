#pragma once

#include <koshi/math/Types.h>
#include <koshi/intersection/Intersect.h>
#include <koshi/base/Resources.h>
#include <koshi/geometry/Geometry.h>
class Scene;

struct IntegratorData
{
    virtual ~IntegratorData() = default;
};

class Transmittance;

class Integrator : public Object
{
public:
    virtual IntegratorData * pre_integrate(const Intersect * intersect, Resources& resources) { return nullptr; }

    // Perform the integration of an intersection. Eg. Direct sampling would generate light paths here.
    virtual Vec3f integrate(const Intersect * intersect, IntegratorData * data, Transmittance& transmittance,  Resources& resources) const = 0;

    // Given a point on the intersect t, how much shadowing should be applied.
    virtual Vec3f shadow(const float& t, const Intersect * intersect, IntegratorData * data, Resources& resources) const = 0;

    // Todo: move this function somewhere else. Shader::shade ???
    static Vec3f shade(const IntersectList * intersects, Resources& resources);
    static Transmittance shadow(const IntersectList * intersects, Resources& resources);
};

// TODO: Rename Transmittance to something else + find better way to store integrators data.
class Transmittance
{
public:
    Transmittance(const IntersectList * intersects, const Array<IntegratorData*>& data) 
    : intersects(intersects), data(data)
    {
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

    inline const IntersectList * get_intersects() { return intersects; }

private:
    const IntersectList * intersects;
    const Array<IntegratorData*> data;
};