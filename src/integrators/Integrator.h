#pragma once

#include <Math/Types.h>
#include <intersection/Intersect.h>
#include <Util/Resources.h>
#include <geometry/Geometry.h>
class Scene;

// struct IntegratorData
// {
// };

class Transmittance;

class Integrator : public Object
{
public:
    // virtual IntegratorData * pre_integrate(const Intersect * intersect) = 0;

    // Perform the integration of an intersection. Eg. Direct sampling would generate light paths here.
    virtual Vec3f integrate(const Intersect * intersect, Transmittance& transmittance, Resources &resources) const = 0;

    virtual Vec3f shadow(const float& t, const Intersect * intersect)
    {
        return (t > intersect->t) ? VEC3F_ZERO : VEC3F_ONES;
    }

    // Todo: move this function somewhere else. Shader::shade ???
    static Vec3f shade(const IntersectList * intersects, Resources &resources);
    static Transmittance shadow(const IntersectList * intersects, Resources &resources);
};

struct IntegratorList
{
    IntegratorList(IntegratorList * next = nullptr) : next(next) {}
    const Intersect * intersect;
    Integrator * integrator;
    //IntegratorData * data;
    IntegratorList * next;
};

class Transmittance
{
public:
    Transmittance(IntegratorList * integrators) : integrators(integrators) {}
    Vec3f shadow(const float& t)
    {
        Vec3f opacity = VEC3F_ONES;
        for(IntegratorList * integrator = integrators; integrator; integrator = integrator->next)
        {
            if(t > integrator->intersect->t)
                opacity *= 1.f - ((1.f - integrator->integrator->shadow(t, integrator->intersect)) * integrator->intersect->opacity);
        }
        return opacity;
    }
private:
    IntegratorList * integrators;
};