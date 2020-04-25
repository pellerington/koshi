#pragma once

#include <Math/Types.h>
#include <intersection/Intersect.h>
#include <Util/Resources.h>
#include <geometry/Geometry.h>
class Scene;

class Integrator : public Object
{
public:
    // virtual IntegratorInstance * pre_integrate(/* blah */) {}

    // Perform the integration of an intersection. Eg. Direct sampling would generate light paths here.
    virtual Vec3f integrate(const Intersect& intersect/*, Transmittance& transmittance*/, Resources &resources) const = 0;

    // Todo: move this function somewhere else. Shader::shade ???
    static Vec3f shade(const IntersectList& intersects,  Resources &resources)
    {
        if(intersects.empty())
            return VEC3F_ZERO;

        Vec3f color = VEC3F_ZERO;

        // Todo: First pass of integrators alowing them to add something to a transmittance context
        // Transmittancec = shadow(intersects, resources) ???? Means we have to redo get_attribute integrator

        for(size_t i = 0; i < intersects.size(); i++)
        {
            const Intersect& intersect = intersects[i];
            Geometry * geometry = intersect.geometry;

            Integrator * integrator = geometry->get_attribute<Integrator>("integrator");
            if(integrator)
                color += integrator->integrate(intersect /*, transmittance */, resources);
        }

        return  color;
    }

    // static Transmitance? shadow(const IntersectList& intersects, Resources &resources)
    //
};
