#pragma once

#include "SceneFile.h"

#include <koshi/integrator/SurfaceSampler.h>
#include <koshi/integrator/VolumeSingleScatter.h>

struct IntegratorSceneFile
{

    static void add_types(Types& types)
    {
        // Surface Integrator
        Type surface_integrator("surface_integrator");
        surface_integrator.create_object_cb = create_surface_integrator;
        surface_integrator.reserved_attributes.push_back("integrators");
        types.add(surface_integrator);

        // Volume Single Scatter Integrator
        Type volume_single_scatter_integrator("volume_single_scatter_integrator");
        volume_single_scatter_integrator.create_object_cb = create_volume_single_scatter_integrator;
        volume_single_scatter_integrator.reserved_attributes.push_back("integrators");
        types.add(volume_single_scatter_integrator);
    }

    static Object * create_surface_integrator(AttributeAccessor& accessor, Object * parent)
    {
        return new SurfaceSampler;
    }

    static Object * create_volume_single_scatter_integrator(AttributeAccessor& accessor, Object * parent)
    {
        return new VolumeSingleScatter;
    }

};