#pragma once

#include "import/SceneFile.h"

#include <integrators/SurfaceMaterialSampler.h>
#include <integrators/SurfaceLightSampler.h>
#include <integrators/SurfaceMultipleImportanceSampler.h>
#include <integrators/VolumeSingleScatter.h>

struct IntegratorSceneFile
{

    static void add_types(Types& types)
    {
        // Surface Material Integrator
        Type surface_material_integrator("surface_material_integrator");
        surface_material_integrator.create_object_cb = create_surface_material_integrator;
        types.add(surface_material_integrator);

        // Surface Light Integrator
        Type surface_light_integrator("surface_light_integrator");
        surface_light_integrator.create_object_cb = create_surface_light_integrator;
        types.add(surface_light_integrator);

        // Surface Light Integrator
        Type surface_mis_integrator("surface_mis_integrator");
        surface_mis_integrator.create_object_cb = create_surface_mis_integrator;
        surface_mis_integrator.reserved_attributes.push_back("integrators");
        types.add(surface_mis_integrator);

        // Volume Single Scatter Integrator
        Type volume_single_scatter_integrator("volume_single_scatter_integrator");
        volume_single_scatter_integrator.create_object_cb = create_volume_single_scatter_integrator;
        volume_single_scatter_integrator.reserved_attributes.push_back("integrators");
        types.add(volume_single_scatter_integrator);

    }

    static Object * create_surface_material_integrator(AttributeAccessor& accessor, Object * parent)
    {
        return new SurfaceMaterialSampler;
    }

    static Object * create_surface_light_integrator(AttributeAccessor& accessor, Object * parent)
    {
        return new SurfaceLightSampler;
    }

    static Object * create_surface_mis_integrator(AttributeAccessor& accessor, Object * parent)
    {
        SurfaceMultipleImportanceSampler * integrator = new SurfaceMultipleImportanceSampler;
        ObjectGroup * group = accessor.get_objects("integrators");
        integrator->set_attribute("integrators", group);
        return integrator;
    }

    static Object * create_volume_single_scatter_integrator(AttributeAccessor& accessor, Object * parent)
    {
        return new VolumeSingleScatter;
    }

};