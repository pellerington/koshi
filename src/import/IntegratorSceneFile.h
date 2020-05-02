#pragma once

#include "import/SceneFile.h"

#include <integrators/IntegratorSurfaceMaterialSampler.h>
#include <integrators/IntegratorSurfaceLightSampler.h>
#include <integrators/IntegratorSurfaceMultipleImportanceSampler.h>

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
    }

    static Object * create_surface_material_integrator(AttributeAccessor& accessor, Object * parent)
    {
        return new IntegratorSurfaceMaterialSampler;
    }

    static Object * create_surface_light_integrator(AttributeAccessor& accessor, Object * parent)
    {
        return new IntegratorSurfaceLightSampler;
    }

    static Object * create_surface_mis_integrator(AttributeAccessor& accessor, Object * parent)
    {
        IntegratorSurfaceMultipleImportanceSampler * integrator = new IntegratorSurfaceMultipleImportanceSampler;
        ObjectGroup * group = accessor.get_objects("integrators");
        integrator->set_attribute("integrators", group);
        return integrator;
    }

};