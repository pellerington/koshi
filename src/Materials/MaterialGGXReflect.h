#pragma once

#include <Materials/Material.h>
#include <Materials/Fresnel.h>
#include <Materials/GGX.h>

class MaterialGGXReflect : public Material
{
public:
    MaterialGGXReflect(const AttributeVec3f &specular_color_attribute, const AttributeFloat &roughness_attribute);
    Type get_type() { return Material::GGXReflect; }

    MaterialInstance * instance(const GeometrySurface * surface, Resources &resources);
    struct MaterialInstanceGGXReflect : public MaterialInstance
    {
        Vec3f specular_color;
        Fresnel * fresnel;
        float roughness;
        float roughness_sqr;
    };

    bool sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, const float sample_reduction, Resources &resources);
    bool evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample);

private:

    const AttributeVec3f specular_color_attribute;
    const AttributeFloat roughness_attribute;

    /* DEALING WITH FRESNEL:
        - include fresnel in the constructor.
        - if we didn't include one then default = FresnelMetallic/FrenelDielectric
        - if we include one and its nullptr then dont calculate it.

        - include a fresnel float in the Instance struct.
        - if we are computing our own frenel then add that here.
        - otherwise if its nullptr put 1.f.

        - fresnel object will need to change though.
    */
};
