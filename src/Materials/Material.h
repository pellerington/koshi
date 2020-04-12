#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <Math/Types.h>
#include <Math/RNG.h>

#include <Util/Surface.h>
#include <Util/Attribute.h>
#include <Util/Resources.h>
class Surface;

#define UNIFORM_SAMPLE false

struct MaterialSample
{
    Vec3f wo;
    Vec3f weight;
    float pdf;
    float quality = 1.f;

    enum Type { None, Diffuse, Glossy, Specular };
    Type type = Type::None;

    struct Data { virtual ~Data() = default; };
    Data * data = nullptr;

    virtual ~MaterialSample() = default;
};

struct MaterialInstance
{
    const Surface * surface;
    virtual ~MaterialInstance() = default;
};

class Material
{
public:
    enum Type
    {
        None,
        Lambert,
        BackLambert,
        GGXReflect,
        GGXRefract,
        Dielectric,
        Subsurface
    };
    virtual Type get_type() { return None; }

    virtual MaterialInstance * instance(const Surface * surface, Resources &resources) { return nullptr; }

    virtual bool sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, const float sample_reduction, Resources &resources) { return false; }
    virtual bool evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample) { return false; }

    virtual const float get_ior() { return 1.f; } // This should require the material_instance.
};
