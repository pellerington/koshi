#pragma once

#include <vector>
#include <queue>
#include <memory>
#include "../Math/Types.h"
#include "../Util/Surface.h"
#include "../Util/Attribute.h"
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

    struct VoidData { virtual ~VoidData() = default; };
    VoidData * data = nullptr;

    virtual ~MaterialSample() = default;
};

class Material
{
public:
    // Global variables should be set in the construtor. Instanced variables should be set in the instance method (ie texture evalutation).
    virtual std::shared_ptr<Material> instance(const Surface * surface = nullptr) { return std::shared_ptr<Material>(new Material()); }

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

    virtual bool sample_material(std::vector<MaterialSample> &samples, const float sample_reduction = 1.f) { return false; }
    virtual bool evaluate_material(MaterialSample &sample) { return false; }
    virtual const float get_ior() { return 1.f; }

protected:
    const Surface * surface = nullptr;
};
