#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <Math/Types.h>
#include <Math/RNG.h>
#include <base/Object.h>

#include <Util/Attribute.h>
#include <Util/Resources.h>

#define UNIFORM_SAMPLE false

struct MaterialSample
{
    Vec3f wo;
    Vec3f weight;
    float pdf;
};

struct MaterialInstance
{
    const GeometrySurface * surface;
    virtual ~MaterialInstance() = default;
};

class Material : public Object
{
public:
    virtual MaterialInstance * instance(const GeometrySurface * surface, Resources &resources) { return nullptr; }

    virtual bool sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, const float sample_reduction, Resources &resources) { return false; }
    virtual bool evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample) { return false; }
};
