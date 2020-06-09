#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <Math/Types.h>
#include <Math/Random.h>
#include <base/Object.h>

#include <Util/Attribute.h>
#include <Util/Resources.h>
#include <Util/Array.h>
#include <geometry/Surface.h>
#include <intersection/Interiors.h>

#define UNIFORM_SAMPLE false

// TODO: Remove max lobes?
#define MAX_LOBES 16

class Integrator;

struct MaterialSample
{
    Vec3f wo;
    Vec3f weight;
    float pdf;
};

struct MaterialLobe
{
    // TODO: Add a constructor which takes resources and surface. This way we can instansiate our stuff easier.

    const Surface * surface;
    Random2D rng;

    // TODO: Make these virtual functions?
    Vec3f color;
    float roughness;

    // TODO: Make this a virtual function, so we can return an integrator/geometry data, AND save memory->
    Integrator * interior = nullptr;

    virtual bool sample(MaterialSample& sample, Resources& resources) const = 0;

    virtual Vec3f weight(const Vec3f& wo, Resources& resources) const = 0;
    virtual float pdf(const Vec3f& wo, Resources& resources) const = 0;

    enum Type { Diffuse, Glossy, Specular };
    virtual Type type() const = 0;

    // Hemisphere : FRONT / BACK / BOTH?

    virtual ~MaterialLobe() = default;
};

// TODO: Remove this typdef and rename everything as lobes instead of material_instance
typedef Array<MaterialLobe*> MaterialInstance;

class Material : public Object
{
public:
    virtual MaterialInstance instance(const Surface * surface, Resources &resources) = 0;

    // inline virtual Vec3f emission() const { return VEC3_ZERO; }

    // inline virtual Opacity???
};
