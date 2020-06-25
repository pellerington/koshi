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

class Integrator;

struct MaterialSample
{
    Vec3f wo;
    Vec3f weight;
    float pdf;
};

struct MaterialLobe
{
    Random2D rng;

    // TODO: Add a contructor for this stuff.
    const Surface * surface;
    Vec3f wi;
    Vec3f normal;
    Transform3f transform;

    // TODO: Make these virtual functions?
    Vec3f color;
    float roughness;    

    virtual bool sample(MaterialSample& sample, Resources& resources) const = 0;
    virtual Vec3f weight(const Vec3f& wo, Resources& resources) const = 0;
    virtual float pdf(const Vec3f& wo, Resources& resources) const = 0;

    // TODO: Make this a virtual function, so we can return an integrator/geometry data, AND save memory->
    Integrator * interior = nullptr;

    enum ScatterType { DIFFUSE, GLOSSY, SPECULAR };
    virtual ScatterType get_scatter_type() const = 0;

    enum Hemisphere { FRONT, BACK, SPHERE };
    virtual Hemisphere get_hemisphere() const = 0;
};

// TODO: Remove this typdef and rename everything as lobes instead of material_instance
typedef Array<MaterialLobe*> MaterialInstance;

class Material : public Object
{
public:
    virtual MaterialInstance instance(const Surface * surface, const Intersect * intersect, Resources &resources) = 0;

    // inline virtual Vec3f emission() const { return VEC3_ZERO; }

    // inline virtual Opacity???
};
