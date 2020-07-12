#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <math/Types.h>
#include <math/Random.h>
#include <base/Object.h>

#include <Util/Resources.h>
#include <Util/Array.h>
#include <geometry/Surface.h>
#include <intersection/Interiors.h>
#include <texture/Texture.h>

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
    Random<2> rng;

    // TODO: Add a contructor for this stuff.
    const Surface * surface;
    Vec3f wi;
    Vec3f normal;
    Transform3f transform;

    // TODO: Make these virtual functions instead.
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

// TODO: Remove this typdef and rename everything as lobes instead of lobes
typedef Array<MaterialLobe*> MaterialLobes;

// TODO: Generalize the material with a contructor for normals and opacity.
class Material : public Object
{
public:

    Material(const Texture * normal_texture = nullptr, const Texture * opacity_texture = nullptr)
    : normal_texture(normal_texture), opacity_texture(opacity_texture)
    {
    }

    virtual MaterialLobes instance(const Surface * surface, const Intersect * intersect, Resources& resources)
    {
        return MaterialLobes(resources.memory, 1u);
    }

    virtual Vec3f emission(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) 
    { 
        return VEC3F_ZERO; 
    }

    virtual Vec3f opacity(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources)
    {
        return (opacity_texture) ? opacity_texture->evaluate(u, v, w, intersect, resources) : VEC3F_ONES;
    }

protected:
    const Texture * normal_texture;
    const Texture * opacity_texture;
};
