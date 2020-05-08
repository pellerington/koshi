#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <Math/Types.h>
#include <Math/Random.h>
#include <base/Object.h>

#include <Util/Attribute.h>
#include <Util/Resources.h>

#define UNIFORM_SAMPLE false

// TODO: Max lobes could be templatable.
#define MAX_LOBES 16

struct MaterialSample
{
    Vec3f wo;
    Vec3f weight;
    float pdf;
};

struct MaterialLobe /* : public Data */
{
    // TODO: Add a constructor which takes resources and surface. This way we can instansiate our stuff easier.

    const GeometrySurface * surface;
    RandomNumberGen2D rng;
    Vec3f color;
    float roughness;

    virtual bool sample(MaterialSample& sample, Resources& resources) const = 0;

    virtual Vec3f weight(const Vec3f& wo, Resources& resources) const = 0;
    virtual float pdf(const Vec3f& wo, Resources& resources) const = 0;

    enum Type { Diffuse, Glossy, Specular };
    virtual Type type() const = 0;

    // Hemisphere : FRONT / BACK / BOTH?

    virtual ~MaterialLobe() = default;
};

class MaterialInstance
{
public:
    inline void push(MaterialLobe * lobe) { lobes[num_lobes++] = lobe; }
    inline size_t size() const { return num_lobes; }
    inline const MaterialLobe * operator[](const size_t& i) const { return lobes[i]; }
    inline Vec3f weight(const Vec3f& wo, Resources& resources) const
    {
        Vec3f weight;
        for(size_t i = 0; i < num_lobes; i++)
            weight += lobes[i]->weight(wo, resources);
        return weight;
    }
private:
    uint num_lobes = 0;
    MaterialLobe* lobes[MAX_LOBES];
};

class Material : public Object
{
public:
    virtual MaterialInstance instance(const GeometrySurface * surface, Resources &resources) = 0;
};
