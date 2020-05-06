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

struct MaterialSample
{
    Vec3f wo;
    Vec3f weight;
    float pdf = 0.f; // TODO: Seperate this out.
};

struct MaterialLobe /* : public Data */
{
    // TODO: Add a constructor which takes resources and surface. This way we can instansiate our stuff easier.

    const GeometrySurface * surface;
    RandomNumberGen2D rng;
    Vec3f color;
    float roughness;

    virtual bool sample(MaterialSample& sample, Resources& resources) const = 0;

    // TODO: Seperate Evaluate into PDF and Weight, so we can compose pdf's more easily.
    virtual bool evaluate(MaterialSample& sample, Resources& resources) const = 0;

    enum Type { Diffuse, Glossy, Specular };
    virtual Type type() const = 0;

    // Hemisphere : FRONT / BACK / BOTH?

    virtual ~MaterialLobe() = default;
};

class MaterialInstance
{
public:
    inline void push(MaterialLobe * lobe) { lobes.push_back(lobe); }
    inline size_t size() const { return lobes.size(); }
    inline const MaterialLobe * operator[](const size_t& i) const { return lobes[i]; }
    void evaluate(MaterialSample& sample, Resources& resources) const
    {
        for(size_t i = 0; i < lobes.size(); i++)
        {
            MaterialSample isample = sample;
            if(lobes[i]->evaluate(isample, resources))
            {
                sample.weight += isample.weight;
                sample.pdf += isample.pdf;
            }
        }
    }
private:
    // TODO: Replace these vectors with scratchpad vectors.
    std::vector<MaterialLobe*>  lobes;
};

class Material : public Object
{
public:
    virtual MaterialInstance instance(const GeometrySurface * surface, Resources &resources) = 0;
};
