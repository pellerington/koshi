#pragma once

#include <koshi/light/LightSampler.h>
#include <koshi/geometry/GeometrySphere.h>

class LightSamplerSphere : public LightSampler
{
public:
    LightSamplerSphere(GeometrySphere * geometry);

    void pre_render(Resources& resources);

    struct LightSamplerDataSphere : public LightSamplerData
    {
        enum EvalType { SPHERE_SOLID_ANGLE, SPHERE_AREA, ELLIPSOID_SOLID_ANGLE, ELLIPSOID_AREA };
        EvalType eval_type;
    };
    const LightSamplerData * pre_integrate(const Surface * surface, Resources& resources) const;
    bool sample(LightSample& sample, const LightSamplerData * data, Resources& resources) const;
    float evaluate(const Intersect * intersect, const LightSamplerData * data, Resources& resources) const;

    LightType get_light_type() const { return (radius > EPSILON_F) ? LightType::AREA : LightType::POINT; }

    struct LightSamplerDataSphereSolidAngle : public LightSamplerDataSphere
    {
        Random<2> rng;
        Vec3f cd;
        float cd_len_sqr;
        float cd_len;
        float sin_max_sqr;
        float cos_max;
        Transform3f basis;
    };
    bool sample_sphere_sa(LightSample& sample, const LightSamplerDataSphereSolidAngle * data, Resources& resources) const;
    float evaluate_sphere_sa(const Intersect * intersect, const LightSamplerDataSphereSolidAngle * data, Resources& resources) const;

    struct LightSamplerDataSphereArea : public LightSamplerDataSphere
    {
    };
    bool sample_sphere_area(LightSample& sample, const LightSamplerDataSphereArea * data, Resources& resources) const;
    float evaluate_sphere_area(const Intersect * intersect, const LightSamplerDataSphereArea * data, Resources& resources) const;

    // TODO: Implement solid angle for ellipses.

private:
    GeometrySphere * geometry;
    Material * material;
    
    Vec3f center;
    Vec3f radius;
    Vec3f radius_sqr;
    float area;
    bool ellipsoid;
};
