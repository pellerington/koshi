#include <light/LightSamplerSphere.h>

LightSamplerSphere::LightSamplerSphere(GeometrySphere * geometry)
: geometry(geometry)
{
    // TODO: No Guarentee we will get this, do it in pre_render() or combine into Light ??
    light = geometry->get_attribute<Light>("light");

    const Transform3f& obj_to_world = geometry->get_obj_to_world();

    center = obj_to_world * VEC3F_ZERO;

    radius.x = obj_to_world.multiply(Vec3f(SPHERE_RADIUS, 0.f, 0.f), false).length();
    radius.y = obj_to_world.multiply(Vec3f(0.f, SPHERE_RADIUS, 0.f), false).length();
    radius.z = obj_to_world.multiply(Vec3f(0.f, 0.f, SPHERE_RADIUS), false).length();

    radius_sqr = radius*radius;

    const float p = 8.f / 5.f;
    area = FOUR_PI * powf((powf(radius.x*radius.y, p) + powf(radius.x*radius.z, p) + powf(radius.y*radius.z, p)) / 3.f, 1.f / p);

    ellipsoid = fabs(radius.x-radius.y) > 0.01f || fabs(radius.x-radius.z) > 0.01f || fabs(radius.y-radius.z) > 0.01f;
}

const LightSamplerData * LightSamplerSphere::pre_integrate(const Surface * surface, Resources& resources) const
{
    bool inside = (geometry->get_world_to_obj() * surface->position).sqr_length() < SPHERE_RADIUS_SQR;
    if(ellipsoid || inside)
    {
        LightSamplerDataSphereArea * data = resources.memory->create<LightSamplerDataSphereArea>();
        data->eval_type = LightSamplerDataSphere::SPHERE_AREA;
        data->surface = surface;
        return data;
    }
    else
    {
        LightSamplerDataSphereSolidAngle * data = resources.memory->create<LightSamplerDataSphereSolidAngle>();
        data->eval_type = LightSamplerDataSphere::SPHERE_SOLID_ANGLE;
        data->surface = surface;
        data->rng = resources.random_service->get_random_2D();

        data->cd = center - surface->position;
        data->cd_len_sqr = data->cd.sqr_length();
        data->cd_len = sqrtf(data->cd_len_sqr);
        data->cd = data->cd / data->cd_len;
        data->sin_max_sqr = radius_sqr[0] / data->cd_len_sqr;
        data->cos_max = sqrtf(std::max(0.f, 1.f - data->sin_max_sqr));
        data->basis = Transform3f::basis_transform(-data->cd);

        return data;
    }
}

bool LightSamplerSphere::sample(LightSample& sample, const LightSamplerData * data, Resources& resources) const
{
    const LightSamplerDataSphere * light_data = (const LightSamplerDataSphere *)data;
    if(light_data->eval_type == LightSamplerDataSphere::SPHERE_AREA)
        return sample_sphere_area(sample, (const LightSamplerDataSphereArea *)data, resources);
    else if(light_data->eval_type == LightSamplerDataSphere::SPHERE_SOLID_ANGLE)
        return sample_sphere_sa(sample, (const LightSamplerDataSphereSolidAngle *)data, resources);
    else
        return false;
}

float LightSamplerSphere::evaluate(const Intersect * intersect, const LightSamplerData * data, Resources& resources) const
{
    const LightSamplerDataSphere * light_data = (const LightSamplerDataSphere *)data;
    if(light_data->eval_type == LightSamplerDataSphere::SPHERE_AREA)
        return evaluate_sphere_area(intersect, (const LightSamplerDataSphereArea *)data, resources);
    else if(light_data->eval_type == LightSamplerDataSphere::SPHERE_SOLID_ANGLE)
        return evaluate_sphere_sa(intersect, (const LightSamplerDataSphereSolidAngle *)data, resources);
    else
        return 0.f;
}

bool LightSamplerSphere::sample_sphere_sa(LightSample& sample, const LightSamplerDataSphereSolidAngle * data, Resources& resources) const
{
    const Vec2f rnd = data->rng.rand();

    const float theta = TWO_PI * rnd[0];
    const float cos_phi = 1.f - rnd[1] + rnd[1] * data->cos_max;
    const float sin_phi = sqrtf(std::max(0.f, 1.f - cos_phi * cos_phi));

    const float sd_len = data->cd_len * cos_phi - sqrtf(std::max(0.f, radius_sqr[0] - data->cd_len_sqr * sin_phi * sin_phi));
    const float cos_alpha = (data->cd_len_sqr + radius_sqr[0] - sd_len * sd_len) / (2.f * data->cd_len * radius[0]);
    const float sin_alpha = sqrtf(std::max(0.f, 1.f - cos_alpha * cos_alpha));

    const Vec3f normal = data->basis * Vec3f(sin_alpha * cosf(theta), cos_alpha, sin_alpha * sinf(theta));
    sample.position = center + (radius[0] + RAY_OFFSET) * normal;

    const Vec3f direction = (sample.position - data->surface->position).normalized();
    Intersect light_intersect(Ray(sample.position, direction));
    light_intersect.geometry = geometry;
    Surface * light_surface = resources.memory->create<Surface>(sample.position, normal, 0.f, 0.f, 0.f, true);
    light_intersect.geometry_data = light_surface;

    sample.intensity = light->get_intensity(light_surface->u, light_surface->v, 0.f, &light_intersect, resources);

    sample.pdf = 1.f / (TWO_PI * (1.f - data->cos_max));

    return true;
}

float LightSamplerSphere::evaluate_sphere_sa(const Intersect * intersect, const LightSamplerDataSphereSolidAngle * data, Resources& resources) const
{
    return 1.f / (TWO_PI * (1.f - data->cos_max));
}

bool LightSamplerSphere::sample_sphere_area(LightSample& sample, const LightSamplerDataSphereArea * data, Resources& resources) const
{
    // TODO: Fill me in for ellipses and inside the sphere.
    return false;
}

float LightSamplerSphere::evaluate_sphere_area(const Intersect * intersect, const LightSamplerDataSphereArea * data, Resources& resources) const
{
    return 0.f;
}
