#include <lights/LightSamplerSphere.h>

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

bool LightSamplerSphere::sample_light(const uint num_samples, const Surface * surface, std::vector<LightSample>& light_samples, Resources& resources)
{
    if(ellipsoid || (geometry->get_world_to_obj() * surface->position).sqr_length() < SPHERE_RADIUS_SQR)
        return sample_area(num_samples, surface, light_samples, resources);
    return sample_sa(num_samples, surface, light_samples, resources);
}

float LightSamplerSphere::evaluate_light(const Intersect * intersect, const Surface * surface, Resources& resources)
{
    if(ellipsoid || (geometry->get_world_to_obj() * surface->position).sqr_length() < SPHERE_RADIUS_SQR)
        return evaluate_area(intersect, surface, resources);
    return evaluate_sa(intersect, surface, resources);
}

bool LightSamplerSphere::sample_sa(const uint num_samples, const Surface * surface, std::vector<LightSample>& light_samples, Resources& resources)
{
    Vec3f cd = center - surface->position;
    const float cd_len_sqr = cd.sqr_length();
    const float cd_len = sqrtf(cd_len_sqr);
    cd = cd / cd_len;

    const float sin_max_sqr = radius_sqr[0] / cd_len_sqr;
    const float cos_max = sqrtf(std::max(0.f, 1.f - sin_max_sqr));

    Transform3f basis = Transform3f::basis_transform(-cd);

    Random2D rng = resources.random_service->get_random_2D();

    for(uint i = 0; i < num_samples; i++)
    {
        const Vec2f rnd = rng.rand();

        const float theta = TWO_PI * rnd[0];
        const float cos_phi = 1.f - rnd[1] + rnd[1] * cos_max;
        const float sin_phi = sqrtf(std::max(0.f, 1.f - cos_phi * cos_phi));

        const float sd_len = cd_len * cos_phi - sqrtf(std::max(0.f, radius_sqr[0] - cd_len_sqr * sin_phi * sin_phi));
        const float cos_alpha = (cd_len_sqr + radius_sqr[0] - sd_len * sd_len) / (2.f * cd_len * radius[0]);
        const float sin_alpha = sqrtf(std::max(0.f, 1.f - cos_alpha * cos_alpha));

        light_samples.emplace_back();
        LightSample &light_sample = light_samples.back();

        const Vec3f normal = basis * Vec3f(sin_alpha * cosf(theta), cos_alpha, sin_alpha * sinf(theta));
        light_sample.position = center + (radius[0] + RAY_OFFSET) * normal;

        const Vec3f direction = (light_sample.position - surface->position).normalized();
        Intersect light_intersect(Ray(light_sample.position, direction));
        light_intersect.geometry = geometry;
        Surface * light_surface = resources.memory->create<Surface>(light_sample.position, normal, 0.f, 0.f, 0.f, true);
        light_intersect.geometry_data = light_surface;

        light_sample.intensity = light->get_intensity(light_surface->u, light_surface->v, 0.f, &light_intersect, resources);

        light_sample.pdf = 1.f / (TWO_PI * (1.f - cos_max));
    }

    return true;
}

float LightSamplerSphere::evaluate_sa(const Intersect * intersect, const Surface * surface, Resources& resources)
{
    const float cd_len_sqr = (center - surface->position).sqr_length();
    const float sin_max_sqr = radius_sqr[0] / cd_len_sqr;
    const float cos_max = sqrtf(std::max(0.f, 1.f - sin_max_sqr));
    return 1.f / (TWO_PI * (1.f - cos_max));
}

bool LightSamplerSphere::sample_area(const uint num_samples, const Surface * surface, std::vector<LightSample>& light_samples, Resources& resources)
{
    // TODO: Fill me in for ellipses and inside the sphere.
    return false;
}

float LightSamplerSphere::evaluate_area(const Intersect * intersect, const Surface * surface, Resources& resources)
{
    return 0.f;
}
