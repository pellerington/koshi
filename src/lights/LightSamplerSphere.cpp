#include <lights/LightSamplerSphere.h>

LightSamplerSphere::LightSamplerSphere(GeometrySphere * geometry)
: geometry(geometry)
{
    light = geometry->light.get();

    // Approximate area
    const Transform3f& obj_to_world = geometry->get_obj_to_world();
    const float x_len = obj_to_world.multiply(Vec3f(1.f, 0.f, 0.f), false).length();
    const float y_len = obj_to_world.multiply(Vec3f(0.f, 1.f, 0.f), false).length();
    const float z_len = obj_to_world.multiply(Vec3f(0.f, 0.f, 1.f), false).length();
    const float p = 8.f / 5.f;
    area = FOUR_PI * pow((pow(x_len*y_len, p) + pow(x_len*z_len, p) + pow(y_len*z_len, p)) / 3.f, 1.f / p);
}

bool LightSamplerSphere::sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources)
{
    if(geometry->is_elliptoid() /* or inside sphere */)
        return sample_area(num_samples, pos, pfar, light_samples, resources);
    return sample_sa(num_samples, pos, pfar, light_samples, resources);
}

bool LightSamplerSphere::evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources)
{
    if(geometry->is_elliptoid() /* or inside sphere */)
        return evaluate_area(intersect, pos, pfar, light_sample, resources);
    return evaluate_sa(intersect, pos, pfar, light_sample, resources);
}

bool LightSamplerSphere::sample_sa(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources)
{
    const Vec3f center_dir_t = geometry->get_world_center() - *pos;
    const float center_t_sqr = center_dir_t.sqr_length();
    const float center_t = sqrtf(center_t_sqr);
    const Vec3f center_dir = center_dir_t / center_t;
    const float max_angle = atan2f(geometry->get_world_radius(), center_t);
    const float c = center_dir_t.dot(center_dir_t) - geometry->get_world_radius_sqr();

    Transform3f basis = Transform3f::basis_transform(center_dir);
    const float sample_pdf = center_t_sqr / (TWO_PI * geometry->get_world_radius_sqr());
    RNG &rng = resources.rng; rng.Reset2D();

    for(uint i = 0; i < num_samples; i++)
    {
        const Vec2f rnd = rng.Rand2D();

        const float theta = TWO_PI * rnd[0];
        const float phi = max_angle * sqrtf(rnd[1]);
        const Vec3f sampled_dir = basis * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta));

        // Intersect sphere
        const float a = sampled_dir.dot(sampled_dir);
        const float b = 2.0 * -center_dir_t.dot(sampled_dir);
        const float discriminant = b*b - 4*a*c;
        const float t = (-b - sqrtf(discriminant)) / (2.f*a);

        light_samples.emplace_back();
        LightSample &light_sample = light_samples.back();
        light_sample.position = *pos + t * sampled_dir;
        light_sample.intensity = light->get_emission();
        light_sample.pdf = sample_pdf;
    }

    return true;
}

bool LightSamplerSphere::evaluate_sa(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources)
{
    light_sample.position = intersect.position;
    light_sample.intensity = light->get_emission();
    light_sample.pdf = (geometry->get_world_center() - *pos).sqr_length() / (TWO_PI * geometry->get_world_radius_sqr());

    return true;
}

bool LightSamplerSphere::sample_area(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources)
{
    // TODO: This doesnt work for ellipses at the moment

    const Vec3f obj_pos = geometry->get_world_to_obj() * *pos;
    RNG &rng = resources.rng; rng.Reset2D();

    for(uint i = 0; i < num_samples; i++)
    {
        const Vec2f rnd = rng.Rand2D();

        const float theta = TWO_PI * rnd[0];
        const float phi = acosf(1.f - 2.f * rnd[1]);
        const Vec3f obj_sampled_pos = Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta));
        const Vec3f obj_sampled_dir = (obj_sampled_pos - obj_pos).normalized();

        // Intersect sphere
        float t = -1.f;
        const float a = obj_sampled_dir.dot(obj_sampled_dir);
        const float b = 2.f * obj_pos.dot(obj_sampled_dir);
        const float c = obj_pos.dot(obj_pos) - 1.f; // We can do this before the loop!
        const float discriminant = b*b - 4.f*a*c;
        if(discriminant >= 0.f)
            t = (-b - sqrtf(discriminant)) / (2.f*a);

        if(t < 0.f)
            continue;

        light_samples.emplace_back();
        LightSample &light_sample = light_samples.back();
        light_sample.position = geometry->get_obj_to_world() * (obj_pos + obj_sampled_dir * t);
        light_sample.intensity = light->get_emission();

        const Vec3f light_to_pos = light_sample.position - *pos;
        const float sqr_len = light_to_pos.sqr_length();
        const float cos_theta = fabs((light_to_pos / sqrtf(sqr_len)).dot((light_sample.position - geometry->get_world_center()).normalized()));

        light_sample.pdf = sqr_len / (area * cos_theta);
    }

    return true;
}

bool LightSamplerSphere::evaluate_area(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources)
{
    light_sample.position = intersect.position;
    light_sample.intensity = light->get_emission();

    const Vec3f light_to_pos = light_sample.position - *pos;
    const float sqr_len = light_to_pos.sqr_length();
    const float cos_theta = fabs((light_to_pos / sqrtf(sqr_len)).dot((light_sample.position - geometry->get_world_center()).normalized()));

    light_sample.pdf = sqr_len / (area * cos_theta);

    return true;
}
