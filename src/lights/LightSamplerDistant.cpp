#include <lights/LightSamplerDistant.h>

LightSamplerDistant::LightSamplerDistant(Geometry * geometry)
: geometry(geometry)
{
    light = geometry->get_attribute<Light>("light");
    direction = geometry->get_obj_to_world().multiply(Vec3f(0, 1, 0), false).normalized();
}

bool LightSamplerDistant::sample_light(const uint num_samples, const Intersect * intersect, std::vector<LightSample>& light_samples, Resources& resources)
{
    light_samples.emplace_back();
    LightSample &light_sample = light_samples.back();

    light_sample.position = intersect->surface.position + direction*1e8f;

    Intersect light_intersect(Ray(intersect->surface.position, direction));
    light_intersect.geometry = geometry;
    light_intersect.surface.set(light_sample.position, -direction, 0.f, 0.f, direction);
    light_sample.intensity = INV_EPSILON_F * light->get_intensity(&light_intersect, resources);
    light_sample.pdf = INV_EPSILON_F;

    return true;
}