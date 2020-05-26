#include <lights/LightSamplerDirectional.h>
#include <geometry/GeometryDistant.h>

LightSamplerDirectional::LightSamplerDirectional(Geometry * geometry)
: geometry(geometry)
{
    light = geometry->get_attribute<Light>("light");
    direction = geometry->get_obj_to_world().multiply(Vec3f(0, 1, 0), false).normalized();
}

bool LightSamplerDirectional::sample_light(const uint num_samples, const GeometrySurface * surface, std::vector<LightSample>& light_samples, Resources& resources)
{
    light_samples.emplace_back();
    LightSample &light_sample = light_samples.back();

    light_sample.position = surface->position + direction*INV_EPSILON_F;

    Intersect light_intersect(Ray(surface->position, direction));
    light_intersect.geometry = geometry;
    GeometryDistant * geometry_distant = resources.memory.create<GeometryDistant>(0.f, 0.f, direction);
    light_intersect.geometry_data = geometry_distant;
    light_sample.intensity = INV_EPSILON_F * light->get_intensity(geometry_distant->u, geometry_distant->v, 0.f, &light_intersect, resources);
    light_sample.pdf = INV_EPSILON_F;

    return true;
}