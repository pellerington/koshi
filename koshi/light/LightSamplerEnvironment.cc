#include <koshi/light/LightSamplerEnvironment.h>
#include <koshi/material/MaterialLight.h>
#include <koshi/math/Helpers.h>
#include <koshi/math/Color.h>

void LightSamplerEnvironment::pre_render(Resources& resources)
{
    material = geometry->get_attribute<Material>("material");

    MaterialLight * material_light = dynamic_cast<MaterialLight*>(material);

    const Texture * intensity_texture = (material_light) ? material_light->get_intensity_texture() : nullptr;
    Vec3f delta = (intensity_texture) ? intensity_texture->delta() : VEC3F_ONES;
    delta.clamp(1.f / 2048.f, 1.f / 32.f);

    uint u_resolution = 1.f / delta.u;
    uint v_resolution = 1.f / delta.v;

    cdfu = std::vector<std::vector<float>>(v_resolution, std::vector<float>(u_resolution));
    cdfv = std::vector<float>(v_resolution);

    float u_offset = 0.5f / u_resolution, v_offset = 0.5f / v_resolution;
    for(uint v = 0; v < v_resolution; v++)
    {
        for(uint u = 0; u < u_resolution; u++)
        {
            // TODO: Fill out all the intersection as well.
            cdfu[v][u] = luminance(material->emission(u * delta.u + u_offset, v * delta.v + v_offset, 0.f, nullptr, resources));
            cdfu[v][u] += (u > 0) ? cdfu[v][u-1] : 0.f;

            // std::cout << u * delta.u + u_offset << " " << v * delta.v + v_offset << " | ";
        }

        cdfv[v] = cdfu[v][u_resolution-1] * sinf((v * delta.v + v_offset) * PI);
        cdfv[v] += (v > 0) ? cdfv[v-1] : 0.f;

        for(uint u = 0; u < u_resolution; u++)
            cdfu[v][u] /= cdfu[v][u_resolution-1];

        // std::cout << "\n";
    }
    for(uint v = 0; v < v_resolution; v++)
        cdfv[v] /= cdfv[v_resolution-1];
}

const LightSamplerData * LightSamplerEnvironment::pre_integrate(const Surface * surface, Resources& resources) const
{
    LightSamplerDataEnvironment * data = resources.memory->create<LightSamplerDataEnvironment>();
    data->surface = surface;
    data->rng = resources.random_service->get_random<2>();
    return data;
}

bool LightSamplerEnvironment::sample(LightSample& sample, const LightSamplerData * data, Resources& resources) const
{
    const LightSamplerDataEnvironment * light_data = (const LightSamplerDataEnvironment *)data;
    const Surface * surface = light_data->surface;

    const float * rnd = light_data->rng.rand();

    const uint vi = std::lower_bound(cdfv.begin(), cdfv.end(), rnd[0]) - cdfv.begin();
    const float vmin = (vi > 0) ? cdfv[vi-1] : 0.f;
    const float v = (vi + (rnd[0] - vmin) / (cdfv[vi] - vmin)) / cdfv.size();

    const uint ui = std::lower_bound(cdfu[vi].begin(), cdfu[vi].end(), rnd[1]) - cdfu[vi].begin();
    const float umin = (ui > 0) ? cdfu[vi][ui-1] : 0.f;
    const float u = (ui + (rnd[1] - umin) / (cdfu[vi][ui] - umin)) / cdfu[vi].size();

    const float theta = u * TWO_PI;
    const float phi = v * PI;

    Vec3f direction = geometry->get_obj_to_world().multiply(Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta)), false);

    sample.position = surface->position + direction*INV_EPSILON_F;

    Intersect light_intersect(Ray(surface->position, direction));
    light_intersect.geometry = geometry;
    SurfaceDistant * distant = resources.memory->create<SurfaceDistant>(u, v, 0.f);
    light_intersect.geometry_data = distant;

    sample.intensity = material->emission(distant->u, distant->v, distant->w, &light_intersect, resources);

    sample.pdf = (cdfv[vi] - vmin) * (cdfu[vi][ui] - umin) * (cdfv.size()) * (cdfu[vi].size());
    sample.pdf /= (PI * TWO_PI * sinf(phi));

    return true;
}

float LightSamplerEnvironment::evaluate(const Intersect * intersect, const LightSamplerData * data, Resources& resources) const
{
    const Surface * light_surface = (const Surface*)intersect->geometry_data;

    const uint vi = light_surface->v * cdfv.size(), ui = light_surface->u * cdfu[vi].size();
    const float vmin = (vi > 0) ? cdfv[vi-1] : 0.f, umin = (ui > 0) ? cdfu[vi][ui-1] : 0.f;
    float pdf = (cdfv[vi] - vmin) * (cdfu[vi][ui] - umin) * (cdfv.size()) * (cdfu[vi].size());

    const float phi = light_surface->v * PI;
    pdf /= (PI * TWO_PI * sinf(phi));

    return pdf;
}
