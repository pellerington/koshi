#include <integrators/SurfaceIntegratorMultiImportanceSampling.h>
#include <base/ObjectGroup.h>

void SurfaceIntegratorMultiImportanceSampling::pre_render(Scene * scene)
{
    ObjectGroup * input_group = get_attribute<ObjectGroup>("integrators");
    if(!input_group) return;

    for(uint i = 0; i < input_group->size(); i++)
    {
        SurfaceIntegrator * integrator = input_group->get<SurfaceIntegrator>(i);
        if(!integrator) continue;

        integrator->pre_render(scene);
        integrators.push_back(integrator);
    }
}

std::vector<SurfaceSample> SurfaceIntegratorMultiImportanceSampling::integrate_surface(
    MaterialInstance * material_instance, Material * material, 
    const Intersect& intersect, const GeometrySurface * surface,
    Resources& resources) const
{
    std::vector<SurfaceSample> samples;
    for(uint i = 0; i < integrators.size(); i++)
    {
        std::vector<SurfaceSample> sub_samples = integrators[i]->integrate_surface(material_instance, material, intersect, surface, resources);

        for(uint j = 0; j < sub_samples.size(); j++)
        {
            const float pdf_sqr = sub_samples[j].pdf * sub_samples[j].pdf;
            float pdf_sqr_sum = pdf_sqr;
            for(uint k = 0; k < integrators.size(); k++)
            {
                if(k == i) continue;
                float temp_pdf = integrators[k]->evaluate(intersect, sub_samples[j], resources);
                pdf_sqr_sum += temp_pdf * temp_pdf;
            }
            sub_samples[j].weight *= pdf_sqr / pdf_sqr_sum;
        }

        std::move(sub_samples.begin(), sub_samples.end(), std::back_inserter(samples));
    }
    return samples;
}

float SurfaceIntegratorMultiImportanceSampling::evaluate(const Intersect& intersect, const SurfaceSample& sample, Resources& resources)
{
    float pdf = 0.f;
    for(uint i = 0; i < integrators.size(); i++)
        pdf += integrators[i]->evaluate(intersect, sample, resources);
    return pdf;
}

