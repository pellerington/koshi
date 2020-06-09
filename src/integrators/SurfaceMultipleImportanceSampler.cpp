#include <integrators/SurfaceMultipleImportanceSampler.h>
#include <base/ObjectGroup.h>

void SurfaceMultipleImportanceSampler::pre_render(Resources& resources)
{
    ObjectGroup * input_group = get_attribute<ObjectGroup>("integrators");
    if(!input_group) return;

    for(uint i = 0; i < input_group->size(); i++)
    {
        SurfaceSampler * integrator = input_group->get<SurfaceSampler>(i);
        if(!integrator) continue;

        // integrator->pre_render(scene);
        integrators.push_back(integrator);
    }
}

std::vector<SurfaceSample> SurfaceMultipleImportanceSampler::integrate_surface(
    const MaterialInstance& material_instance, 
    const Intersect * intersect, const Surface * surface,
    Interiors& interiors, Resources& resources) const
{
    std::vector<SurfaceSample> samples;
    for(uint i = 0; i < integrators.size(); i++)
    {
        std::vector<SurfaceSample> sub_samples = integrators[i]->integrate_surface(material_instance, intersect, surface, interiors, resources);

        for(uint j = 0; j < sub_samples.size(); j++)
        {
            if(sub_samples[j].pdf < INV_EPSILON_F)
            {
                const float pdf_sqr = sub_samples[j].pdf * sub_samples[j].pdf;
                float pdf_sqr_sum = pdf_sqr;
                for(uint k = 0; k < integrators.size(); k++)
                {
                    if(k == i) continue;
                    float temp_pdf = integrators[k]->evaluate(sub_samples[j], material_instance, intersect, surface, resources);
                    pdf_sqr_sum += temp_pdf * temp_pdf;
                }
                sub_samples[j].weight *= pdf_sqr / pdf_sqr_sum;
            }
        }

        std::move(sub_samples.begin(), sub_samples.end(), std::back_inserter(samples));
    }
    return samples;
}

float SurfaceMultipleImportanceSampler::evaluate(const SurfaceSample& sample, 
    const MaterialInstance& material_instance,
    const Intersect * intersect, const Surface * surface, 
    Resources& resources) const
{
    float pdf = 0.f;
    for(uint i = 0; i < integrators.size(); i++)
        pdf += integrators[i]->evaluate(sample, material_instance, intersect, surface, resources);
    return pdf;
}

