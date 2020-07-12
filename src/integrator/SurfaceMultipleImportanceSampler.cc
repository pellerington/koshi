#include <integrator/SurfaceMultipleImportanceSampler.h>
#include <base/ObjectGroup.h>

void SurfaceMultipleImportanceSampler::pre_render(Resources& resources)
{
    SurfaceSampler::pre_render(resources);

    ObjectGroup * input_group = get_attribute<ObjectGroup>("integrators");
    if(!input_group) return;

    for(uint i = 0; i < input_group->size(); i++)
    {
        SurfaceSampler * integrator = input_group->get<SurfaceSampler>(i);
        if(!integrator) continue;
        integrators.push_back(integrator);
    }
}

IntegratorData * SurfaceMultipleImportanceSampler::pre_integrate(const Intersect * intersect, Resources& resources)
{
    SurfaceMultipleImportanceSamplerData * data = resources.memory->create<SurfaceMultipleImportanceSamplerData>(resources);
    data->surface = dynamic_cast<const Surface *>(intersect->geometry_data);
    for(uint i = 0; i < integrators.size(); i++)
    {
        Integrator * integrator = integrators[i];
        IntegratorData *  integrator_data = integrator->pre_integrate(intersect, resources);
        data->integrator_data.push(dynamic_cast<SurfaceSamplerData*>(integrator_data));
    }
    return data;
}

void SurfaceMultipleImportanceSampler::scatter_surface(
    Array<SurfaceSample>& samples,
    const MaterialLobes& lobes, 
    const Intersect * intersect, SurfaceSamplerData * data,
    Interiors& interiors, Resources& resources) const
{
    SurfaceMultipleImportanceSamplerData * sampler_data = (SurfaceMultipleImportanceSamplerData *)data;
    for(uint i = 0; i < integrators.size(); i++)
    {
        const uint prev_size = samples.size();
        integrators[i]->scatter_surface(samples, lobes, intersect, sampler_data->integrator_data[i], interiors, resources);

        for(uint j = prev_size; j < samples.size(); j++)
        {
            if(samples[j].pdf < INV_EPSILON_F)
            {
                const float pdf_sqr = samples[j].pdf * samples[j].pdf;
                float pdf_sqr_sum = pdf_sqr;
                for(uint k = 0; k < integrators.size(); k++)
                {
                    if(k == i) continue;
                    float temp_pdf = integrators[k]->evaluate(samples[j], lobes, intersect, sampler_data->integrator_data[k], resources);
                    pdf_sqr_sum += temp_pdf * temp_pdf;
                }
                samples[j].weight *= pdf_sqr / pdf_sqr_sum;
            }
        }
    }
}

float SurfaceMultipleImportanceSampler::evaluate(
    const SurfaceSample& sample, 
    const MaterialLobes& lobes,
    const Intersect * intersect, SurfaceSamplerData * data,
    Resources& resources) const
{
    SurfaceMultipleImportanceSamplerData * sampler_data = (SurfaceMultipleImportanceSamplerData *)data;
    float pdf = 0.f;
    for(uint i = 0; i < integrators.size(); i++)
        pdf += integrators[i]->evaluate(sample, lobes, intersect, sampler_data->integrator_data[i], resources);
    return pdf;
}

