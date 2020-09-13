#include <koshi/integrator/VolumeSingleScatter.h>

#include <koshi/math/Color.h>
#include <koshi/integrator/SurfaceSampler.h>
#include <koshi/base/Scene.h>

void * VolumeSingleScatter::pre_integrate(const Intersect * intersect, Resources& resources)
{
    const Volume * volume = dynamic_cast<const Volume*>(intersect->geometry_data);
    if(!volume || !volume->segment || !volume->material) return nullptr;

    if(volume->material->homogenous())
    {
        VolumeSingleScatterData * data = resources.memory->create<VolumeSingleScatterData>();
        data->volume = volume;
        return data;
    }
    else
    {
        VolumeSingleScatterHeterogenousData * data = resources.memory->create<VolumeSingleScatterHeterogenousData>(resources);
        data->volume = volume;
        float inv_tlen = 1.f / intersect->tlen;

        Vec3f transmittance = VEC3F_ONES;

        for(Volume::Segment * segment = volume->segment; segment; segment = segment->next)
        {
            float max_density = segment->max_density.max();
            float inv_max_density = 1.f / max_density;
            float tcurr = segment->t0;

            while(true)
            {
                tcurr -= logf(resources.random_service->rand()) * inv_max_density;

                if(tcurr > segment->t1)
                    break;
                
                const float d = (tcurr - intersect->t) * inv_tlen;
                Vec3f uvw = volume->uvw_near * d + volume->uvw_far * (1.f - d);
                Vec3f density = volume->material->get_density(uvw, intersect, resources);
                if(density.null()) continue;
                transmittance = transmittance * (VEC3F_ONES - density * inv_max_density);

                auto& sample = data->samples.push();
                sample.transmittance = transmittance;
                sample.t = tcurr;

                if(transmittance < 0.01f)
                    break;
            }
        }
        return data;
    }
}

Vec3f VolumeSingleScatter::integrate(const Intersect * intersect, void * data, Transmittance& transmittance, Resources& resources) const
{
    if(!data) return VEC3F_ZERO;

    VolumeSingleScatterData * vdata = (VolumeSingleScatterData *)data;
    const Volume * volume = vdata->volume;

    Vec3f color = VEC3F_ZERO;

    // Move to a different system...
    float min_quality = std::pow(1.f / SAMPLES_PER_HEMISPHERE, resources.settings->depth);

    // CHECK THE DEPTH AND QUALITY ECT....
    const uint depth = intersect->path ? intersect->path->depth : 0;
    const float quality = intersect->path ? intersect->path->quality : 1.f;
    if(depth > resources.settings->max_depth || quality < min_quality || depth > 1 || (!volume->material->has_scatter() && !volume->material->has_emission())) 
        return color;

    uint num_samples = std::max(12.f * quality * resources.settings->sampling_quality, 1.f);
    // TODO: Store this in the settings.
    Integrator * scatter_integrator = resources.scene->get_object<Integrator>("default_integrator");

    auto integrate_scattering = [&] (const float& t)
    {
        const float d = (t - intersect->t) / intersect->tlen;
        const Vec3f uvw = volume->uvw_near * d + volume->uvw_far * (1.f - d);

        // Get the scatter and emission.
        const Vec3f density = volume->material->get_density(uvw, intersect, resources);
        const Vec3f scatter = density * volume->material->get_scatter(uvw, intersect, resources);
        const Vec3f emission = density * volume->material->get_emission(uvw, intersect, resources);

        if(scatter.null()) return VEC3F_ZERO;

        Intersect scatter_intersect(intersect->ray, intersect->path);
        scatter_intersect.t = t;
        scatter_intersect.geometry = intersect->geometry;
        scatter_intersect.geometry_primitive = intersect->geometry_primitive;
        Surface surface(intersect->ray.get_position(t), intersect->ray.dir, uvw.u, uvw.v, uvw.w, true);
        surface.material = volume->material->get_surface_material();
        scatter_intersect.geometry_data = &surface;
        
        void * data = scatter_integrator->pre_integrate(&scatter_intersect, resources);
        return scatter * scatter_integrator->integrate(&scatter_intersect, data, transmittance, resources);
    };

    if(volume->material->homogenous())
    {
        float max_density = volume->segment->max_density.max();
        float inv_max_density = 1.f / max_density;
        float max_density_exp = 1.f - std::exp(-max_density * intersect->tlen);
        Random<1> rng = resources.random_service->get_random<1>();
        for(uint i = 0; i < num_samples; i++)
        {
            float t = -logf(1.0f - rng.rand()[0] * max_density_exp) * inv_max_density;
            if(t > volume->t1) continue;
            const float pdf = (max_density * std::exp(-max_density * t)) / max_density_exp;
            t += volume->t0;
            const Vec3f in_scatter = integrate_scattering(t);
            color += in_scatter / (pdf * num_samples);
        }
    }
    else
    {
        VolumeSingleScatterHeterogenousData * vhdata = (VolumeSingleScatterHeterogenousData *)data;
        if(!vhdata->samples.size())
            return color;

        Array<float> transmittance_cdf(resources.memory, vhdata->samples.size());
        transmittance_cdf.push(vhdata->samples[0].t - volume->t0);
        for(uint i = 0; i < vhdata->samples.size()-1; i++)
            transmittance_cdf.push(transmittance_cdf[i] + luminance(vhdata->samples[i].transmittance) * (vhdata->samples[i+1].t - vhdata->samples[i].t));
        transmittance_cdf.push(transmittance_cdf[transmittance_cdf.size()-1] + luminance(vhdata->samples[vhdata->samples.size()-1].transmittance) * (volume->t1 - vhdata->samples[vhdata->samples.size()-1].t));
        const float inv_transmittance_cdfmax = 1.f / transmittance_cdf[transmittance_cdf.size()-1];
        for(uint i = 0; i < transmittance_cdf.size(); i++)
            transmittance_cdf[i] *= inv_transmittance_cdfmax;

        Random<1> rng = resources.random_service->get_random<1>();

        for(uint i = 0; i < num_samples; i++)
        {
            const float r = rng.rand()[0];
            uint cindex = 0; while(r > transmittance_cdf[cindex]) cindex++;

            const float& ct0 = (cindex > 0) ? vhdata->samples[cindex-1].t : volume->t0;
            const float& ct1 = (cindex < (transmittance_cdf.size() - 1)) ? vhdata->samples[cindex].t : volume->t1;
            const float prev_cdf = (cindex > 0) ? transmittance_cdf[cindex-1] : 0.f;
            const float t = ct0 + (ct1 - ct0) * (r - prev_cdf) / (transmittance_cdf[cindex] - prev_cdf);
            const float pdf = (transmittance_cdf[cindex] - prev_cdf) / (ct1 - ct0);

            const Vec3f in_scatter = integrate_scattering(t);
            color += in_scatter / (pdf * num_samples);
        }
    }

    return color;
}

Vec3f VolumeSingleScatter::shadow(const float& t, const Intersect * intersect, void * data, Resources& resources) const
{
    if(!data) return VEC3F_ONES;
    VolumeSingleScatterData * vdata = (VolumeSingleScatterData *)data;
    if(vdata->volume->material->homogenous())
    {
        if(t > vdata->volume->t0)
        {
            const float tlen = std::min(vdata->volume->segment->t1, t) - vdata->volume->segment->t0;
            return Vec3f::exp(-tlen * vdata->volume->segment->max_density);
        }
    }
    else
    {
        VolumeSingleScatterHeterogenousData * vhdata = (VolumeSingleScatterHeterogenousData *)data;
        if(!vhdata->samples.size())
            return VEC3F_ONES;
        if(t > vhdata->volume->t1)
            return vhdata->samples[vhdata->samples.size()-1].transmittance;
        // TODO: Binary search would be faster.
        for(uint i = 0; i < vhdata->samples.size(); i++)
            if(vhdata->samples[i].t < t)
                return vhdata->samples[i].transmittance;
    }
    return VEC3F_ONES;
}