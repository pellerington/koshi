#include "MultiScatVolumeIntegrator.h"

MultiScatVolumeIntegrator::MultiScatVolumeIntegrator(Scene * scene, Ray &ray, const VolumeStack& volumes)
: VolumeIntegrator(scene, ray, volumes)
{
    weight = VEC3F_ONES;
    tmax = FLT_MAX;
    has_scatter = false;

    if(!volumes.num_volumes())
        return;

    float t = volumes.tmin;

    for(auto volume_isect = volumes.begin(); volume_isect != volumes.end(); volume_isect++)
    {
        t = volume_isect->tmin;

        while(t < volume_isect->tmax)
        {
            //Sample a position
            float sampling_density = volume_isect->max_density.max();
            t += -std::log(RNG::Rand()) / sampling_density;

            if(t >= volume_isect->tmax) break;

            Vec3f density;
            Vec3f scattering;
            std::vector<Vec3f> scattering_cache(volume_isect->volumes.size());
            for(uint i = 0; i < volume_isect->volumes.size(); i++)
            {
                Vec3f idensity = volume_isect->volumes[i]->get_density();
                density += idensity;
                scattering_cache[i] = idensity * volume_isect->volumes[i]->get_scattering();
                scattering += scattering_cache[i];
            }
            Vec3f absorbtion = density - scattering;

            Vec3f null_density = sampling_density - density;
            null_density.abs();
            // TODO: Ignore zero densities
            // null_density *= density.lambda([](const float n) { return (n == 0.f) ? 0.f : 1.f; });

            // TODO: We need to pass in history probabilites somehow
            float n_prob = null_density.max();
            float s_prob = scattering.max();
            float a_prob = absorbtion.max();
            const float inv_sum = 1.f / (n_prob + s_prob + a_prob);
            n_prob *= inv_sum; s_prob *= inv_sum; a_prob *= inv_sum;

            const float r = RNG::Rand();

            // Absorbtion event
            if(r < a_prob)
            {
                ray.tmax = tmax = t;
                weight *= (absorbtion / (sampling_density * a_prob));
                for(uint i = 0; i < volume_isect->volumes.size(); i++)
                    weighted_emission += volume_isect->volumes[i]->get_emission();
                weighted_emission *= weight;
                return;
            }

            // Scattering event
            else if(r < 1.f - n_prob)
            {
                // Cap the ray tmax so lights cant be evaluated further.
                ray.tmax = tmax = t;
                weight *= scattering / (sampling_density * s_prob);
                has_scatter = true;

                // if(volume_isect->volumes.size() == 1)
                // {
                    // Set up sample
                    sample.pos = ray.get_position(t);
                    volume_isect->volumes[0]->sample_volume(ray.dir, sample);
                    sample.weight = weight;
                    sample.quality = 1.f;
                    sample.exit_volumes = &volume_isect->volumes;
                    return;
                // }
                // else
                // {
                //     // Pick a scatter!
                //     // Also divide by the probability we picked this scatter?
                // }
            }

            // Null event
            else
            {
                weight *= null_density / (sampling_density * n_prob);
            }
        }
    }
}

Vec3f MultiScatVolumeIntegrator::emission()
{
    return weighted_emission;
}

Vec3f MultiScatVolumeIntegrator::shadow(const float &t)
{
    // TODO: If we have lights between our sample then this is incorrect! We should store a stack of weights
    // Possible better soultion. Have lights be intersectable just like any other object in the scene.
    // This would mean the ray would naturally terminate at the light.
    // Could be an emissive material which samples a ray straight past it if the user wants it to be transparent.

    if(t < tmax)
        return weight;
    return VEC3F_ZERO;
}

void MultiScatVolumeIntegrator::scatter(std::vector<VolumeSample> &samples)
{
    if(has_scatter)
        samples.push_back(sample);
}
