#include "VolumeIntegrator.h"

Vec3f VolumeIntegrator::shadow(const float &t)
{

    // Do shadow per volume ( so we can do homo easy and hetro using residual ratio )

    Vec3f tr = VEC3F_ONES;
    for(auto curr_volume = volumes.begin(); curr_volume != volumes.end(); curr_volume++)
    {
        if(t < curr_volume->tmax)
        {
            tr *= Vec3f::exp(curr_volume->max_density * (curr_volume->tmin - t));
            break;
        }
        else
            tr *= Vec3f::exp(curr_volume->max_density * (curr_volume->tmin - curr_volume->tmax));
    }

    return tr;
}

// SHould we pass our path tracer object here so we can use it directly???
Vec3f VolumeIntegrator::integrate(Vec3f &surface_weight, std::vector<VolumeSample> &samples, VolumeSample * in_sample)
{
    if(!volumes.num_volumes())
    {
        surface_weight = VEC3F_ONES;
        return VEC3F_ZERO;
    }

    Vec3f weight = /*(in_sample) ? in_sample->weight :*/ VEC3F_ONES;

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

            Vec3f null_density = sampling_density - density;
            null_density.abs();

            Vec3f absorbtion = density - scattering;

            float n_prob = null_density.max();
            float s_prob = scattering.max();
            float a_prob = absorbtion.max();
            float inv_sum = 1.f / (n_prob + s_prob + a_prob);
            n_prob *= inv_sum; s_prob *= inv_sum; a_prob *= inv_sum;

            const float r = RNG::Rand();

            // Absorbtion event
            if(r < a_prob)
            {
                Vec3f emission;
                for(uint i = 0; i < volume_isect->volumes.size(); i++)
                    emission += volume_isect->volumes[i]->get_emission();

                surface_weight = VEC3F_ZERO;
                return weight * (absorbtion / (sampling_density * a_prob)) * emission;
            }

            // Scattering event
            else if(r < 1.f - n_prob)
            {
                // if(volume_isect->volumes.size() > 1)
                // {
                //      // Pick a scatter!
                //      // Also divide by the probability we picked this scatter?
                // }
                // else

                samples.emplace_back();
                VolumeSample& sample = samples.back();

                // Set up sample
                sample.pos = ray.get_position(t);
                volume_isect->volumes[0]->sample_volume(ray.dir, sample);
                sample.weight = weight * scattering / (sampling_density * s_prob);
                sample.quality = (in_sample) ? in_sample->quality : 1.f;
                sample.exit_volumes = &volume_isect->volumes;
                samples.push_back(sample);

                // Cap the ray t so lights cant be evaluated further.
                ray.t = t;

                // Dont evaluate the surface.
                surface_weight = VEC3F_ZERO;

                // Return no emission
                return VEC3F_ZERO;
            }

            // Null event
            else
            {
                weight *= null_density / (sampling_density * n_prob);
            }
        }
    }

    surface_weight = weight;
    return VEC3F_ZERO;
}
