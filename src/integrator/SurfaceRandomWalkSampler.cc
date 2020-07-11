#include <integrator/SurfaceRandomWalkSampler.h>
#include <material/MaterialRandomWalk.h>
#include <intersection/Intersector.h>

void SurfaceRandomWalkSampler::scatter_surface(
    Array<SurfaceSample>& samples,
    const MaterialInstance& material_instance,
    const Intersect * intersect, SurfaceSamplerData * data, 
    Interiors& interiors, Resources& resources) const
{
    const Surface * surface = data->surface;
    const uint depth = intersect->path ? intersect->path->depth : 0;
    const float quality = intersect->path ? intersect->path->quality : 1.f;

    if(!surface->facing)
        return;

    for(uint l = 0; l < material_instance.size(); l++)
    {
        const MaterialLobeRandomWalk * subsurface = dynamic_cast<const MaterialLobeRandomWalk *>(material_instance[l]);
        if(subsurface)
        {
            uint num_samples = SAMPLES_PER_SA * resources.settings->sampling_quality * quality;

            PathData next_path;
            next_path.depth = depth + 1;
            next_path.quality = quality / num_samples;
            next_path.prev_path = intersect->path;

            float max_density = subsurface->density.max();
            float inv_max_density = 1.f / max_density;
            Vec3f null_density = max_density - subsurface->density;

            for(uint i = 0; i < num_samples; i++)
            {
                // TODO: Move this into a function in the lobe? so we can have a generic "ScatterSampler" or have a lobe integrator?

                // Sample an initial direction.
                const float * rnd = subsurface->rng.rand();
                const float theta = TWO_PI * rnd[0];
                const float r = sqrtf(rnd[1]);
                const float x = r * cosf(theta), z = r * sinf(theta), y = sqrtf(std::max(EPSILON_F, 1.f - rnd[1]));
                Vec3f wo = subsurface->transform * -Vec3f(x, y, z);                
 
                // Setup wieghts and pdf.
                Vec3f weight = VEC3F_ONES;
                Vec3f position = surface->back_position;

                // Find scatter postion.
                uint bounces = 0;
                while(bounces++ < 32)
                {
                    // Sample a length
                    const float tmax = -logf(resources.random_service->rand()) * inv_max_density;

                    // Intersect the sampled tmax.
                    Ray ray(position, wo, 0.f, tmax);
                    IntersectList * intersects = subsurface->intersector->intersect(ray, &next_path, resources, &intersection_callback);
                    if(intersects->hit())
                    {
                        SurfaceSample& sample = samples.push();
                        sample.intersects = intersects;
                        ((Surface *)intersects->get(0)->geometry_data)->material = subsurface->exit_material;
                        sample.li = Integrator::shade(intersects, resources);
                        sample.weight = weight;
                        sample.pdf = 1.f;                        
                        sample.scatter = true;
                        break;
                    }

                    // Update the position;
                    position = ray.get_position(tmax);

                    // Event probabilities.
                    float nprob = (weight * null_density).avg();
                    float sprob = (weight * subsurface->scatter).avg();
                    const float inv_sum = 1.f / (nprob + sprob);
                    nprob *= inv_sum; sprob *= inv_sum;

                    // Null event
                    if(resources.random_service->rand() < nprob)
                    {
                        weight *= null_density * inv_max_density / nprob;
                    }
                    // Scatter event
                    else
                    {
                        const float theta = TWO_PI * resources.random_service->rand();
                        float cos_phi = 0.f;
                        if(subsurface->g > EPSILON_F || subsurface->g < -EPSILON_F)
                        {
                            float a = (1.f - subsurface->g_sqr) / (1.f - subsurface->g + 2.f * subsurface->g * resources.random_service->rand());
                            cos_phi = (0.5f * subsurface->g_inv) * (1.f + subsurface->g_sqr - a * a);
                        }
                        else
                        {
                            cos_phi = 1.f - 2.f * resources.random_service->rand();
                        }
                        float sin_phi = sqrtf(std::max(EPSILON_F, 1.f - cos_phi * cos_phi));
                        const float x = sin_phi * cosf(theta), z = sin_phi * sinf(theta), y = cos_phi;
                        wo = Transform3f::basis_transform(wo) * Vec3f(x, y, z);

                        weight *= subsurface->scatter * inv_max_density / sprob;
                    }
                }
            }

            float inv_num_samples = 1.f / num_samples;
            for(uint i = 0; i < num_samples; i++)
                samples[i].weight *= inv_num_samples;

        }
    }
}

void SurfaceRandomWalkSampler::post_intersection_callback(IntersectList * intersects, void * data, Resources& resources)
{
    for(uint i = 0; i < intersects->size(); i++)
        if(((Surface*)intersects->get(i)->geometry_data)->facing)
            i = intersects->pop(i) - 1;

    if(intersects->size() > 1)
    {
        float tmin = FLT_MAX;
        for(uint i = 0; i < intersects->size(); i++)
            tmin = std::min(tmin, intersects->get(i)->t);
        for(uint i = 0; i < intersects->size(); i++)
            if(intersects->get(i)->t > tmin)
                i = intersects->pop(i) - 1;
    }
}

IntersectionCallbacks SurfaceRandomWalkSampler::intersection_callback(nullptr, nullptr, SurfaceRandomWalkSampler::post_intersection_callback, nullptr);
