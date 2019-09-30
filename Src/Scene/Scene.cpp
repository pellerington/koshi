#include "Scene.h"

#if EMBREE

void Scene::pre_render()
{
    rtc_scene = rtcNewScene(Embree::rtc_device);
    for(size_t i = 0; i < objects.size(); i++)
    {
        const uint rtcid = objects[i]->attach_to_scene(rtc_scene);
        rtc_to_obj[rtcid] = objects[i];
    }
    rtcSetSceneBuildQuality(rtc_scene, RTCBuildQuality::RTC_BUILD_QUALITY_HIGH);
    rtcCommitScene(rtc_scene);
}

bool Scene::intersect(Ray &ray, Surface &surface)
{
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);

    RTCRayHit rtcRayHit;
    rtcRayHit.ray.org_x = ray.pos[0]; rtcRayHit.ray.org_y = ray.pos[1]; rtcRayHit.ray.org_z = ray.pos[2];
    rtcRayHit.ray.dir_x = ray.dir[0]; rtcRayHit.ray.dir_y = ray.dir[1]; rtcRayHit.ray.dir_z = ray.dir[2];
    rtcRayHit.ray.tnear = 0.f;
    rtcRayHit.ray.tfar = ray.t;
    rtcRayHit.ray.time = 0.f;
    rtcRayHit.ray.mask = -1;
    rtcRayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtcRayHit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtcRayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    /* intersect ray with scene */
    rtcIntersect1(rtc_scene, &context, &rtcRayHit);

    ray.t = rtcRayHit.ray.tfar;
    if (rtcRayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
    {
        ray.hit = false;
        return false;
    }
    else
    {
        ray.hit = true;
        rtc_to_obj[rtcRayHit.hit.geomID]->process_intersection(rtcRayHit, ray, surface);
        return true;
    }
}

#else

void Scene::pre_render()
{
    std::vector<std::shared_ptr<Object>> sub_objects;
    for(size_t i = 0; i < objects.size(); i++)
    {
        std::vector<std::shared_ptr<Object>> i_sub_objects = objects[i]->get_sub_objects();
        sub_objects.insert(sub_objects.end(), i_sub_objects.begin(), i_sub_objects.end());
    }
    accelerator = std::unique_ptr<Accelerator>(new Accelerator(sub_objects));
}

bool Scene::intersect(Ray &ray, Surface &surface)
{
    if(accelerator)
    {
        return accelerator->intersect(ray, surface);
    }
    else
    {
        for(size_t i = 0; i < objects.size(); i++)
        {
            objects[i]->intersect(ray, surface);
        }
        return ray.hit;
    }
}

#endif

bool Scene::evaluate_lights(const Ray &ray, LightSample &light_sample)
{
    light_sample.intensity = 0.f;
    light_sample.pdf = 0.f;

    for(uint i = 0; i < lights.size(); i++)
    {
        LightSample isample;
        if(lights[i]->evaluate_light(ray, isample))
        {
            light_sample.intensity += isample.intensity;
            light_sample.pdf += isample.pdf;
            // return an array so that the position makes sense.
        }
    }

    return true;
}

Vec3f Scene::evaluate_environment_light(const Ray &ray)
{
    LightSample light_sample;
    if(environment_light)
    {
        if(!environment_light->evaluate_light(ray, light_sample))
            return VEC3F_ZERO;
        return light_sample.intensity;
    }
    return VEC3F_ZERO;
}

bool Scene::sample_lights(const Surface &surface, std::deque<LightSample> &light_samples, const float sample_multiplier)
{
    for(size_t i = 0; i < lights.size(); i++)
    {
        const uint num_samples = std::max(1.f, lights[i]->estimated_samples(surface) * sample_multiplier);
        lights[i]->sample_light(num_samples, surface, light_samples);
    }
    return true;
}

bool Scene::add_object(std::shared_ptr<Object> object)
{
    objects.push_back(object);
    return true;
}

bool Scene::add_material(std::shared_ptr<Material> material)
{
    materials.push_back(material);
    return true;
}

bool Scene::add_light(std::shared_ptr<Light> light)
{
    if(light->type == Light::Environment)
        environment_light = light;
    else
        lights.push_back(light);
    return true;
}

bool Scene::add_texture(std::shared_ptr<Texture> texture)
{
    textures.push_back(texture);
    return true;
}
