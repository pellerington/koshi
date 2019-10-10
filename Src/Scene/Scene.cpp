#include "Scene.h"

struct RTCIntersectContextExtended : public RTCIntersectContext {
    Scene * scene;
    VolumeStack * volume_stack;
};

void Scene::get_volumes_callback(const RTCFilterFunctionNArguments * args)
{
    *(args->valid ) = 0;
    RTCIntersectContextExtended * context = (RTCIntersectContextExtended*)args->context;
    Scene * scene = context->scene;

    uint geomID = RTCHitN_geomID(args->hit, args->N, 0);
    scene->rtc_to_obj[geomID]->process_volume_intersection(args, context->volume_stack);
}

void Scene::pre_render()
{
    rtc_scene = rtcNewScene(Embree::rtc_device);
    for(size_t i = 0; i < objects.size(); i++)
    {
        const RTCGeometry& geom = objects[i]->get_rtc_geometry();
        rtcCommitGeometry(geom);
        const uint rtcid = rtcAttachGeometry(rtc_scene, geom);
        rtc_to_obj[rtcid] = objects[i];
        if(objects[i]->volume)
            rtcSetGeometryIntersectFilterFunction(geom, &Scene::get_volumes_callback);
    }
    rtcSetSceneBuildQuality(rtc_scene, RTCBuildQuality::RTC_BUILD_QUALITY_HIGH);
    rtcCommitScene(rtc_scene);
}

Surface Scene::intersect(Ray &ray, VolumeStack * volume_stack)
{
    RTCIntersectContextExtended context;
    context.scene = this;
    context.volume_stack = volume_stack;
    RTCIntersectContext * rtc_context = &context;
    rtcInitIntersectContext(rtc_context);

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
    rtcIntersect1(rtc_scene, rtc_context, &rtcRayHit);

    volume_stack->build(rtcRayHit.ray.tfar);

    ray.t = rtcRayHit.ray.tfar;
    if (rtcRayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
    {
        ray.hit = false;
        return Surface();
    }
    else
    {
        ray.hit = true;
        Surface surface = rtc_to_obj[rtcRayHit.hit.geomID]->process_intersection(rtcRayHit, ray);
        surface.set_volumes(volume_stack->exit_volumes());
        return surface;
    }
}

bool Scene::evaluate_lights(const Ray &ray, std::vector<LightSample> &light_results)
{
    for(uint i = 0; i < lights.size(); i++)
    {
        LightSample isample;
        if(lights[i]->evaluate_light(ray, isample))
            light_results.push_back(isample);
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

bool Scene::sample_lights(const Surface &surface, std::vector<LightSample> &light_samples, const float sample_multiplier)
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
