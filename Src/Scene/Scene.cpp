#include "Scene.h"

void Scene::intersection_callback(const RTCFilterFunctionNArguments * args)
{
    IntersectContext * context = (IntersectContext*) args->context;
    const uint geomID = RTCHitN_geomID(args->hit, args->N, 0);
    std::shared_ptr<Object> obj = context->scene->rtc_to_obj[geomID];

    // If we are invisible don't do anything else
    if(!(*(args->valid) = obj->process_visibility_intersection(context->ray->camera) ? -1 : 0))
        return;

    const double t = RTCRayN_tfar(args->ray, args->N, 0);
    const Vec3f normal = Vec3f::normalize(Vec3f(RTCHitN_Ng_x(args->hit, args->N, 0), RTCHitN_Ng_y(args->hit, args->N, 0), RTCHitN_Ng_z(args->hit, args->N, 0)));
    const bool front = normal.dot(-context->ray->dir) > 0.f;

    if(obj->volume)
        obj->process_volume_intersection(t, front, context->volume_stack);

    *(args->valid) = (obj->material != nullptr || obj->light != nullptr) ? -1 : 0;
}

void Scene::pre_render()
{
    rtc_scene = rtcNewScene(Embree::rtc_device);
    for(size_t i = 0; i < objects.size(); i++)
    {
        const RTCGeometry& geom = objects[i]->get_rtc_geometry();
        if(objects[i]->volume || objects[i]->variable_visibility())
        {
            objects[i]->set_filter_function(&Scene::intersection_callback);
            rtcSetGeometryIntersectFilterFunction(geom, &Scene::intersection_callback);
        }
        rtcCommitGeometry(geom);
        const uint rtcid = rtcAttachGeometry(rtc_scene, geom);
        rtc_to_obj[rtcid] = objects[i];
        objects[i]->set_id(rtcid);
    }
    rtcSetSceneBuildQuality(rtc_scene, RTCBuildQuality::RTC_BUILD_QUALITY_HIGH);
    rtcCommitScene(rtc_scene);
}

Intersect Scene::intersect(Ray &ray)
{
    VolumeStack volume_stack(ray.in_volumes);

    IntersectContext context;
    context.scene = this;
    context.ray = &ray;
    context.volume_stack = &volume_stack;

    RTCIntersectContext * rtc_context = &context;
    rtcInitIntersectContext(rtc_context);

    RTCRayHit rtcRayHit;
    rtcRayHit.ray.org_x = ray.pos[0]; rtcRayHit.ray.org_y = ray.pos[1]; rtcRayHit.ray.org_z = ray.pos[2];
    rtcRayHit.ray.dir_x = ray.dir[0]; rtcRayHit.ray.dir_y = ray.dir[1]; rtcRayHit.ray.dir_z = ray.dir[2];
    rtcRayHit.ray.tnear = ray.tmin;
    rtcRayHit.ray.tfar = ray.tmax;
    rtcRayHit.ray.time = 0.f;
    rtcRayHit.ray.mask = -1;
    rtcRayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtcRayHit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtcRayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    // Intersect ray with scene
    rtcIntersect1(rtc_scene, rtc_context, &rtcRayHit);

    volume_stack.build(rtcRayHit.ray.tfar);

    if (rtcRayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
    {
        ray.hit = false;
        Surface surface = (ray.tmax == FLT_MAX) ? Surface(ray.dir) : Surface();
        return Intersect(nullptr, std::move(surface), std::move(volume_stack));
    }
    else
    {
        ray.t = rtcRayHit.ray.tfar;
        ray.tmax = rtcRayHit.ray.tfar;
        ray.hit = true;
        Surface surface = rtc_to_obj[rtcRayHit.hit.geomID]->process_intersection(rtcRayHit, ray);
        return Intersect(rtc_to_obj[rtcRayHit.hit.geomID], std::move(surface), std::move(volume_stack));
    }
}

void Scene::sample_lights(const Surface &surface, std::vector<LightSample> &light_samples, const float sample_multiplier)
{
    for(size_t i = 0; i < lights.size(); i++)
    {
        // Make a better num_samples estimator using bbox.
        const uint num_samples = std::max(1.f, SAMPLES_PER_SA * sample_multiplier);
        lights[i]->sample_light(num_samples, &surface.position, nullptr, light_samples);
    }
}

void Scene::evaluate_distant_lights(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample)
{
    for(size_t i = 0; i < distant_lights.size(); i++)
    {
        LightSample isample;
        if(distant_lights[i]->evaluate_light(intersect, pos, pfar, isample))
        {
            light_sample.intensity += isample.intensity;
            light_sample.pdf += isample.pdf;
        }
    }
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

bool Scene::add_light(std::shared_ptr<Object> light)
{
    lights.push_back(light);
    add_object(light);
    return true;
}

bool Scene::add_distant_light(std::shared_ptr<Object> light)
{
    distant_lights.push_back(light);
    lights.push_back(light);
    add_object(light);
    return true;
}

bool Scene::add_texture(std::shared_ptr<Texture> texture)
{
    textures.push_back(texture);
    return true;
}
