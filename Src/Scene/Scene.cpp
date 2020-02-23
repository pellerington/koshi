#include "Scene.h"

void Scene::pre_render()
{
    rtc_scene = rtcNewScene(Embree::rtc_device);
    for(size_t i = 0; i < objects.size(); i++)
    {
        objects[i]->set_id(i);
        const RTCGeometry& geom = objects[i]->get_rtc_geometry();
        if(objects[i]->use_intersection_filter())
            rtcSetGeometryIntersectFilterFunction(geom, &Scene::intersection_callback);
        rtcCommitGeometry(geom);
        rtcAttachGeometryByID(rtc_scene, geom, i);
        rtcSetGeometryUserData(geom, objects[i].get());
    }
    rtcSetSceneBuildQuality(rtc_scene, RTCBuildQuality::RTC_BUILD_QUALITY_HIGH);
    rtcCommitScene(rtc_scene);
}

void Scene::intersection_callback(const RTCFilterFunctionNArguments * args)
{
    Object * obj = (Object*)args->geometryUserPtr;
    obj->process_intersection_visibility(args);
    if(obj->volume)
        obj->process_intersection_volume(args);
}

Intersect Scene::intersect(Ray &ray)
{
    Intersect intersect(ray);

    IntersectContext context;
    context.ray = &ray;
    context.volumes = &intersect.volumes;
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

    // Perform intersection
    rtcIntersect1(rtc_scene, rtc_context, &rtcRayHit);

    // Build our final volumes
    intersect.volumes.build(rtcRayHit.ray.tfar);

    // Setup the intersection
    if (rtcRayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
    {
        ray.hit = false;
        intersect.object = (ray.tmax == FLT_MAX) ? distant_lights : nullptr;
    }
    else
    {
        ray.t = rtcRayHit.ray.tfar;
        ray.tmax = rtcRayHit.ray.tfar;
        ray.hit = true;

        intersect.object = objects[rtcRayHit.hit.geomID];
        intersect.object->process_intersection(intersect.surface, rtcRayHit, ray);
    }

    return intersect;
}

void Scene::sample_lights(const Surface &surface, std::vector<LightSample> &light_samples, const float sample_multiplier, Resources &resources)
{
    for(size_t i = 0; i < lights.size(); i++)
    {
        // Make a better num_samples estimator
        const uint num_samples = std::max(1.f, SAMPLES_PER_SA * sample_multiplier);
        lights[i]->sample_light(num_samples, &surface.position, nullptr, light_samples, resources);
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
    distant_lights->add_light(light);
    add_light(light);
    return true;
}

bool Scene::add_texture(std::shared_ptr<Texture> texture)
{
    textures.push_back(texture);
    return true;
}
