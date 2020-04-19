#include <embree/EmbreeIntersector.h>
#include <embree/EmbreeGeometry.h>

#include <Scene/Scene.h>

void EmbreeIntersector::pre_render()
{
    // Attach the intersectors here, but find a way to attach default attributes in the future.

    // Build the scene
    rtc_scene = rtcNewScene(Embree::rtc_device);
    std::vector<std::shared_ptr<Geometry>>& objects = scene->get_objects();
    for(size_t i = 0; i < objects.size(); i++)
    {
        EmbreeGeometry * embree_geometry = objects[i]->get_attribute<EmbreeGeometry>("embree_geometry");
        if(embree_geometry)
        {
            // objects[i]->set_id(i);
            RTCGeometry geom = embree_geometry->get_rtc_geometry();
            // if(objects[i]->use_intersection_filter())
            //     rtcSetGeometryIntersectFilterFunction(geom, &Scene::intersection_callback);
            rtcCommitGeometry(geom);
            rtcAttachGeometryByID(rtc_scene, geom, i);
            rtcSetGeometryUserData(geom, objects[i].get());
        }
    }
    rtcSetSceneBuildQuality(rtc_scene, RTCBuildQuality::RTC_BUILD_QUALITY_HIGH);
    rtcCommitScene(rtc_scene);
}

// void Scene::intersection_callback(const RTCFilterFunctionNArguments * args)
// {
//     Object * obj = (Object*)args->geometryUserPtr;
//     obj->process_intersection_visibility(args);
//     if(obj->volume)
//         obj->process_intersection_volume(args);
// }

Intersect EmbreeIntersector::intersect(Ray &ray)
{
    Intersect intersect(ray);

    EmbreeIntersectContext context;
    context.ray = &ray;
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

    // Setup the intersection
    if (rtcRayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
    {
        ray.hit = false;
        // intersect.geometry = (ray.tmax == FLT_MAX) ? scene->get_distant_lights().get() : nullptr;
        // End hit callbacks.
    }
    else
    {
        ray.t = rtcRayHit.ray.tfar;
        ray.tmax = rtcRayHit.ray.tfar;
        ray.hit = true;

        intersect.geometry = (Geometry*)rtcGetGeometryUserData(rtcGetGeometry(rtc_scene, rtcRayHit.hit.geomID));
        intersect.surface.set_hit
        (
            ray.get_position(ray.t),
            Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized(),
            Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized(),
            rtcRayHit.hit.u,
            rtcRayHit.hit.v
        );
    }

    return intersect;
}
