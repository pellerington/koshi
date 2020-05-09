#include <embree/EmbreeIntersector.h>
#include <embree/EmbreeGeometry.h>
#include <geometry/Geometry.h>
#include <base/Scene.h>

EmbreeIntersector::EmbreeIntersector(Scene * scene) : Intersector(scene)
{
    // Build the scene
    rtc_scene = rtcNewScene(Embree::rtc_device);
    for(auto object = scene->begin(); object != scene->end(); ++object)
    {
        Geometry * geometry = dynamic_cast<Geometry*>(object->second);
        if(geometry)
        {
            EmbreeGeometry * embree_geometry = geometry->get_attribute<EmbreeGeometry>("embree_geometry");
            if(embree_geometry)
            {
                RTCGeometry geom = embree_geometry->get_rtc_geometry();
                // if(objects[i]->use_intersection_filter())
                //     rtcSetGeometryIntersectFilterFunction(geom, &Scene::intersection_callback);
                rtcCommitGeometry(geom);
                rtcAttachGeometry(rtc_scene, geom);
                rtcSetGeometryUserData(geom, geometry);
            }
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

IntersectList * EmbreeIntersector::intersect(const Ray& ray, const PathData * path, Resources& resources)
{
    IntersectList * intersects = resources.memory.create<IntersectList>(ray, path);

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

    // Finalize the intersection
    if (rtcRayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
    {
        null_intersection_callbacks(intersects, resources);
    }
    else
    {
        Intersect * intersect = intersects->push(resources);
        intersect->geometry = (Geometry*)rtcGetGeometryUserData(rtcGetGeometry(rtc_scene, rtcRayHit.hit.geomID));
        intersect->surface.set
        (
            ray.get_position(rtcRayHit.ray.tfar),
            Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized(),
            rtcRayHit.hit.u,
            rtcRayHit.hit.v,
            ray.dir
        );
    }

    return intersects;
}
