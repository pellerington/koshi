#include <embree/EmbreeIntersector.h>
#include <embree/EmbreeGeometry.h>
#include <geometry/Geometry.h>
#include <intersection/Opacity.h>
#include <integrators/Integrator.h>
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
                rtcSetGeometryIntersectFilterFunction(geom, &EmbreeIntersector::intersect_callback);
                rtcCommitGeometry(geom);
                rtcAttachGeometry(rtc_scene, geom);
                rtcSetGeometryUserData(geom, geometry);
            }
        }
    }
    rtcSetSceneBuildQuality(rtc_scene, RTCBuildQuality::RTC_BUILD_QUALITY_HIGH);
    rtcCommitScene(rtc_scene);

    default_integrator = dynamic_cast<Integrator*>(scene->get_object("default_integrator"));
}

void EmbreeIntersector::intersect_callback(const RTCFilterFunctionNArguments * args)
{
    EmbreeIntersectContext * context = (EmbreeIntersectContext*)args->context;
    Geometry * geometry = (Geometry*)args->geometryUserPtr;
    const Ray& ray = context->intersects->ray;

    // Ignore if a duplicate
    for(uint i = 0; i < context->intersects->size(); i++)
    {
        Intersect * intersect = context->intersects->get(i);
        if(intersect->geometry == geometry && intersect->t == RTCRayN_tfar(args->ray, args->N, 0))
            return;
    }

    // Push intersect data
    Intersect * intersect = context->intersects->push(*context->resources);
    intersect->t = RTCRayN_tfar(args->ray, args->N, 0);
    intersect->t_len = 0;
    intersect->geometry = geometry;
    Surface * surface = context->resources->memory->create<Surface>(
        ray.get_position(intersect->t),
        Vec3f(RTCHitN_Ng_x(args->hit, args->N, 0), 
              RTCHitN_Ng_y(args->hit, args->N, 0), 
              RTCHitN_Ng_z(args->hit, args->N, 0)).normalized(),
        RTCHitN_u(args->hit, args->N, 0),
        RTCHitN_v(args->hit, args->N, 0),
        0.f,
        ray.dir
    );
    intersect->geometry_data = surface;

    // Get the opacity of the intersect
    Opacity * opacity = geometry->get_attribute<Opacity>("opacity");
    if(opacity) surface->set_opacity(opacity->get_opacity(surface->u, surface->v, 0.f, intersect, *context->resources));

    // Close any segments of this geometry
    if(!surface->facing)
        for(uint i = 0; i < context->intersects->size(); i++)
        {
            Intersect * intersect = context->intersects->get(i);
            if(intersect->geometry == geometry && intersect->t_len > 0.f)
                intersect->t_len = RTCRayN_tfar(args->ray, args->N, 0) - intersect->t;
        }

    // Is this hit solid?
    if(!(surface->opacity >= 1.f))
        args->valid[0] = 0;

    // Add an integrator
    intersect->integrator = geometry->get_attribute<Integrator>("integrator");
    if(!intersect->integrator)
        intersect->integrator = context->default_integrator;

    // TODO: Add a maximum/limit to the amount of intersections
}

IntersectList * EmbreeIntersector::intersect(const Ray& ray, const PathData * path, Resources& resources,
    PreIntersectionCallback * pre_intersect_callback, void * pre_intersect_data)
{
    IntersectList * intersects = resources.memory->create<IntersectList>(resources, ray, path);

    if(pre_intersect_callback)
        pre_intersect_callback(intersects, resources, pre_intersect_data);

    // Setup context
    EmbreeIntersectContext context;
    context.intersects = intersects;
    context.default_integrator = default_integrator;
    context.resources = &resources;
    RTCIntersectContext * rtc_context = &context;
    rtcInitIntersectContext(rtc_context);

    // Setup embree ray/hit
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

    // Perform null intersect callbacks
    if (rtcRayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
        null_intersection_callbacks(intersects, resources);

    // Finalize the intersection
    intersects->finalize(rtcRayHit.ray.tfar);

    return intersects;
}
