#include <koshi/embree/EmbreeIntersector.h>
#include <koshi/embree/EmbreeGeometry.h>
#include <koshi/geometry/Geometry.h>
#include <koshi/integrator/Integrator.h>
#include <koshi/material/Material.h>
#include <koshi/base/Scene.h>

std::unordered_map<Geometry*, EmbreeIntersector::EmbreeGeometryInstance> EmbreeIntersector::instances = std::unordered_map<Geometry*, EmbreeGeometryInstance>();
std::unordered_map<Object*, Intersector*> EmbreeIntersector::intersectors = std::unordered_map<Object*, Intersector*>();

EmbreeIntersector::EmbreeIntersector(ObjectGroup * objects) : Intersector(objects)
{
    // Build the scene
    rtc_scene = rtcNewScene(Embree::rtc_device);
    for(uint i = 0; i < objects->size(); i++)
    {
        Geometry * geometry = objects->get<Geometry>(i);
        if(geometry)
        {
            if(instances.find(geometry) != instances.end())
            {
                rtcAttachGeometry(rtc_scene, instances[geometry].instance);
            }
            else
            {
                EmbreeGeometry * embree_geometry = geometry->get_attribute<EmbreeGeometry>("embree_geometry");
                if(embree_geometry)
                {
                    instances[geometry].geometry = embree_geometry->get_rtc_geometry();
                    rtcSetGeometryIntersectFilterFunction(instances[geometry].geometry, &EmbreeIntersector::intersect_callback);
                    rtcSetGeometryUserData(instances[geometry].geometry, geometry);
                    rtcCommitGeometry(instances[geometry].geometry);

                    instances[geometry].scene = rtcNewScene(Embree::rtc_device);
                    rtcAttachGeometry(instances[geometry].scene, instances[geometry].geometry); 
                    rtcSetSceneBuildQuality(instances[geometry].scene, RTCBuildQuality::RTC_BUILD_QUALITY_HIGH);
                    rtcCommitScene(instances[geometry].scene);

                    // TODO: Do instanceing ourselves so we can load in and out on demand.
                    instances[geometry].instance = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_INSTANCE);
                    const float * transform = geometry->get_obj_to_world().get_array();
                    rtcSetGeometryTransform(instances[geometry].instance, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, transform);
                    rtcSetGeometryInstancedScene(instances[geometry].instance, instances[geometry].scene);
                    // TODO: Add a seperate object intersect filter.
                    //rtcSetGeometryIntersectFilterFunction(geom, &EmbreeIntersector::intersect_callback);
                    rtcSetGeometryUserData(instances[geometry].instance, geometry); 
                    rtcCommitGeometry(instances[geometry].instance);

                    rtcAttachGeometry(rtc_scene, instances[geometry].instance);
                }
            }
        }
    }
    rtcSetSceneBuildQuality(rtc_scene, RTCBuildQuality::RTC_BUILD_QUALITY_HIGH);
    rtcCommitScene(rtc_scene);

    intersectors[objects] = this;
}

Intersector * EmbreeIntersector::get_intersector(ObjectGroup * group)
{
    auto cached_intersector = intersectors.find(group);
    if(cached_intersector != intersectors.end())
        return cached_intersector->second;
    return new EmbreeIntersector(group);
}

Intersector * EmbreeIntersector::get_intersector(Geometry * geometry)
{
    auto cached_intersector = intersectors.find(geometry);
    if(cached_intersector != intersectors.end())
        return cached_intersector->second;
    ObjectGroup group;
    group.push(geometry);
    Intersector * intersector = new EmbreeIntersector(&group);
    intersectors.erase(&group);
    intersectors[geometry] = intersector;
    return intersector;
}

// void EmbreeIntersector::object_callback(const RTCFilterFunctionNArguments * args)
// {

// }

void EmbreeIntersector::intersect_callback(const RTCFilterFunctionNArguments * args)
{
    EmbreeIntersectContext * context = (EmbreeIntersectContext*)args->context;
    Geometry * geometry = (Geometry*)args->geometryUserPtr;
    const Ray& ray = context->intersects->ray;

    args->valid[0] = 0;

    // TODO: Do this in an object callback on the intersector
    // Check our visibility.
    if(context->intersects->path->depth == 0 && geometry->get_visibility().hide_camera)
        return;

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
    intersect->tlen = 0;
    intersect->geometry = geometry;
    intersect->geometry_primitive = RTCHitN_primID(args->hit, args->N, 0);
    Vec3f normal = geometry->get_obj_to_world().multiply(Embree::normal(args), false).normalized();
    Surface * surface = context->resources->memory->create<Surface>(
        ray.get_position(intersect->t),
        normal,
        RTCHitN_u(args->hit, args->N, 0),
        RTCHitN_v(args->hit, args->N, 0),
        0.f,
        ray.dir.dot(normal) < 0.f
    );
    intersect->geometry_data = surface;

    // Add the material to the surface.
    surface->material = geometry->get_attribute<Material>("material");
    surface->opacity = (surface->material) ? surface->material->opacity(surface->u, surface->v, surface->w, intersect, *context->resources) : VEC3F_ONES;

    // Close any segments of this geometry
    if(!surface->facing)
    {
        for(uint i = 0; i < context->intersects->size(); i++)
        {
            Intersect * intersect = context->intersects->get(i);
            if(intersect->geometry == geometry && intersect->tlen > 0.f)
                intersect->tlen = RTCRayN_tfar(args->ray, args->N, 0) - intersect->t;
        }
    }

    // Check hit is solid
    if(surface->opacity >= 1.f)
        args->valid[0] = -1;

    // Add an integrator
    intersect->integrator = geometry->get_attribute<Integrator>("integrator");
    if(!intersect->integrator)
        intersect->integrator = context->resources->scene->get_object<Integrator>("default_integrator");
}

IntersectList * EmbreeIntersector::intersect(const Ray& ray, const PathData * path, Resources& resources, IntersectionCallbacks * callback)
{
    // Setup intersect list
    IntersectList * intersects = resources.memory->create<IntersectList>(resources, ray, path);

    // Call preintersection callbacks.
    if(callback && callback->pre_intersection_cb)
        callback->pre_intersection_cb(intersects, callback->pre_intersection_data, resources);
    for(uint i = 0; i < callbacks.size(); i++)
        if(callbacks[i]->pre_intersection_cb)
            callbacks[i]->pre_intersection_cb(intersects, callbacks[i]->pre_intersection_data, resources);

    // Setup context
    EmbreeIntersectContext context;
    context.intersects = intersects;
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

    // Finalize the intersection
    intersects->tend = rtcRayHit.ray.tfar;
    for(uint i = 0; i < intersects->size(); i++)
    {
        Intersect * intersect = intersects->get(i);
        if(intersect->t > intersects->tend)
            i = intersects->pop(i) - 1;
        else if(intersect->t + intersect->tlen > intersects->tend)
            intersect->tlen = intersects->tend - intersect->t;
    }

    // Perform post intersect callbacks
    if(callback && callback->post_intersection_cb)
        callback->post_intersection_cb(intersects, callback->post_intersection_data, resources);
    for(uint i = 0; i < callbacks.size(); i++)
        if(callbacks[i]->post_intersection_cb)
            callbacks[i]->post_intersection_cb(intersects, callbacks[i]->post_intersection_data, resources);

    return intersects;
}
