#pragma once

#include <embree/Embree.h>
#include <geometry/GeometryVolume.h>
#include <geometry/Volume.h>
#include <intersection/IntersectHelpers.h>
#include <intersection/Intersect.h>

class EmbreeGeometryVolume : public EmbreeGeometry
{
public:
    EmbreeGeometryVolume(GeometryVolume * volume)
    {
        geometry = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(geometry, 1);
        rtcSetGeometryBoundsFunction(geometry, bounds_callback, volume);
        rtcSetGeometryIntersectFunction(geometry, intersect_callback);
    }

    static void bounds_callback(const RTCBoundsFunctionArguments * args)
    {
        Geometry * geometry = (Geometry*)args->geometryUserPtr;
        const Box3f& bbox = geometry->get_bbox();
        args->bounds_o->lower_x = bbox.min().x; args->bounds_o->upper_x = bbox.max().x;
        args->bounds_o->lower_y = bbox.min().y; args->bounds_o->upper_y = bbox.max().y;
        args->bounds_o->lower_z = bbox.min().z; args->bounds_o->upper_z = bbox.max().z;
    }

    static void intersect_callback(const RTCIntersectFunctionNArguments* args)
    {
        GeometryVolume * geometry = (GeometryVolume*)args->geometryUserPtr;
        EmbreeIntersectContext * context = (EmbreeIntersectContext*)args->context;
        args->valid[0] = 0;

        const Ray& ray = context->intersects->ray;

        const Ray obj_ray(geometry->get_world_to_obj().multiply(ray.pos, true), 
                          geometry->get_world_to_obj().multiply(ray.dir, false)); 

        float t0, t1;
        if(!intersect_bbox(obj_ray, geometry->get_obj_bbox(), t0, t1))
            return;

        Intersect * intersect = context->intersects->push(*context->resources);

        intersect->t = t0;
        intersect->t_len = t1 - t0;
        intersect->geometry = geometry;
        Volume * volume = context->resources->memory->create<Volume>();

        for(auto bound = geometry->get_bound().begin(); bound != geometry->get_bound().end(); ++bound)
        {
            if(intersect_bbox(obj_ray, bound->bbox, t0, t1))
            {
                Volume::Segment * segment = context->resources->memory->create<Volume::Segment>();
                segment->t0 = t0;
                segment->t1 = t1;
                segment->min_density = bound->min_density;
                segment->max_density = bound->max_density;

                //Find the position of the segment.
                Volume::Segment ** position = &volume->segment;
                while(*position && (*position)->t0 < t0)
                    position = &((*position)->next);
                segment->next = *position;
                *position = segment; 
            }
        }

        // volume->uvw0 = 
        // volume->uvw1 =

        // volume->material = 

        intersect->geometry_data = volume;

        intersect->integrator = geometry->get_attribute<Integrator>("integrator");

    }

};
