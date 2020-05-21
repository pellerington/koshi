#pragma once

#include <embree/Embree.h>

#include <intersection/Intersector.h>

class EmbreeIntersector : public Intersector
{
public:
    EmbreeIntersector(Scene * scene);

    static void intersect_callback(const RTCFilterFunctionNArguments * args);
    IntersectList * intersect(const Ray& ray, const PathData * path, Resources& resources,
        PreIntersectionCallback * pre_intersect_callback = nullptr, void * pre_intersect_data = nullptr);

private:
    RTCScene rtc_scene;
    Integrator * default_integrator;

};
