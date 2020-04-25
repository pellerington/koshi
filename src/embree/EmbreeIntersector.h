#pragma once

#include <embree/Embree.h>

#include <intersection/Intersector.h>

class EmbreeIntersector : public Intersector
{
public:
    EmbreeIntersector(Scene * scene);

    IntersectList intersect(const Ray &ray, const PathData * path = nullptr);

private:
    RTCScene rtc_scene;

};
