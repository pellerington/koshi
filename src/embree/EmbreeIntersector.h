#pragma once

#include <embree/Embree.h>

#include <intersection/Intersector.h>

class EmbreeIntersector : public Intersector
{
public:

    EmbreeIntersector(Scene * scene) : Intersector(scene) {}

    void pre_render();

    Intersect intersect(Ray &ray);

private:
    RTCScene rtc_scene;

};
