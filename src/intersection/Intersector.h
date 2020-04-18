#pragma once

#include <intersection/Intersect.h>
#include <intersection/Ray.h>
class Scene;

class Intersector
{
public:
    Intersector(Scene * scene) : scene(scene) {}

    virtual void pre_render() = 0;

    virtual Intersect intersect(Ray& ray) = 0;
    
protected:
    Scene * scene;
};