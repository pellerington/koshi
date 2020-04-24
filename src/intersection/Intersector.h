#pragma once

#include <intersection/Intersect.h>
#include <intersection/Ray.h>
class Scene;

class Intersector
{
public:
    Intersector(Scene * scene);

    virtual void pre_render() = 0;

    virtual IntersectList intersect(const Ray& ray, const PathData * path = nullptr) = 0;
    
    void null_intersection_callbacks(IntersectList& intersects)
    {
        for(size_t i = 0; i < null_callbacks.size(); i++)
            null_callbacks[i].first(intersects, null_callbacks[i].second);
    }

protected:
    Scene * scene;

    std::vector<std::pair<NullIntersectionCallback*, Geometry*>> null_callbacks;
};