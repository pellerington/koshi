#pragma once

#include <intersection/Intersect.h>
#include <intersection/Ray.h>
class Scene;

class Intersector
{
public:
    Intersector(Scene * scene);

    virtual IntersectList * intersect(const Ray& ray, const PathData * path, Resources& resouces) = 0;
    
    void null_intersection_callbacks(IntersectList * intersects, Resources& resources)
    {
        for(size_t i = 0; i < null_callbacks.size(); i++)
            null_callbacks[i].first(intersects, null_callbacks[i].second, resources);
    }

protected:
    Scene * scene;

    std::vector<std::pair<NullIntersectionCallback*, Geometry*>> null_callbacks;
};