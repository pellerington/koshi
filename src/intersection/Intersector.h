#pragma once

#include <intersection/Intersect.h>
#include <intersection/Ray.h>
#include <intersection/InteriorMedium.h>
class Scene;

typedef void (PreIntersectionCallback)(IntersectList * intersects, Resources& resources, void * data);

class Intersector
{
public:
    Intersector(Scene * scene);

    virtual IntersectList * intersect(const Ray& ray, const PathData * path, Resources& resouces, 
        PreIntersectionCallback * pre_intersect_callback = nullptr, void * pre_intersect_data = nullptr) = 0;
    
    void null_intersection_callbacks(IntersectList * intersects, Resources& resources)
    {
        for(size_t i = 0; i < null_callbacks.size(); i++)
            null_callbacks[i].first(intersects, null_callbacks[i].second, resources);
    }

protected:
    Scene * scene;

    std::vector<std::pair<NullIntersectionCallback*, Geometry*>> null_callbacks;
};