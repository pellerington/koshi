#pragma once

#include <koshi/base/ObjectGroup.h>
#include <koshi/intersection/Intersect.h>
#include <koshi/intersection/Ray.h>
#include <koshi/intersection/IntersectCallbacks.h>
class Scene;

class Intersector
{
public:
    Intersector(ObjectGroup * objects);

    // TODO: Do I like this method of getting specific intersectors?
    virtual Intersector * get_intersector(ObjectGroup * group) = 0;
    virtual Intersector * get_intersector(Geometry * geometry) = 0;

    // TODO: Add Limit to number of intersects.
    virtual IntersectList * intersect(const Ray& ray, const PathData * path, Resources& resouces, IntersectionCallbacks * callback = nullptr) = 0;

protected:
    std::vector<IntersectionCallbacks*> callbacks;
};