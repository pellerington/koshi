#pragma once

#include <embree/Embree.h>

#include <intersection/Intersector.h>

class EmbreeIntersector : public Intersector
{
public:
    EmbreeIntersector(ObjectGroup * objects);

    virtual Intersector * get_intersector(ObjectGroup * group);
    virtual Intersector * get_intersector(Geometry * geometry);

    static void intersect_callback(const RTCFilterFunctionNArguments * args);
    IntersectList * intersect(const Ray& ray, const PathData * path, Resources& resources, IntersectionCallbacks * callback = nullptr);

private:
    RTCScene rtc_scene;

    // TODO: Store geometry/scene and instance seperatly when we add instancing.
    struct EmbreeGeometryInstance {
        RTCGeometry geometry;
        RTCScene scene;
        RTCGeometry instance;
    };
    static std::unordered_map<Geometry*, EmbreeGeometryInstance> instances;

    static std::unordered_map<Object*, Intersector*> intersectors;
};
