#pragma once

#include <memory>
#include <iostream>
#include <vector>

#include "../Scene/Embree.h"
#include "../Materials/Material.h"
#include "../Util/Surface.h"
#include "../Util/Ray.h"
class Surface;
class Material;

enum class ObjectType
{
    Triangle,
    Mesh,
    Sphere
};

class Object
{
public:
    Object() : material(nullptr) {}
    Object(std::shared_ptr<Material> material) : material(material) {}
    virtual ObjectType get_type() = 0;
    virtual std::vector<std::shared_ptr<Object>> get_sub_objects() = 0;
    virtual bool intersect(Ray &ray, Surface &surface) = 0;
    const Eigen::AlignedBox3f get_bbox() { return bbox; };

#if EMBREE
    virtual void process_intersection(RTCRayHit &rtcRayHit, Ray &ray, Surface &surface) = 0;
    uint attach_to_scene(RTCScene &rtc_scene)
    {
        rtcCommitGeometry(mesh);
        uint geomID = rtcAttachGeometry(rtc_scene, mesh);
        return geomID;
    }
#endif

    std::shared_ptr<Material> material;
protected:

#if EMBREE
    RTCGeometry mesh;
#endif

    Eigen::AlignedBox3f bbox;

};
