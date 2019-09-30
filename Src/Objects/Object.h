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
    const Box3f get_bbox() { return bbox; };

    virtual Surface process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray) = 0;
    uint attach_to_scene(RTCScene &rtc_scene)
    {
        rtcCommitGeometry(mesh);
        uint geomID = rtcAttachGeometry(rtc_scene, mesh);
        return geomID;
    }

    std::shared_ptr<Material> material;
    
protected:
    RTCGeometry mesh;
    Box3f bbox;

};
