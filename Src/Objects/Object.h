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

class Object
{
public:
    Object() : material(nullptr), obj_to_world(Transform3f()), world_to_obj(Transform3f()) {}
    Object(std::shared_ptr<Material> material, const Transform3f &obj_to_world)
    : material(material), obj_to_world(obj_to_world), world_to_obj(Transform3f::inverse(obj_to_world)) {}

    enum Type
    {
        Triangle,
        Mesh,
        Sphere,
        Box
        // then attach volume properties to stuff!
        // If stuff has no ptr to volprop and material is a surface
        // If it has volprop and material is vol but not added to vol structure
        // If it has volprop and no material add it to vol strucutre
        // Volprop, Homogenous/SingleScatter(Min,Max,Delta)/MultiScatter

    };
    virtual Type get_type() = 0;

    const Box3f get_bbox() { return bbox; };

    virtual Surface process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray) = 0;
    uint attach_to_scene(RTCScene &rtc_scene)
    {
        rtcCommitGeometry(mesh);
        uint geomID = rtcAttachGeometry(rtc_scene, mesh);
        return geomID;
    }

    std::shared_ptr<Material> material; // This should be const
    const Transform3f obj_to_world;
    const Transform3f world_to_obj;

protected:
    RTCGeometry mesh;
    Box3f bbox;
};
