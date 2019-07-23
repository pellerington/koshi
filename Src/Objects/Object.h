#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <iostream>
#include <vector>

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
    void add_material(std::shared_ptr<Material> _material) { material = _material; };
    virtual bool intersect(Ray &ray, Surface &surface) = 0;
    const Eigen::AlignedBox3f get_bbox() { return bbox; };

    virtual std::vector<std::shared_ptr<Object>> get_sub_objects() = 0;
    /* virtual void apply_transform(Eigen::Affine3f transform) = 0; */

    std::shared_ptr<Material> material;
protected:
    Eigen::AlignedBox3f bbox;
};
