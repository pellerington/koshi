#pragma once

#include <koshi/texture/Texture.h>
#include <koshi/intersection/Intersect.h>
#include <koshi/geometry/Geometry.h>

class TextureGeometryAttribute : public Texture
{
public:

    TextureGeometryAttribute(const std::string& attribute_name) : attribute_name(attribute_name) {}

    Vec3f evaluate(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) const
    {
        const GeometryAttribute * attribute = intersect->geometry->get_geometry_attribute(attribute_name);
        if(!attribute) return VEC3F_ZERO;
        return attribute->evaluate(u, v, w, intersect->geometry_primitive, resources);
    }

    Vec3f delta() const 
    { 
        return VEC3F_ZERO;
    }

    bool null() const 
    { 
        return false;
    };

private:
    const std::string attribute_name;
};
