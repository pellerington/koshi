#pragma once

#include <koshi/Koshi.h>
#include <koshi/Intersect.h>
#include <koshi/material/LobeArray.h>

KOSHI_OPEN_NAMESPACE

// TODO: Replace this with function/ptx generation during runtime, based on a materialx / xml file.

DEVICE_FUNCTION void generate_material(LobeArray& lobes, const Intersect& intersect, const Ray& ray)
{
    GeometryMeshAttribute * normal_attr = ((GeometryMesh *)intersect.geometry)->getAttribute("normals");
    const Vec3f normal = (normal_attr) ? intersect.obj_to_world.multiply<false>(normal_attr->evaluate(intersect)).normalize() : intersect.normal;

    // frenel reflection amount
    const float r0 = (1.5f - 1.f) / (1.5f + 1.f);
    const float r = r0 + (1.f - r0) * pow((1.f - abs(normal.dot(ray.direction))), 5);

    Reflect& reflect = lobes.push<Reflect>();
    reflect.normal = normal;
    reflect.color = r;
    // TODO: Setting reflect.color to 1 makes it black...

    Lambert& lambert = lobes.push<Lambert>();
    GeometryMeshAttribute * color_attr = ((GeometryMesh *)intersect.geometry)->getAttribute("displayColor");
    lambert.color = (color_attr) ? (Vec3f)color_attr->evaluate(intersect) : 1.f;
    lambert.color *= (1.f - r);
    lambert.normal = normal;
    lambert.world_transform = Transform::Local(lambert.normal);
}

KOSHI_CLOSE_NAMESPACE