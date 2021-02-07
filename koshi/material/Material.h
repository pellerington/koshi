#pragma once

#include <koshi/Koshi.h>
#include <koshi/Intersect.h>
#include <koshi/material/LobeArray.h>

KOSHI_OPEN_NAMESPACE

// TODO: Replace this with function/ptx generation during runtime, based on a materialx / xml file.

DEVICE_FUNCTION void generate_material(LobeArray& lobes, const Intersect& intersect)
{
    Lambert& lambert = lobes.push<Lambert>();
    GeometryMeshAttribute * color_attr = ((GeometryMesh *)intersect.geometry)->getAttribute("displayColor");
    lambert.color = (color_attr) ? (Vec3f)color_attr->evaluate(intersect) : 1.f;
    GeometryMeshAttribute * normal_attr = ((GeometryMesh *)intersect.geometry)->getAttribute("normals");
    lambert.normal = (normal_attr) ? intersect.obj_to_world.multiply<false>(normal_attr->evaluate(intersect)).normalize() : intersect.normal;
    lambert.world_transform = Transform::Local(lambert.normal);
}

KOSHI_CLOSE_NAMESPACE