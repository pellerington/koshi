#pragma once

#include <memory>
#include <vector>
#include <koshi/base/Object.h>
#include <koshi/base/Resources.h>
#include <koshi/base/Array.h>
#include <koshi/intersection/Ray.h>
#include <koshi/geometry/Surface.h>
class Geometry;
class Integrator;

// TODO: Template max intersects in the future.
#define MAX_INTERSECTS 16

// Todo: Rename pathData to something more general.
// Todo: make this more user friendly and configurable. Let people attach arbitary data to it.
// Todo: move into it's own file.
struct PathData
{
    uint depth;
    float quality;
    // LPE
    // Interiors
    const PathData * prev_path;
};

class IntersectList;

struct Intersect
{
    Intersect(const Ray& ray, const PathData * path = nullptr)
    : ray(ray), t(0.f), tlen(0.f), interior(false), geometry(nullptr), geometry_data(nullptr), geometry_primitive(0), integrator(nullptr), path(path)
    {}

    const Ray ray;
    float t, tlen;
    bool interior;

    // std::string hit_type (Surface || SurfaceDistant || Volume) this could help avoid dynamic_cast
    Geometry * geometry;
    GeometryData * geometry_data;
    uint geometry_primitive;

    Integrator * integrator;

    // TODO: Cleanup this.
    const PathData * path;
};

class IntersectList
{
public:
    IntersectList(Resources& resources, const Ray& ray, const PathData * path = nullptr)
    : ray(ray), path(path), tend(FLT_MAX), intersects(resources.memory, 4)
    {}

    const Ray ray;
    const PathData * path;

    inline size_t size() const { return intersects.size(); }
    inline bool empty() const { return !intersects.size(); }
    inline bool hit() const { return intersects.size() > 0; }

    inline Intersect * get(const uint& i) { return intersects[i]; }
    inline const Intersect * get(const uint& i) const { return intersects[i]; }
    inline Intersect * get_front()
    {
        if(empty()) return nullptr;
        uint index = 0; float t = FLT_MAX;
        for(uint i = 0; i < intersects.size(); i++)
            if(intersects[i]->t < t)
            {
                index = i;
                t = intersects[i]->t;
            }
        return intersects[index];
    }

    inline Intersect * push(Resources& resources) 
    {
        Intersect * intersect = resources.memory->create<Intersect>(ray, path);
        intersects.push(intersect);
        return intersect;
    }

    inline uint pop(const uint& i) 
    {
        intersects[i] = intersects[intersects.size() - 1];
        intersects.resize(intersects.size() - 1);
        return i;
    }

    float tend;

private:
    Array<Intersect*> intersects;
};