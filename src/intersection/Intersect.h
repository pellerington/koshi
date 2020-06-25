#pragma once

#include <memory>
#include <vector>
#include <base/Object.h>
#include <Util/Resources.h>
#include <Util/Array.h>
#include <intersection/Ray.h>
#include <geometry/Surface.h>
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
    double quality;
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
    : ray(ray), intersects(resources.memory), path(path)
    {}

    const Ray ray;

    inline size_t size() const { return intersects.size(); }
    inline bool empty() const { return !intersects.size(); }
    inline bool hit() const { return intersects.size() > 0; }

    inline Intersect * get(const uint& i) { return intersects[i]; }
    inline const Intersect * get(const uint& i) const { return intersects[i]; }

    inline Intersect * push(Resources& resources) 
    {
        Intersect * intersect = resources.memory->create<Intersect>(ray, path);
        intersects.push(intersect);
        return intersect;
    }

    inline void pop() 
    {
        if(!intersects.size()) return;
        intersects.resize(intersects.size() - 1);
    }

    inline void finalize(const float& tmax)
    {
        uint curr_size = intersects.size();
        for(uint i = 0; i < curr_size;)
        {
            if(intersects[i]->t > tmax)
            {
                curr_size--;
                if(curr_size) intersects[i] = intersects[curr_size];
                intersects.resize(curr_size);
            }
            else
            {
                if(intersects[i]->t + intersects[i]->tlen > tmax)
                    intersects[i]->tlen = tmax - intersects[i]->t;
                i++;
            }
        }
    }


private:
    Array<Intersect*> intersects;
    const PathData * path;
};

// TODO: Move intersect callbacks to thier own file.
// HitIntersectionCallback
typedef void (NullIntersectionCallback)(IntersectList * intersects, Geometry * geometry, Resources& resources);
struct IntersectionCallbacks : public Object
{
    NullIntersectionCallback * null_intersection_cb = nullptr;
};