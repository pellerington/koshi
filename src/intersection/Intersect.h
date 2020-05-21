#pragma once

#include <memory>
#include <vector>
#include <base/Object.h>
#include <Util/Resources.h>
#include <intersection/Ray.h>
#include <intersection/GeometrySurface.h>
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
    // InteriorMedium
    const PathData * prev_path;
};

class IntersectList;

// Intersect should be an array of hits
// Intersect should hold core details like Ray ect.
// Hits should store the actualy surface/geometry of all hits
// Use the [] operator to get a single hit.
// Maybe call them Intersect and IntersectList? Or do something clever where Intersect[i] returns an intersect where only i's things are acceible?
struct Intersect
{
    Intersect(const Ray& ray, const PathData * path = nullptr)
    : ray(ray), t(0.f), t_len(0.f), opacity(VEC3F_ONES), geometry(nullptr), integrator(nullptr), path(path), next(nullptr)
    {}

    const Ray ray;
    float t, t_len;
    Vec3f opacity;

    Geometry * geometry;
    GeometrySurface surface;
    // TODO: Make this a DATA type so we can have a GeometrySurface and GeometryVolume?
    // GeometryData * data;
    // <class T>
    // T * get_data() { return (T*)data; }

    Integrator * integrator;

    // TODO: Cleanup these.
    const PathData * path;
    Intersect * next;
};

class IntersectList
{
public:
    IntersectList(const Ray& ray, const PathData * path = nullptr)
    : ray(ray), intersect0(nullptr), num_intersects(0), path(path)
    {}

    const Ray ray;

    inline size_t size() const { return num_intersects; }
    inline bool empty() const { return !num_intersects; }
    inline bool hit() const { return num_intersects > 0; }

    inline Intersect * get(const size_t& index) 
    { 
        Intersect * intersect = intersect0;
        for(uint i = 0; i < index; i++)
            intersect = intersect->next;
        return intersect;
    }
    inline const Intersect * get(const size_t& index) const 
    { 
        Intersect * intersect = intersect0;
        for(uint i = 0; i < index; i++)
            intersect = intersect->next;
        return intersect;
    }

    inline Intersect * push(Resources& resources) 
    {
        num_intersects++;
        Intersect * intersect1 = intersect0;
        intersect0 = resources.memory.create<Intersect>(ray, path);
        intersect0->next = intersect1;
        return intersect0;
    }

    inline void pop() 
    {
        if(!num_intersects) return;
        intersect0 = intersect0->next;
        num_intersects--;
    }

    inline void finalize(const float& tmax)
    {
        Intersect ** intersect = &intersect0;
        while(*intersect)
        {
            if((*intersect)->t > tmax)
            {
                *intersect = (*intersect)->next;
                num_intersects--;
            }
            else
            {
                if((*intersect)->t + (*intersect)->t_len > tmax)
                    (*intersect)->t_len = tmax - (*intersect)->t;
                intersect = &((*intersect)->next);
            }
        }
    }


private:
    Intersect * intersect0;
    uint num_intersects;
    const PathData * path;
};

// TODO: Move intersect callbacks to thier own file.
// HitIntersectionCallback
typedef void (NullIntersectionCallback)(IntersectList * intersects, Geometry * geometry, Resources& resources);
struct IntersectionCallbacks : public Object
{
    NullIntersectionCallback * null_intersection_cb = nullptr;
};