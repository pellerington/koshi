#pragma once

#include <memory>
#include <vector>
#include <base/Object.h>
#include <Util/Resources.h>
#include <intersection/Ray.h>
#include <intersection/GeometrySurface.h>
class Geometry;

// TODO: Template max intersects in the future.
#define MAX_INTERSECTS 8

// Todo: make this more user friendly and configurable. Let people attach arbitary data to it.
// Todo: move into it's own file.
struct PathData
{
    uint depth;
    double quality;
    // LPE
    // IorStack
    const PathData * prev_path;
};

// Intersect should be an array of hits
// Intersect should hold core details like Ray ect.
// Hits should store the actualy surface/geometry of all hits
// Use the [] operator to get a single hit.
// Maybe call them Intersect and IntersectList? Or do something clever where Intersect[i] returns an intersect where only i's things are acceible?
struct Intersect
{
    Intersect(const Ray& ray, const PathData * path = nullptr)
    : ray(ray), t(0.f), t_len(0.f), geometry(nullptr), path(path)
    {}

    const Ray ray;
    float t, t_len;

    Geometry * geometry;
    GeometrySurface surface;
    // TODO: Make this a DATA type so we can have a GeometrySurface and GeometryVolume?
    // GeometryData * data;
    // <class T>
    // T * get_data() { return (T*)data; }

    const PathData * path;
};

class IntersectList
{
public:
    IntersectList(const Ray& ray, const PathData * path = nullptr)
    : ray(ray), path(path)
    {}

    const Ray ray;

    inline size_t size() const { return num_intersects; }
    inline bool empty() const { return !num_intersects; }
    inline bool hit() const { return num_intersects > 0; }

    inline Intersect * operator[](const size_t& i) { return intersects[i]; }
    inline const Intersect * operator[](const size_t& i) const { return intersects[i]; }
    inline Intersect * get(const size_t& i) { return intersects[i]; }
    inline const Intersect * get(const size_t& i) const { return intersects[i]; }

    inline Intersect * push(Resources& resources) {
        if(num_intersects == MAX_INTERSECTS)
            return nullptr;
        return intersects[num_intersects++] = resources.memory.create<Intersect>(ray, path);
    }

private:
    Intersect * intersects[MAX_INTERSECTS];
    uint num_intersects = 0;
    const PathData * path;
};

typedef void (NullIntersectionCallback)(IntersectList * intersects, Geometry * geometry, Resources& resources);
struct IntersectionCallbacks : public Object
{
    NullIntersectionCallback * null_intersection_cb = nullptr;
};