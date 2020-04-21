#pragma once

#include <memory>
#include <vector>
#include <base/Object.h>
#include <intersection/Ray.h>
#include <intersection/GeometrySurface.h>
class Geometry;

// Intersect should be an array of hits
// Intersect should hold core details like Ray ect.
// Hits should store the actualy surface/geometry of all hits
// Use the [] operator to get a single hit.
// Maybe call them Intersect and IntersectList? Or do something clever where Intersect[i] returns an intersect where only i's things are acceible?
struct Intersect
{
    Intersect(const Ray& _ray)
    : ray(_ray), t(0.f), t_len(0.f), geometry(nullptr)
    {}

    const Ray ray;
    float t, t_len;

    Geometry * geometry;
    GeometrySurface surface;
    // TODO: Make this a DATA type so we can have a GeometrySurface and GeometryVolume?
    // GeometryData * data;
    // <class T>
    // T * get_data() { return (T*)data; }
};

class IntersectList
{
public:
    IntersectList(const Ray& _ray)
    : ray(_ray)
    {}

    const Ray ray;
    inline size_t size() const { return intersects.size(); }
    inline bool hit() const { return !intersects.empty(); }
    inline bool empty() const { return intersects.empty(); }

    // TODO: Currently this is not safe is we resize vector. Use a linked list or something instead.
    inline Intersect& operator[](const size_t& i) { return intersects[i]; }
    inline const Intersect& operator[](const size_t& i) const { return intersects[i]; }
    Intersect& push() {
        intersects.push_back(Intersect(ray)); 
        return intersects.back();
    }

private:
    std::vector<Intersect> intersects;
};

typedef void (NullIntersectionCallback)(IntersectList& intersects, Geometry * geometry);
struct IntersectionCallbacks : public Object
{
    NullIntersectionCallback * null_intersection_cb = nullptr;
};