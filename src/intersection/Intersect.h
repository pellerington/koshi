#pragma once

#include <memory>
#include <vector>
#include <base/Object.h>
#include <intersection/Ray.h>
#include <Util/Surface.h>
class Geometry;

// Intersect should be an array of hits
// Intersect should hold core details like Ray ect.
// Hits should store the actualy surface/geometry of all hits
// Use the [] operator to get a single hit.
// Maybe call them Intersect and IntersectList? Or do something clever where Intersect[i] returns an intersect where only i's things are acceible?
struct Intersect
{
    Intersect(Ray& ray)
    : ray(ray), geometry(nullptr),
    surface(ray.dir, ray.ior->get_curr_ior(), ray.ior->get_prev_ior()),
    ior(ray.ior)
    {}

    Ray& ray;

    Geometry * geometry;

    Surface surface;
    const IorStack * ior;
};

typedef void (NullIntersectionCallback)(Intersect& intersect, Geometry * geometry);

struct IntersectionCallbacks : public Object
{
    NullIntersectionCallback * null_intersection_cb = nullptr;
};