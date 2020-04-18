#pragma once

#include <cfloat>
#include <unordered_set>
#include <Math/Types.h>
#include <Util/IorStack.h>
class Volume;

struct Ray
{
    Ray(const Vec3f &pos, const Vec3f &dir, const bool &camera = false, const IorStack * ior = nullptr)
    : pos(pos), dir(dir), inv_dir(1.f / dir), camera(camera), ior(ior) {}

    // Gets the position at t along the ray.
    inline Vec3f get_position(const float &_t) const { return pos + _t * dir; }

    // Basic info about our ray.
    const Vec3f pos;
    const Vec3f dir;
    const Vec3f inv_dir;

    // Variables. These can be altered afterwards.
    bool hit = false;
    float t = FLT_MAX;
    float tmin = 0.f;
    float tmax = FLT_MAX;

    // Optional info about our ray.
    const bool camera;
    const IorStack * ior;
    // const std::vector<Volume*> * in_volumes = nullptr;
};

// DEFINATLY MOVE THEsE SOMEWHERE ELSE!
inline bool intersect_bbox(const Ray &ray, const Box3f &box)
{
    const Vec3f t1 = (box.min() - ray.pos) * ray.inv_dir;
    const Vec3f t2 = (box.max() - ray.pos) * ray.inv_dir;
    const float tmin = Vec3f::min(t1, t2).max();
    const float tmax = Vec3f::max(t1, t2).min();

    if (tmax < 0 || tmin > tmax)
        return false;

    return true;
}

inline bool intersect_bbox(const Ray &ray, const Box3f &box, float &_tmin, float &_tmax)
{
    const Vec3f t1 = (box.min() - ray.pos) * ray.inv_dir;
    const Vec3f t2 = (box.max() - ray.pos) * ray.inv_dir;
    const float tmin = Vec3f::min(t1, t2).max();
    const float tmax = Vec3f::max(t1, t2).min();

    if (tmax < 0 || tmin > tmax)
        return false;

    _tmin = tmin;
    _tmax = tmax;

    return true;
}


/*
inline bool get_refation(const Surface &surface, const float &eta, Vec3f &out)
{
    //Dont use surface pass in normal wi and ior_in
    float n_dot_wi = clamp(surface.normal.dot(surface.wi), -1.f, 1.f);
    float k = 1.f - eta * eta * (1.f - n_dot_wi * n_dot_wi);
    if(k < 0) return false;

    return eta * surface.wi + (eta * fabs(n_dot_wi) - sqrtf(k)) * ((surface.front) ? surface.normal : -surface.normal);
}
*/
