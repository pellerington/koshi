// TODO: Fill these in with sphere ect, AND use them later.

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

inline bool intersect_bbox(const Ray &ray, const Box3f &box, float &t0, float &t1)
{
    const Vec3f t1 = (box.min() - ray.pos) * ray.inv_dir;
    const Vec3f t2 = (box.max() - ray.pos) * ray.inv_dir;
    const float tmin = Vec3f::min(t1, t2).max();
    const float tmax = Vec3f::max(t1, t2).min();

    if (tmax < 0 || tmin > tmax)
        return false;

    t0 = tmin;
    t1 = tmax;

    return true;
}