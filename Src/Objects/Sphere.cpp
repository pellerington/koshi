#include "Sphere.h"

#if !EMBREE

Sphere::Sphere(Vec3f position, float scale, std::shared_ptr<Material> material)
: Object(material), position(position), scale(scale), scale_sqr(scale*scale)
{
    bbox = Eigen::AlignedBox3f(position-Vec3f(scale, scale, scale), position+Vec3f(scale, scale, scale));
}

bool Sphere::intersect(Ray &ray, Surface &surface)
{
    float t0, t1;

    Vec3f L = position - ray.o;
    float tca = L.dot(ray.dir);
    // if (tca < 0) return false;
    float d2 = L.dot(L) - tca * tca;
    if (d2 > scale_sqr) return false;
    float thc = sqrt(scale_sqr - d2);
    t0 = tca - thc;
    t1 = tca + thc;

    if (t0 > t1)
        std::swap(t0, t1);

    if (t0 < 0)
    {
        t0 = t1;
        if (t0 < 0)
            return false;
    }

    float t = t0;
    Vec3f hit_position = ray.o + t * ray.dir;
    Vec3f normal = (hit_position - position).normalized();
    // ADD DOULE SIDED OPTION

    if (t < ray.t)
    {
        ray.t = t;
        ray.hit = true;
        surface.position = hit_position;
        surface.wi = ray.dir;
        surface.enter = normal.dot(ray.dir) < 0;
        surface.normal = normal;
        surface.u = 0.f;
        surface.v = 0.f;
        surface.object = this;
        return true;
    }

    return false;
}

#endif
