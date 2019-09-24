#include "Triangle.h"

#if !EMBREE

#include <iostream>
#include <cfloat>

Triangle::Triangle(std::shared_ptr<Vec3f> v0, std::shared_ptr<Vec3f> v1, std::shared_ptr<Vec3f> v2,
                   std::shared_ptr<Vec3f> n0, std::shared_ptr<Vec3f> n1, std::shared_ptr<Vec3f> n2,
                   std::shared_ptr<Material> material)
: Object(material), vertices({v0, v1, v2}), normals({n0, n1, n2})
{
    init();
}

Triangle::Triangle(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2, std::shared_ptr<Material> material)
: Object(material),
  vertices({std::make_shared<Vec3f>(v0), std::make_shared<Vec3f>(v1), std::make_shared<Vec3f>(v2)}),
  normals({nullptr, nullptr, nullptr})
{
    init();
}


Triangle::Triangle(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2, const Vec3f &n0, const Vec3f &n1, const Vec3f &n2, std::shared_ptr<Material> material)
: Object(material),
  vertices({std::make_shared<Vec3f>(v0), std::make_shared<Vec3f>(v1), std::make_shared<Vec3f>(v2)}),
  normals({std::make_shared<Vec3f>(n0), std::make_shared<Vec3f>(n1), std::make_shared<Vec3f>(n2)})
{
    init();
}

void Triangle::init()
{
    bbox = Box3f(Vec3f::min(*vertices[0], Vec3f::min(*vertices[1], *vertices[2])),
                 Vec3f::max(*vertices[0], Vec3f::max(*vertices[1], *vertices[2])));

    normal = ((*vertices[0] - *vertices[1]).cross(*vertices[0] - *vertices[2])).normalized();
    smooth_normals =  normals[0] && normals[1] && normals[2];
}

bool Triangle::intersect(Ray &ray, Surface &surface)
{
    Vec3f edge1, edge2, h, s, q;
    float a,f,u,v;
    edge1 = *vertices[1] - *vertices[0];
    edge2 = *vertices[2] - *vertices[0];
    h = ray.dir.cross(edge2);
    a = edge1.dot(h);
    if (a > -EPSILON_F && a < EPSILON_F)
        return false;
    f = 1.0f/a;
    s = ray.pos - *vertices[0];
    u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f)
        return false;
    q = s.cross(edge1);
    v = f * ray.dir.dot(q);
    if (v < 0.0f || u + v > 1.0f)
        return false;
    const float t = f * edge2.dot(q);

    if (t < ray.t)
    {
        ray.t = t;
        ray.hit = true;
        surface.position = ray.get_point(t);
        surface.wi = ray.dir;
        surface.enter = normal.dot(ray.dir) < 0;
        surface.normal = (smooth_normals) ? (1.f - u - v) * *normals[0] + u * *normals[1] + v * *normals[2] : normal;

        // TODO: Remap to texture space? or do later
        surface.u = u;
        surface.v = v;

        surface.object = this;
        return true;
    }
    else
        return false;
}

#endif
