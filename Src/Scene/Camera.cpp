#include "Camera.h"

#include <iostream>

Camera::Camera(Eigen::Affine3f transform, Vec2i resolution, uint samples_per_pixel, float focal_length)
: transform(transform), resolution(resolution), samples_per_pixel(samples_per_pixel), aspect_ratio((float) resolution[0] / resolution[1]), focal_length(focal_length)
{
}

bool Camera::sample_pixel(const Vec2i &pixel, Ray &ray, const Vec2f * rng) const
{
    // Are we out of bounds?
    if(pixel[0] >= resolution[0] || pixel[1] >= resolution[1] || pixel[0] < 0 || pixel[1] < 0)
        return false;

    // Find size of pixel and end position
    Vec3f pixel_delta((-1.f / resolution.x()) * aspect_ratio, -1.f / resolution.y(), 0);
    Vec3f pixel_position(((float)(resolution.x() - pixel.x()) / resolution.x() - 0.5f) * aspect_ratio, ((float)(resolution.y() - pixel.y()) / resolution.y() - 0.5f), focal_length);

    // Set ray
    pixel_position = transform * (pixel_position + pixel_delta * ((rng == nullptr) ? Vec3f(RNG::Rand(), RNG::Rand(), 0) : Vec3f((*rng)[0], (*rng)[1], 0)));
    ray.o = transform.translation();
    ray.dir = (pixel_position - ray.o).normalized();

    return true;
}
