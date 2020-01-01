#include "Camera.h"

#include <iostream>

Camera::Camera(const Transform3f &transform, const Vec2u &resolution, const uint &samples_per_pixel, const float &focal_length)
: transform(transform), origin(transform * Vec3f(0.f, 0.f, 0.f, 1.f)), resolution(resolution), samples_per_pixel(samples_per_pixel)
, aspect_ratio((float) resolution.x / resolution.y), focal_length(focal_length), pixel_delta(-1.f / resolution.x * aspect_ratio, -1.f / resolution.y, 0)
{
}

Ray Camera::sample_pixel(const Vec2u &pixel, const Vec2f &rng) const
{
    // Set ray
    Vec3f pixel_position(((float)(resolution.x - pixel.x) / resolution.x - 0.5f) * aspect_ratio, ((float)(resolution.y - pixel.y) / resolution.y - 0.5f), focal_length);
    pixel_position = pixel_position + pixel_delta * Vec3f(rng.x, rng.y, 0.f);
    pixel_position = transform * pixel_position;
    return Ray(origin, (pixel_position - origin).normalized(), true);
}
