#pragma once

#include <Math/Transform3f.h>
#include <intersection/Ray.h>

class Camera
{
public:
    Camera(const Transform3f& transform = Transform3f(), const Vec2u& resolution = Vec2u(0), const uint& samples_per_pixel = 1, const float& focal_length = 1.f);

    Vec2u get_image_resolution() const { return resolution; }
    Ray sample_pixel(const Vec2u &pixel, const Vec2f &rng) const;
    uint get_pixel_samples(const Vec2u &pixel) const { return samples_per_pixel; }
private:
    const Transform3f transform;
    const Vec3f origin;
    const Vec2u resolution;
    const uint samples_per_pixel;
    const float aspect_ratio;
    const float focal_length;
    const Vec3f pixel_delta;
};
