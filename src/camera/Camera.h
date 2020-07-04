#pragma once

#include <math/Transform3f.h>
#include <intersection/Ray.h>
#include <camera/PixelSampler.h>

class Camera
{
public:
    Camera(const Transform3f& transform, const Vec2u& resolution, const float& focal_length);

    Vec2u get_image_resolution() const { return resolution; }

    Ray sample_pixel(const uint& x, const uint& y, const float rng[2]) const;

private:
    const Transform3f transform;
    const Vec3f origin;
    const Vec2u resolution;
    const Vec2f inv_resolution;
    const float aspect_ratio;
    const float focal_length;
    const Vec2f pixel_delta;

    PixelSampleCallback * delta_sampler;
};
