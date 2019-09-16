#pragma once

#include "../Math/RNG.h"
#include "../Util/Surface.h"

class Camera
{
public:
    Camera(Eigen::Affine3f transform = Eigen::Affine3f::Identity(), Vec2i resolution = Vec2i(0, 0), uint samples_per_pixel = 1, float focal_length = 1.f);

    Vec2i get_image_resolution() const { return resolution; }
    bool sample_pixel(const Vec2i &pixel, Ray &r, const Vec2f * rng = nullptr) const;
    uint get_pixel_samples(const Vec2i &pixel) const { return samples_per_pixel; }
private:
    Eigen::Affine3f transform;
    Vec2i resolution;
    uint samples_per_pixel;
    float aspect_ratio;
    float focal_length;
};
