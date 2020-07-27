#include <koshi/camera/Camera.h>

Camera::Camera(const Transform3f& transform, const Vec2u& resolution,const float& focal_length)
: transform(transform), origin(transform * Vec3f(0.f, 0.f, 0.f, 1.f)), resolution(resolution), 
  inv_resolution(Vec2f(1.f/resolution.x, 1.f/resolution.y)),
  aspect_ratio((float) resolution.x / resolution.y), focal_length(focal_length), 
  pixel_delta(-1.f / resolution.x * aspect_ratio, -1.f / resolution.y)
{
    delta_sampler = &GaussianFilterSampler::sample;
}

Ray Camera::sample_pixel(const uint& x, const uint& y, const float rng[2]) const
{
    Vec2f pixel_position = 0.5f - Vec2f(x, y) * inv_resolution;
    pixel_position.x *= aspect_ratio;
    pixel_position += pixel_delta * delta_sampler(rng);

    Vec3f position = transform * Vec3f(pixel_position.x, pixel_position.y, focal_length);
    return Ray(origin, (position - origin).normalized());
}   
