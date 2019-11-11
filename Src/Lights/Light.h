#pragma once

struct LightSample {
    Vec3f position;
    Vec3f intensity;
    float pdf = 0.f;
};


class Light
{
public:
    Light(const Vec3f &intensity) : intensity(intensity) {}

    Vec3f get_emission(/*const Surface &surface*/)
    {
        return intensity;
    }
private:
    // Add saturation/spectrum here too.
    const Vec3f intensity;
};
