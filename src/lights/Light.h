#pragma once

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
