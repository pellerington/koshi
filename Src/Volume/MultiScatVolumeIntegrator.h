#pragma once

#include "VolumeIntegrator.h"

class MultiScatVolumeIntegrator : public VolumeIntegrator
{
public:
    MultiScatVolumeIntegrator(Scene * scene, Ray &ray, const VolumeStack& volumes);

    Vec3f shadow(const float &t);

    Vec3f emission(/* float pdf for direct sampling???*/);

    void scatter(std::vector<VolumeSample> &samples);

private:
    Vec3f weight;
    float tmax;

    Vec3f weighted_emission;

    bool has_scatter;
    VolumeSample sample;
};
