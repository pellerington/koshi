#pragma once

#include "../Scene/Scene.h"
#include "VolumeStack.h"

class VolumeIntegrator
{
public:
    VolumeIntegrator(Scene * scene, Ray &ray, const VolumeStack& volumes) : scene(scene), ray(ray), volumes(volumes) {}
    Vec3f shadow(const float &t);
    Vec3f integrate(Vec3f &out_weight, std::vector<VolumeSample> &samples, VolumeSample * in_sample);

private:
    Scene * scene;
    Ray& ray;
    const VolumeStack& volumes;

    // enum absorbtion vs singlescatter vs multiscatt;
    // enum homogenous vs hetrogenous;
};
