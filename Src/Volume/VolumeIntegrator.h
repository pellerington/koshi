#pragma once

#include "../Scene/Scene.h"
#include "VolumeStack.h"

class VolumeIntegrator
{
public:
    VolumeIntegrator(Scene * scene, Ray &ray, const VolumeStack& volumes) : scene(scene), ray(ray), volumes(volumes) {}
    virtual Vec3f shadow(const float &t) = 0;
    virtual Vec3f emission(/* float pdf for direct sampling???*/) = 0;
    virtual void scatter(std::vector<VolumeSample> &samples) = 0;

protected:
    Scene * scene;
    Ray& ray;
    const VolumeStack& volumes;

    // enum absorbtion vs singlescatter vs multiscatt;
    // enum homogenous vs hetrogenous;
};
