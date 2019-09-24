#pragma once

#include "../Math/Types.h"
#include "../Scene/Scene.h"
#include "../Util/Surface.h"
#include "../Util/Ray.h"

class Integrator
{
public:
    Integrator(Scene * scene) : scene(scene) {};

    // Called before render starts. Eg. Photon map would compute the cache here.
    virtual void pre_render() = 0;

    // Perform the actual integration along a ray. Eg. Bidirectional would actually generate light path here.
    virtual Vec3f integrate(Ray &ray) const = 0;

protected:
    Scene * scene;
};
