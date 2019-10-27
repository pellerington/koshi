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

// Move to integrator helper functions header
inline IorStack get_next_ior(const std::shared_ptr<Material> &material, const Surface &surface, const bool inside_object)
{
    // If we are entering an object add the material ior.
    if(surface.front && inside_object)
        return IorStack(material->get_ior(), &surface.ior);

    // If we are leaving an object pop the last stack off.
    else if(!surface.front && !inside_object)
        return (surface.ior.prev) ? *surface.ior.prev : IorStack();

    // Otherwise keep the same stack.
    return surface.ior;
}
