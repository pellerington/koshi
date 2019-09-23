#pragma once

#include "../Math/Types.h"
#include "../Scene/Scene.h"
#include "../Util/Surface.h"
#include "../Util/Ray.h"

class Integrator
{
public:
    //Constructor. Perform the main constuction in pre_render for one time renders, or create per integration instance.
    Integrator(Scene * scene) : scene(scene) {};

    //Called before render starts. Eg. Photon map would compute the cache here.
    virtual void pre_render() = 0;

    //Creates a copy of the current integrator within the context of a ray. Eg. Pathtracers would intersect and setup the samples here.
    virtual std::shared_ptr<Integrator> create(Ray &ray) = 0;

    //Perform the actual integration. Eg. Bidirectional would actually generate light path here.
    virtual void integrate(const size_t num_samples) = 0;

    //Provides the number of samples required to perform the full integration.
    virtual size_t get_required_samples() = 0;
    //Tells the renderer when it has completed all it's samples.
    virtual bool completed() = 0;

    //Output the color of the integrator.
    virtual Vec3f get_color() = 0;
protected:
    Scene * scene;
};
