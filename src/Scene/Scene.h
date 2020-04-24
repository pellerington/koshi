#pragma once

#include <Math/Types.h>
// Scene should only relly know Object.h class
#include <geometry/Geometry.h>
#include <lights/LightSampler.h>
#include <Materials/Material.h>
#include <Textures/Texture.h>
#include <intersection/Intersect.h>
#include <Scene/Camera.h>

#include <vector>
#include <memory>
#include <map>

class Scene
{
public:
    Scene() {}
    Scene(const Camera& camera) : camera(camera) {}

    void pre_render();

    // static void intersection_callback(const RTCFilterFunctionNArguments * args);
    // Intersect intersect(Ray &ray);


    const Camera camera;

    bool add_object(Geometry * object);
    bool add_material(Material * material);
    bool add_texture(Texture * texture);

    std::vector<Geometry*>& get_objects() { return objects; }

private:
    std::vector<Geometry*> objects;
    std::vector<Material*> materials;
    std::vector<Texture*> textures;
};
