#pragma once

#include <Scene/Camera.h>
#include <base/Object.h>

#include <vector>

class Scene
{
public:
    Scene(const Camera& camera) : camera(camera) {}

    void pre_render();

    const Camera camera;

    bool add_object(Object * object);

    std::vector<Object*>& get_objects() { return objects; }

private:
    std::vector<Object*> objects;
};
