#pragma once

#include <base/Object.h>

#include <map>
class Resources;
class Camera;

class Scene
{
public:
    void pre_render(Resources& resources);

    bool add_object(const std::string& name, Object * object);

    Object * get_object(const std::string& name);
    // TODO: replace this with a better way to get / search objects. Look into database implementations.
    auto begin() { return objects.begin(); }
    auto end() { return objects.end(); }

    void set_camera(Camera * _camera) { camera = _camera; }
    const Camera * get_camera() const { return camera; }

private:
    std::unordered_map<std::string, Object*> objects;
    const Camera * camera;
};
