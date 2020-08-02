#pragma once

#include <koshi/base/Object.h>

#include <map>
class Resources;
class Camera;

class Scene
{
public:
    void pre_render(Resources& resources);

    void add_object(const std::string& name, Object * object)
    {
        objects.insert(std::make_pair(name, object));
    }

    Object * get_object(const std::string& name)
    {
        auto object = objects.find(name);
        if(object != objects.end())
            return object->second;
        return nullptr;
    }

    template<class T>
    T * get_object(const std::string& name)
    {
        auto object = objects.find(name);
        if(object != objects.end())
            return dynamic_cast<T*>(object->second);
        return nullptr;
    }

    auto begin() { return objects.begin(); }
    auto end() { return objects.end(); }

    void set_camera(Camera * _camera) { camera = _camera; }
    Camera * get_camera() const { return camera; }

private:
    std::unordered_map<std::string, Object*> objects;
    Camera * camera;
};
