#pragma once

#include <vector>
#include <mutex>

#include <koshi/Geometry.h>

KOSHI_OPEN_NAMESPACE

// struct GeometryItem {
//     GeometryItem(const std::string& name, Geometry * geometry) : name(name), geometry(geometry) {}
//     std::string name; 
//     Geometry * geometry;
// };

class Scene 
{
public:
    void addGeometry(const std::string& name, Geometry * geometry) 
    { 
        std::lock_guard<std::mutex> guard(geometry_mutex);
        geometries[name] = geometry; 
    }
    void removeGeometry(const std::string& name) 
    {
        std::lock_guard<std::mutex> guard(geometry_mutex);
        geometries.erase(name); 
    }

    const std::unordered_map<std::string, Geometry*>& getGeometries() { return geometries; }

private:
    std::mutex geometry_mutex;
    std::unordered_map<std::string, Geometry*> geometries;
};

KOSHI_CLOSE_NAMESPACE