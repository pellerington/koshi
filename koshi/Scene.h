#pragma once

#include <unordered_map>

#include <koshi/Geometry.h>

KOSHI_OPEN_NAMESPACE

class Scene 
{
public:
    void addGeometry(const std::string& name, Geometry * geometry) { geometries[name] = geometry; }
    const std::unordered_map<std::string, Geometry*>& getGeometries() { return geometries; }

private:
    std::unordered_map<std::string, Geometry*> geometries;
};

KOSHI_CLOSE_NAMESPACE