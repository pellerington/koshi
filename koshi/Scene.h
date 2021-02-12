#pragma once

#include <vector>
#include <mutex>

#include <koshi/Geometry.h>
#include <koshi/GeometryMesh.h>

KOSHI_OPEN_NAMESPACE

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

class DeviceScene
{
public:
    DeviceScene() : num_geometries(0) {}

    void init(Scene * scene)
    {
        // Free our geometries...
        for(uint i = 0; i < geometries.size(); i++)
            CUDA_CHECK(cudaFree(geometries[i]));
        CUDA_CHECK(cudaFree(d_geometries));

        // Copy our geometries...
        const std::unordered_map<std::string, Geometry*>& scene_geometries = scene->getGeometries();
        geometries.resize(scene_geometries.size());
        num_geometries = scene_geometries.size();
        uint i = 0;
        for(auto it = scene_geometries.begin(); it != scene_geometries.end(); ++it, i++)
        {
            // TODO: Each object should sync itself.
            GeometryMesh * geometry = dynamic_cast<GeometryMesh*>(it->second);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&geometries[i]), sizeof(GeometryMesh)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(geometries[i]), geometry, sizeof(GeometryMesh), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMalloc(&d_geometries, sizeof(Geometry*) * geometries.size()));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_geometries), geometries.data(), sizeof(Geometry*) * geometries.size(), cudaMemcpyHostToDevice));
    }

    ~DeviceScene()
    {
        for(uint i = 0; i < geometries.size(); i++)
            CUDA_CHECK(cudaFree(geometries[i]));
        CUDA_CHECK(cudaFree(d_geometries));
    }

    std::vector<Geometry*> geometries;

    uint num_geometries;
    Geometry ** d_geometries;
};

KOSHI_CLOSE_NAMESPACE