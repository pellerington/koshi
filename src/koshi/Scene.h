#pragma once

#include <vector>
#include <mutex>
#include <unordered_map>

#include <koshi/geometry/GeometryMesh.h>
#include <koshi/geometry/GeometryQuad.h>
#include <koshi/geometry/GeometryEnvironment.h>

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
    DeviceScene() : num_geometries(0), d_geometries(nullptr), d_distant_geometries(nullptr) {}

    void init(Scene * scene)
    {
        // Free our geometries... TODO: Replace these ptrs with unique ptrs so we don't have to do this CUDA_FREE crap.
        for(uint i = 0; i < geometries.size(); i++)
            CUDA_FREE(geometries[i]);
        CUDA_FREE(d_geometries);

        // Copy our geometries...
        const std::unordered_map<std::string, Geometry*>& scene_geometries = scene->getGeometries();
        geometries.resize(scene_geometries.size());
        num_geometries = scene_geometries.size();
        num_distant_geometries = 0;
        uint i = 0;
        for(auto it = scene_geometries.begin(); it != scene_geometries.end(); ++it, i++)
        {
            // TODO: Each object should sync itself.
            Geometry * geometry = it->second;
            if(geometry->getType() == Geometry::MESH)
            {
                GeometryMesh * mesh = (GeometryMesh*)geometry;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&geometries[i]), sizeof(GeometryMesh)));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(geometries[i]), mesh, sizeof(GeometryMesh), cudaMemcpyHostToDevice));
            }
            if(geometry->getType() == Geometry::QUAD)
            {
                GeometryQuad * quad = (GeometryQuad*)geometry;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&geometries[i]), sizeof(GeometryQuad)));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(geometries[i]), quad, sizeof(GeometryQuad), cudaMemcpyHostToDevice));
            }
            else if(geometry->getType() == Geometry::ENVIRONMENT)
            {
                GeometryEnvironment * environment = (GeometryEnvironment*)geometry;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&geometries[i]), sizeof(GeometryEnvironment)));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(geometries[i]), environment, sizeof(GeometryEnvironment), cudaMemcpyHostToDevice));
                distant_geometries.push_back(i);
                num_distant_geometries++;
            }
        }
        CUDA_CHECK(cudaMalloc(&d_geometries, sizeof(Geometry*) * geometries.size()));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_geometries), geometries.data(), sizeof(Geometry*) * geometries.size(), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_distant_geometries, sizeof(uint) * distant_geometries.size()));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_distant_geometries), distant_geometries.data(), sizeof(uint) * distant_geometries.size(), cudaMemcpyHostToDevice));
    }

    ~DeviceScene()
    {
        for(uint i = 0; i < geometries.size(); i++)
            CUDA_CHECK(cudaFree(geometries[i]));
        CUDA_CHECK(cudaFree(d_geometries));
        CUDA_CHECK(cudaFree(d_distant_geometries));
    }

    std::vector<Geometry*> geometries;
    std::vector<uint> distant_geometries;

    uint num_geometries;
    Geometry ** d_geometries;
    uint num_distant_geometries;
    uint * d_distant_geometries;
};

KOSHI_CLOSE_NAMESPACE