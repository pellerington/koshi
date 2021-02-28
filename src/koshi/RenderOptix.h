#pragma once

#include <sstream>
#include <iostream>
#include <cstdint>
#include <vector>
#include <thread>
#include <optix.h>

#include <koshi/Aov.h>
#include <koshi/Intersector.h>
#include <koshi/Camera.h>
#include <koshi/Scene.h>
#include <koshi/random/Random.h>

KOSHI_OPEN_NAMESPACE

struct RayGenData
{
};

struct MissData
{
};

struct HitGroupData
{
    uint geometry_id;
};

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

struct Resources
{
    DeviceScene * scene;
    Camera * camera;
    Intersector * intersector;
    RandomGenerator * random_generator;
    Aov * aovs;
    uint aovs_size;
    DEVICE_FUNCTION Aov * getAov(const char * name)
    {
        for(uint i = 0; i < aovs_size; i++)
            if(aovs[i] == name)
                return &aovs[i];
        return nullptr;
    }
};

class RenderOptix 
{
public:
    RenderOptix();
    ~RenderOptix();

    // TODO: These functions should be common to both renderers.
    void setScene(Scene * scene);
    void setCamera(Camera * camera);
    Aov * getAov(const std::string& name);
    Aov * addAov(const std::string& name, const uint& channels);

    void reset();
    // void pause();
    void start();
    void pass();

    const uint& getPasses() { return passes; }

private:

    uint passes;

    RandomGenerator random_generator;

    Scene * scene;
    Camera * camera;
    
    IntersectorOptix * intersector;
    std::vector<Aov> aovs;

    DeviceScene device_scene;
    Resources resources;
    CUdeviceptr d_resources;

    // Optix Specific
    CUstream cuda_stream;
    OptixShaderBindingTable sbt;
    OptixDeviceContext context;
    OptixModule module;
    OptixProgramGroup raygen_prog_group;
    OptixProgramGroup hitgroup_prog_group;
    OptixProgramGroup miss_prog_group;
    OptixPipeline pipeline;

};

KOSHI_CLOSE_NAMESPACE