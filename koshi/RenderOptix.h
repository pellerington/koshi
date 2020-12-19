#pragma once

#include <sstream>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <optix.h>

#include <koshi/Aov.h>
#include <koshi/Intersector.h>
#include <koshi/Camera.h>
#include <koshi/Scene.h>

KOSHI_OPEN_NAMESPACE

struct RayGenData
{
};

struct MissData
{
};

struct HitGroupData
{
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
    Camera * camera;
    Intersector * intersector;
    Aov * aovs;
    uint aovs_size;
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

    // void stop();
    void reset();
    // void pause();
    void start();

private:
    Scene * scene;
    Camera * camera;
    std::vector<Aov> aovs;

    // GPU Specific
    OptixDeviceContext context;
    OptixModule module;
    OptixProgramGroup raygen_prog_group;
    OptixProgramGroup hitgroup_prog_group;
    OptixProgramGroup miss_prog_group;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt = {};

};

KOSHI_CLOSE_NAMESPACE