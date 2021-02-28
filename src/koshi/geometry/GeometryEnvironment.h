#pragma once

#include <string>
#include <cuda_runtime.h>

#include <koshi/geometry/Geometry.h>

KOSHI_OPEN_NAMESPACE

class GeometryEnvironment : public Geometry
{
public:
    GeometryEnvironment() : Geometry(Type::ENVIRONMENT) {}
    void createTexture(const std::string& filename);

    // TODO: Move me into the materials system.
    cudaTextureObject_t cuda_tex;
};

KOSHI_CLOSE_NAMESPACE