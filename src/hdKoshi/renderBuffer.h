#pragma once

#include "pxr/pxr.h"
#include "pxr/imaging/hd/renderBuffer.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec4f.h"
#include <iostream>

#include <koshi/Aov.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdKoshiRenderBuffer : public HdRenderBuffer
{
public:
    HdKoshiRenderBuffer(const SdfPath& id) 
    : HdRenderBuffer(id)
    , width(0), height(0), depth(0), converged(false), format(HdFormatInvalid)
    , buffer(), multi_sampled(false), mappers(0)
    {
    }

    ~HdKoshiRenderBuffer() {}

    virtual bool Allocate(const GfVec3i& dimensions, HdFormat format, bool multi_sampled) override;
    virtual void Sync(HdSceneDelegate * sceneDelegate, HdRenderParam *renderParam, HdDirtyBits *dirtyBits) override;
    virtual void Finalize(HdRenderParam *renderParam) override;
    virtual void Resolve() override {}
    void SetConverged() { converged = true; }
    virtual bool IsConverged() const override { return false; }
    virtual void _Deallocate() override;

    virtual unsigned int GetWidth() const override { return width; }
    virtual unsigned int GetHeight() const override { return height; }
    virtual unsigned int GetDepth() const override { return depth; }
    virtual HdFormat GetFormat() const override { return format; }
    virtual bool IsMultiSampled() const override { return multi_sampled; }

    virtual void * Map() override { mutex.lock(); mappers++; return buffer.data(); }
    virtual void Unmap() override { mappers--; mutex.unlock(); }
    virtual bool IsMapped() const override { return mappers.load() != 0; }

private:
    unsigned int width;
    unsigned int height;
    unsigned int depth;
    bool converged;
    HdFormat format;
    std::vector<uint8_t> buffer;
    bool multi_sampled;
    std::atomic<int> mappers;

    // TODO: This locked access is bad...
    std::mutex mutex;
};

PXR_NAMESPACE_CLOSE_SCOPE