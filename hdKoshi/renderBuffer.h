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
    , width(0), height(0), depth(0), format(HdFormatInvalid)
    , buffer(), multiSampled(false), mappers(0), aov(nullptr)
    {
    }

    ~HdKoshiRenderBuffer() {}

    virtual bool Allocate(const GfVec3i& dimensions, HdFormat format, bool multiSampled) override;

    virtual unsigned int GetWidth() const override { return width; }
    virtual unsigned int GetHeight() const override { return height; }
    virtual unsigned int GetDepth() const override { return depth; }
    virtual HdFormat GetFormat() const override { return format; }
    virtual bool IsMultiSampled() const override { return multiSampled; }

    virtual void * Map() override { mappers++; return buffer.data(); }
    virtual void Unmap() override { mappers--; }
    virtual bool IsMapped() const override { return mappers.load() != 0; }

    virtual void Resolve() override;
    virtual bool IsConverged() const override { return false; }

    void setKoshiAov(const Koshi::Aov * _aov) { aov = _aov; }

    virtual void _Deallocate() override;

private:
    unsigned int width;
    unsigned int height;
    unsigned int depth;
    HdFormat format;
    std::vector<uint8_t> buffer;
    bool multiSampled;
    std::atomic<int> mappers;
    const Koshi::Aov * aov;
};

PXR_NAMESPACE_CLOSE_SCOPE