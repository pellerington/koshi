#include "renderBuffer.h"

#include "renderParam.h"

PXR_NAMESPACE_OPEN_SCOPE

bool HdKoshiRenderBuffer::Allocate(const GfVec3i& dimensions, HdFormat _format, bool _multi_sampled)
{
    width = dimensions[0];
    height = dimensions[1];
    depth = dimensions[2];
    format = _format;
    multi_sampled = multi_sampled;
    buffer.resize(width * height * HdDataSizeOfFormat(format));
    return true;
}

void HdKoshiRenderBuffer::Sync(HdSceneDelegate * sceneDelegate, HdRenderParam *renderParam, HdDirtyBits *dirtyBits)
{
    if (*dirtyBits & DirtyDescription)
        static_cast<HdKoshiRenderParam*>(renderParam)->StopRender();
    HdRenderBuffer::Sync(sceneDelegate, renderParam, dirtyBits);
}

void HdKoshiRenderBuffer::Finalize(HdRenderParam *renderParam)
{
    static_cast<HdKoshiRenderParam*>(renderParam)->StopRender();
    HdRenderBuffer::Finalize(renderParam);
}

void HdKoshiRenderBuffer::_Deallocate()
{
    TF_VERIFY(!IsMapped());
    width = 0;
    height = 0;
    format = HdFormatInvalid;
    multi_sampled = false;
    buffer.resize(0);
    mappers.store(0);
}

PXR_NAMESPACE_CLOSE_SCOPE
