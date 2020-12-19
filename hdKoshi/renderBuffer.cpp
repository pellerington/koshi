#include "renderBuffer.h"

PXR_NAMESPACE_OPEN_SCOPE

bool HdKoshiRenderBuffer::Allocate(const GfVec3i& dimensions, HdFormat _format, bool _multiSampled)
{
    width = dimensions[0];
    height = dimensions[1];
    depth = dimensions[2];
    format = _format;
    multiSampled = _multiSampled;
    buffer.resize(width * height * HdDataSizeOfFormat(format));
    return true;
}

void HdKoshiRenderBuffer::Resolve()
{
    // if (!_multiSampled) {
    //     return;
    // }
}

void HdKoshiRenderBuffer::_Deallocate()
{
    // If the buffer is mapped while we're doing this, there's not a great recovery path...
    TF_VERIFY(!IsMapped());
    width = 0;
    height = 0;
    format = HdFormatInvalid;
    multiSampled = false;
    buffer.resize(0);
    mappers.store(0);
}

PXR_NAMESPACE_CLOSE_SCOPE
