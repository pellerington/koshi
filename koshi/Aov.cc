#include <koshi/Aov.h>

#include <iostream>
#include <koshi/OptixHelpers.h>

KOSHI_OPEN_NAMESPACE

// For now, AOV format can be INT32 or FLOAT32

Aov::Aov(const std::string& name, const Vec2u& resolution, const uint& channels)
: name(name), resolution(resolution), channels(channels)
{
    buffer = (float*)malloc(sizeof(float) * resolution.x * resolution.y * channels);
    CUDA_CHECK(cudaMalloc(&d_buffer, sizeof(float) * resolution.x * resolution.y * channels));
}

Aov::~Aov()
{
    free(buffer);
    CUDA_CHECK(cudaFree(d_buffer));
}

void Aov::clear()
{
    // For all values set == 0.
}

void Aov::copyBuffer(void * dst, const Format& dst_format)
{
    cudaMemcpy(buffer, d_buffer, sizeof(float) * resolution.x * resolution.y * channels, cudaMemcpyDeviceToHost);

    // TODO: This is ugly...
    switch (dst_format)
    {
    case UINT8: {
        uint8_t * typed_dst = (uint8_t*)dst;
        for(uint i = 0; i < resolution.x * resolution.y * channels; i++)
            typed_dst[i] = buffer[i] * UINT8_MAX; // Also Average ect here if needed.
        break;
    }
    default: break;
    }
}

KOSHI_CLOSE_NAMESPACE