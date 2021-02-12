#include <koshi/Aov.h>

#include <iostream>
#include <koshi/OptixHelpers.h>

KOSHI_OPEN_NAMESPACE

// For now, AOV format can be INT32 or FLOAT32

Aov::Aov(const std::string& name, const Vec2u& resolution, const uint& channels)
: name(name), resolution(resolution), channels(channels), mutex(new std::mutex)
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

void Aov::transferDeviceBuffer()
{
    std::lock_guard<std::mutex> guard(*mutex);
    cudaMemcpy(buffer, d_buffer, sizeof(float) * resolution.x * resolution.y * channels, cudaMemcpyDeviceToHost);
}

void Aov::copyBuffer(void * dst, const Format& dst_format, float num_samples)
{
    std::lock_guard<std::mutex> guard(*mutex);
    // TODO: This is ugly...
    switch (dst_format)
    {
    case UINT8: {
        uint8_t * typed_dst = (uint8_t*)dst;
        for(uint i = 0; i < resolution.x * resolution.y * channels; i++)
            typed_dst[i] = (buffer[i] > 1.f) ? UINT8_MAX : ((buffer[i] < 0.f) ? 0u : buffer[i] / num_samples); // Remove hardcoded Average ect here if needed.
        break;
    }
    case FLOAT32: {
        float * typed_dst = (float*)dst;
        for(uint i = 0; i < resolution.x * resolution.y * channels; i++)
            typed_dst[i] = buffer[i] / num_samples; // Also Average ect here if needed.
        break;
    }
    default: break;
    }
}

KOSHI_CLOSE_NAMESPACE