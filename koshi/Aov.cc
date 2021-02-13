#include <koshi/Aov.h>

#include <iostream>
#include <koshi/OptixHelpers.h>

KOSHI_OPEN_NAMESPACE

// For now, AOV format can be INT32 or FLOAT32

Aov::Aov(const std::string& name, const Vec2u& resolution, const uint& channels)
: name(name), resolution(resolution), channels(channels), mutex(new std::mutex)
{
    std::lock_guard<std::mutex> guard(*mutex);
    CUDA_CHECK(cudaMalloc(&d_buffer, sizeof(float) * resolution.x * resolution.y * channels));
}

Aov::~Aov()
{
    std::lock_guard<std::mutex> guard(*mutex);
    // Hacky way to fix std::vector copying. TODO: Find a better way of fixing this in the future. (SHOULD WORK LIKE A SHARED PTR)
    if(mutex.unique()) 
        CUDA_CHECK(cudaFree(d_buffer));
}

void Aov::clear()
{
    // Set all values to 0.
}

void Aov::copy(void * dst, float num_samples)
{
    std::lock_guard<std::mutex> guard(*mutex);
    // TODO: This is ugly...
    float * buffer = (float *)dst;
    cudaMemcpy(buffer, d_buffer, sizeof(float) * resolution.x * resolution.y * channels, cudaMemcpyDeviceToHost);
    for(uint i = 0; i < resolution.x * resolution.y * channels; i++)
        buffer[i] = buffer[i] / num_samples; // averaging should depend on type of buffer... // Maybe do this on the device...
}

KOSHI_CLOSE_NAMESPACE