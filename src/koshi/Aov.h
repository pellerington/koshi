#pragma once

#include <string>
#include <mutex>
#include <memory>
#include <cuda_runtime.h>

#include <koshi/Format.h>
#include <koshi/math/Vec2.h>
#include <koshi/math/Vec3.h>
#include <koshi/math/Vec4.h>
#include <koshi/String.h>

KOSHI_OPEN_NAMESPACE

// For now, AOV format can be INT32 or FLOAT32

class Aov 
{
public:
    Aov(const std::string& name, const Vec2u& resolution, const uint& channels);
    ~Aov();

    DEVICE_FUNCTION void write(const Vec2u& pixel, const Vec4f& value)
    {
        for(uint i = 0; i < channels; i++)
            d_buffer[(pixel.x + resolution.x*pixel.y)*channels + i] += value[i]; // ONLY Perform the ADD if it is averaging or summing...
        return;
    }

    DEVICE_FUNCTION Vec4f read(const Vec2u& pixel)
    {
        Vec4f value;
        for(uint i = 0; i < channels; i++)
            value[i] = d_buffer[(pixel.x + resolution.x*pixel.y)*channels + i]; // Should take into account mode, Average/Total/FirstValue????
        return value;
    }

    DEVICE_FUNCTION bool operator==(const char * _name) { return name == _name; }

    void clear();
    void copy(void * dst, float num_samples);

    std::string getName() const { return name; }
    const Vec2u& getResolution() const { return resolution; }
    const uint& getChannels() const { return channels; }

private:
    String name;
    Vec2u resolution;
    uint channels;
    // const AovType = Average, Total, FirstValue ect...

    std::shared_ptr<std::mutex> mutex;
    float * d_buffer;
};

KOSHI_CLOSE_NAMESPACE