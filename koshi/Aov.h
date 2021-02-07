#pragma once

#include <string>
#include <cuda_runtime.h>

#include <koshi/Format.h>
#include <koshi/math/Vec2.h>
#include <koshi/math/Vec3.h>
#include <koshi/math/Vec4.h>

KOSHI_OPEN_NAMESPACE

// For now, AOV format can be INT32 or FLOAT32

class Aov 
{
public:
    Aov(const std::string& name, const Vec2u& resolution, const uint& channels);
    ~Aov();

    DEVICE_FUNCTION void write(const Vec2u& coord, const Vec4f& value)
    {
        for(uint i = 0; i < channels; i++)
            d_buffer[(coord.x + resolution.x*coord.y)*channels + i] = value[i];
        return;
    }

    void clear();
    void copyBuffer(void * dst, const Format& dst_format);

    const std::string name;
    const Vec2u resolution;
    const uint channels;
    // const AovType = Average, Total, FirstValue ect...

private:
    float * d_buffer;
    float * buffer;
    // const float * samples_buffer;
};

KOSHI_CLOSE_NAMESPACE