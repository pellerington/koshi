#pragma once

class Buffer
{
public:
    Buffer(const uint& size, const uint& dimensions) : size(size), dimensions(dimensions)
    {
        buffer = new float[size * dimensions];
    }

    void set(const uint& index, const Vec3f& v)
    {
        float * item = &buffer[index * dimensions];
        for(uint i = 0; i < dimensions; i++)
            item[i] = v[i];
    }
    
    void add(const uint& index, const Vec3f& v)
    {
        float * item = &buffer[index * dimensions];
        for(uint i = 0; i < dimensions; i++)
            item[i] += v[i];
    }

    Vec3f get(const uint& index)
    {
        Vec3f v;
        float * item = &buffer[index * dimensions];
        for(uint i = 0; i < dimensions; i++)
            v[i] = item[i];
        return v;
    }

    ~Buffer() 
    { 
        delete buffer;
    }

private:
    float * buffer;
    const uint size, dimensions;
};