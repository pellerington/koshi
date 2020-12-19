#pragma once

#include <koshi/Koshi.h>

KOSHI_OPEN_NAMESPACE

template<typename T>
class Vec3
{
public:
    T x, y, z;

    DEVICE_FUNCTION Vec3() : x(0), y(0), z(0) {}
    DEVICE_FUNCTION Vec3(const T& x, const T& y, const T& z) : x(x), y(y), z(z)  {}
    DEVICE_FUNCTION Vec3(const T& n) : x(n), y(n), z(n)  {}

    DEVICE_FUNCTION T& operator[](const int& i) { 
        switch (i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: /* ERROR */ return z;
        }
    }

    DEVICE_FUNCTION const T& operator[](const int& i) const { 
        switch (i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: /* ERROR */ return z;
        }
    }
};

typedef Vec3<float> Vec3f;
typedef Vec3<int> Vec3i;
typedef Vec3<uint> Vec3u;

KOSHI_CLOSE_NAMESPACE