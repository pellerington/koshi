#pragma once

#include <koshi/Koshi.h>

KOSHI_OPEN_NAMESPACE

template<typename T>
class Vec4
{
public:
    T x, y, z, w;

    DEVICE_FUNCTION Vec4() : x(0), y(0), z(0), w(0) {}
    DEVICE_FUNCTION Vec4(const T& x, const T& y, const T& z, const T& w) : x(x), y(y), z(z), w(w)  {}
    DEVICE_FUNCTION Vec4(const T& n) : x(n), y(n), z(n), w(n)  {}

    DEVICE_FUNCTION T& operator[](const int& i) { 
        switch (i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            case 3: return w;
            default: /* ERROR */ return w;
        }
    }

    DEVICE_FUNCTION const T& operator[](const int& i) const { 
        switch (i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            case 3: return w;
            default: /* ERROR */ return w;
        }
    }
};

typedef Vec4<float> Vec4f;
typedef Vec4<int> Vec4i;
typedef Vec4<uint> Vec4u;

KOSHI_CLOSE_NAMESPACE