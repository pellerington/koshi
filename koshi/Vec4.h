#pragma once

#include <koshi/Koshi.h>
#include <koshi/Vec3.h>

KOSHI_OPEN_NAMESPACE

template<typename T>
class Vec4
{
public:
    union {
        struct { T x, y, z, w; };
        struct { T r, g, b, a; };
        T data[4];
    };


    DEVICE_FUNCTION Vec4() : x(0), y(0), z(0), w(0) {}
    DEVICE_FUNCTION Vec4(const T& x, const T& y, const T& z, const T& w) : x(x), y(y), z(z), w(w)  {}
    DEVICE_FUNCTION Vec4(const T& n) : x(n), y(n), z(n), w(n)  {}
    DEVICE_FUNCTION Vec4(const Vec3<T>& v, const T& n) : x(v.x), y(v.y), z(v.z), w(n)  {}

    DEVICE_FUNCTION operator Vec3<T>() { return Vec3<T>(x, y, z); }

    DEVICE_FUNCTION T& operator[](const int& i) { return data[i]; }
    DEVICE_FUNCTION const T& operator[](const int& i) const { return data[i]; }
};

typedef Vec4<float> Vec4f;
typedef Vec4<int> Vec4i;
typedef Vec4<uint> Vec4u;

KOSHI_CLOSE_NAMESPACE