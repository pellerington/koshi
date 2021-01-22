#pragma once

#include <koshi/Koshi.h>

KOSHI_OPEN_NAMESPACE

template<typename T>
class Vec3
{
public:
    union {
        struct { T x, y, z; };
        struct { T u, v, w; };
        struct { T r, g, b; };
        T data[3];
    };

    DEVICE_FUNCTION Vec3() : x(0), y(0), z(0) {}
    DEVICE_FUNCTION Vec3(const T& x, const T& y, const T& z) : x(x), y(y), z(z)  {}
    DEVICE_FUNCTION Vec3(const T& n) : x(n), y(n), z(n)  {}

    DEVICE_FUNCTION T& operator[](const int& i) { return data[i]; }
    DEVICE_FUNCTION const T& operator[](const int& i) const { return data[i]; }

    // Addition
    DEVICE_FUNCTION Vec3<T>& operator+= (const Vec3<T>& v) { x += v.x; y += v.y; z += v.z; return *this; }
    DEVICE_FUNCTION Vec3<T> operator+ (const Vec3<T>& v) const { return Vec3<T>(x+v.x, y+v.y, z+v.z); }
    DEVICE_FUNCTION Vec3<T>& operator+= (const T& n) { x += n; y += n; z += n; return *this; }
    DEVICE_FUNCTION Vec3<T> operator+ (const T& n) const { return Vec3<T>(x+n, y+n, z+n); }
    friend DEVICE_FUNCTION Vec3<T> operator+ (const T& n, const Vec3<T>& v) { return Vec3<T>(v.x+n, v.y+n, v.z+n); }

    // Multiply
    DEVICE_FUNCTION Vec3<T>& operator*= (const Vec3<T>& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
    DEVICE_FUNCTION Vec3<T> operator* (const Vec3<T>& v) const { return Vec3<T>(x*v.x, y*v.y, z*v.z); }
    DEVICE_FUNCTION Vec3<T>& operator*= (const T& n) { x *= n; y *= n; z *= n; return *this; }
    DEVICE_FUNCTION Vec3<T> operator* (const T& n) const { return Vec3<T>(x*n, y*n, z*n); }
    friend DEVICE_FUNCTION Vec3<T> operator* (const T& n, const Vec3<T>& v) { return Vec3<T>(v.x*n, v.y*n, v.z*n); }
};

typedef Vec3<float> Vec3f;
typedef Vec3<int> Vec3i;
typedef Vec3<uint> Vec3u;

KOSHI_CLOSE_NAMESPACE