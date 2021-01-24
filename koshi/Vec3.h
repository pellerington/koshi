#pragma once

#include <cmath>

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

    // Negate
    DEVICE_FUNCTION Vec3<T> operator-() const { return Vec3<T>(-x, -y, -z); }

    // Addition
    DEVICE_FUNCTION Vec3<T>& operator+= (const Vec3<T>& v) { x += v.x; y += v.y; z += v.z; return *this; }
    DEVICE_FUNCTION Vec3<T> operator+ (const Vec3<T>& v) const { return Vec3<T>(x+v.x, y+v.y, z+v.z); }
    DEVICE_FUNCTION Vec3<T>& operator+= (const T& n) { x += n; y += n; z += n; return *this; }
    DEVICE_FUNCTION Vec3<T> operator+ (const T& n) const { return Vec3<T>(x+n, y+n, z+n); }
    friend DEVICE_FUNCTION Vec3<T> operator+ (const T& n, const Vec3<T>& v) { return Vec3<T>(v.x+n, v.y+n, v.z+n); }

    // Subtraction
    DEVICE_FUNCTION Vec3<T>& operator-= (const Vec3<T>& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    DEVICE_FUNCTION Vec3<T> operator- (const Vec3<T>& v) const { return Vec3<T>(x-v.x, y-v.y, z-v.z); }
    DEVICE_FUNCTION Vec3<T>& operator-= (const T& n) { x -= n; y -= n; z -= n; return *this; }
    DEVICE_FUNCTION Vec3<T> operator- (const T& n) const { return Vec3<T>(x-n, y-n, z-n); }
    friend DEVICE_FUNCTION Vec3<T> operator- (const T& n, const Vec3<T>& v) { return Vec3<T>(n-v.x, n-v.y, n-v.z); }

    // Multiplication
    DEVICE_FUNCTION Vec3<T>& operator*= (const Vec3<T>& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
    DEVICE_FUNCTION Vec3<T> operator* (const Vec3<T>& v) const { return Vec3<T>(x*v.x, y*v.y, z*v.z); }
    DEVICE_FUNCTION Vec3<T>& operator*= (const T& n) { x *= n; y *= n; z *= n; return *this; }
    DEVICE_FUNCTION Vec3<T> operator* (const T& n) const { return Vec3<T>(x*n, y*n, z*n); }
    friend DEVICE_FUNCTION Vec3<T> operator* (const T& n, const Vec3<T>& v) { return Vec3<T>(v.x*n, v.y*n, v.z*n); }

    // Division
    DEVICE_FUNCTION Vec3<T>& operator/= (const Vec3<T>& v) { x /= v.x; y /= v.y; z /= v.z; return *this; }
    DEVICE_FUNCTION Vec3<T> operator/ (const Vec3<T>& v) const { return Vec3<T>(x/v.x, y/v.y, z/v.z); }
    DEVICE_FUNCTION Vec3<T>& operator/= (const T& n) { x /= n; y /= n; z /= n; return *this; }
    DEVICE_FUNCTION Vec3<T> operator/ (const T& n) const { return Vec3<T>(x/n, y/n, z/n); }
    friend DEVICE_FUNCTION Vec3<T> operator/ (const T& n, const Vec3<T>& v) { return Vec3<T>(n/v.x, n/v.y, n/v.z); }

    // Geometry
    DEVICE_FUNCTION T dot(const Vec3<T>& v) const { return x*v.x + y*v.y + z*v.z; }
    DEVICE_FUNCTION static Vec3<T> cross(const Vec3<T>& v0, const Vec3<T>& v1) { return Vec3<T>(v0.y*v1.z-v0.z*v1.y, v0.z*v1.x-v0.x*v1.z, v0.x*v1.y-v0.y*v1.x); }
    DEVICE_FUNCTION T length() const { return sqrt(x*x + y*y + z*z); }
    DEVICE_FUNCTION Vec3<T>& normalize() { T l = length(); x /= l; y /= l; z /= l; return *this; }

};

typedef Vec3<float> Vec3f;
typedef Vec3<int> Vec3i;
typedef Vec3<uint> Vec3u;

KOSHI_CLOSE_NAMESPACE