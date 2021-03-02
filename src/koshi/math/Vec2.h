#pragma once

#include <koshi/Koshi.h>

KOSHI_OPEN_NAMESPACE

template<typename T>
class Vec2
{
public:
    union {
        struct { T x, y; };
        struct { T u, v; };
        T data[2];
    };

    DEVICE_FUNCTION Vec2() : x(0), y(0) {}
    DEVICE_FUNCTION Vec2(const T& x, const T& y) : x(x), y(y) {}
    DEVICE_FUNCTION Vec2(const T& n) : x(n), y(n) {}

    DEVICE_FUNCTION T& operator[](const int& i) { return data[i]; }
    DEVICE_FUNCTION const T& operator[](const int& i) const { return data[i]; }

    // Addition
    DEVICE_FUNCTION Vec2<T>& operator+= (const Vec2<T>& v) { x += v.x; y += v.y; return *this; }
    DEVICE_FUNCTION Vec2<T> operator+ (const Vec2<T>& v) const { return Vec2<T>(x+v.x, y+v.y); }
    DEVICE_FUNCTION Vec2<T>& operator+= (const T& n) { x += n; y += n; return *this; }
    DEVICE_FUNCTION Vec2<T> operator+ (const T& n) const { return Vec2<T>(x+n, y+n); }
    friend DEVICE_FUNCTION Vec2<T> operator+ (const T& n, const Vec2<T>& v) { return Vec2<T>(v.x+n, v.y+n); }

    // Division
    DEVICE_FUNCTION Vec2<T>& operator/= (const Vec2<T>& v) { x /= v.x; y /= v.y; return *this; }
    DEVICE_FUNCTION Vec2<T> operator/ (const Vec2<T>& v) const { return Vec2<T>(x/v.x, y/v.y); }
    DEVICE_FUNCTION Vec2<T>& operator/= (const T& n) { x /= n; y /= n; return *this; }
    DEVICE_FUNCTION Vec2<T> operator/ (const T& n) const { return Vec2<T>(x/n, y/n); }
    friend DEVICE_FUNCTION Vec2<T> operator/ (const T& n, const Vec2<T>& v) { return Vec2<T>(n/v.x, n/v.y); }

    // Comparison
    DEVICE_FUNCTION bool operator!= (const Vec2<T>& v) const { return x != v.x || y != v.y; }
};

typedef Vec2<float> Vec2f;
typedef Vec2<int> Vec2i;
typedef Vec2<uint> Vec2u;

KOSHI_CLOSE_NAMESPACE