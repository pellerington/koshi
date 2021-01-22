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
};

typedef Vec2<float> Vec2f;
typedef Vec2<int> Vec2i;
typedef Vec2<uint> Vec2u;

KOSHI_CLOSE_NAMESPACE