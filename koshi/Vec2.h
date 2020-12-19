#pragma once

#include <koshi/Koshi.h>

KOSHI_OPEN_NAMESPACE

template<typename T>
class Vec2
{
public:
    T x, y;

    DEVICE_FUNCTION Vec2() : x(0), y(0) {}
    DEVICE_FUNCTION Vec2(const T& x, const T& y) : x(x), y(y)  {}
    DEVICE_FUNCTION Vec2(const T& n) : x(n), y(n)  {}

    DEVICE_FUNCTION T& operator[](const int& i) { 
        switch (i) {
            case 0: return x;
            case 1: return y;
            default: return 0;
        }    
    }
    DEVICE_FUNCTION const T& operator[](const int& i) const {
        switch (i) {
            case 0: return x;
            case 1: return y;
            default: return 0;
        }    
    }
};

typedef Vec2<float> Vec2f;
typedef Vec2<int> Vec2i;
typedef Vec2<uint> Vec2u;

KOSHI_CLOSE_NAMESPACE