#pragma once

#include <koshi/Intersect.h>

// TODO: Remove this limitation. 
#define MAX_INTERSECTS 8

KOSHI_OPEN_NAMESPACE

class IntersectList
{
public:
    DEVICE_FUNCTION IntersectList() : intersects_size(0) {}
    DEVICE_FUNCTION Intersect& push() { return intersects[intersects_size++]; }
    DEVICE_FUNCTION const uint& size() const { return intersects_size; }
    DEVICE_FUNCTION Intersect& operator[](const int& i) { return intersects[i]; }
    DEVICE_FUNCTION const Intersect& operator[](const int& i) const { return intersects[i]; }
    // DEVICE_FUNCTION void finalize(const double& tmax)

private:
    // const Ray ray;
    uint intersects_size;
    Intersect intersects[MAX_INTERSECTS];
};

KOSHI_CLOSE_NAMESPACE