#pragma once

#include <koshi/material/Lambert.h>
#include <koshi/material/BackLambert.h>
#include <koshi/material/Reflect.h>

#define MAX_LOBES 16

KOSHI_OPEN_NAMESPACE

union LobeData
{
    DEVICE_FUNCTION LobeData() {}
    DEVICE_FUNCTION ~LobeData() {}
    Lambert lambert;
    BackLambert back_lambert;
    Reflect reflect;
};

// TODO: Rename this to material config and have it store emitted light ect as well

// TODO: Lobe Array should be a templated class, since we will use it so much

class LobeArray
{
public:
    DEVICE_FUNCTION LobeArray() : num_lobes(0) {}
    template<typename T>
    DEVICE_FUNCTION T& push() { return *(T*)&lobes[num_lobes++] = T(); }
    DEVICE_FUNCTION const uint& size() const { return num_lobes; }
    DEVICE_FUNCTION bool empty() const { return (num_lobes == 0); }
    DEVICE_FUNCTION const Lobe * operator[](const uint& i) const { return (Lobe*)&lobes[i]; }
private:
    uint num_lobes;
    LobeData lobes[MAX_LOBES];
};

KOSHI_CLOSE_NAMESPACE