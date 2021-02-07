#pragma once

#include <koshi/Sample.h>

KOSHI_OPEN_NAMESPACE

class Lobe
{
public:
    enum Type { NONE, LAMBERT, BACK_LAMBERT };
    enum Side { FRONT, BACK, SPHERE };
    DEVICE_FUNCTION Lobe(const Type& type, const Side& side) : type(type), side(side) {}
    DEVICE_FUNCTION const Type& getType() const { return type; }
    DEVICE_FUNCTION const Side& getSide() const { return side; }
private:
    Type type;
    Side side;
};

KOSHI_CLOSE_NAMESPACE