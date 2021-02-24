#pragma once

#include <koshi/math/Vec3.h>

KOSHI_OPEN_NAMESPACE

struct Sample
{
    Vec3f wo;
    Vec3f value;
    float pdf;
};

KOSHI_CLOSE_NAMESPACE