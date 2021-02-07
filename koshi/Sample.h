#pragma once

#include <koshi/math/Vec3.h>

KOSHI_OPEN_NAMESPACE

struct Sample
{
    Vec3f wo;
    Vec3f value;
    Vec3f pdf;
};

KOSHI_CLOSE_NAMESPACE