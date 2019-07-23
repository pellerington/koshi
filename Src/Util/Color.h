#pragma once

#include "../Math/Types.h"

inline bool is_black(const Vec3f& color)
{
    return !color.r() && !color.g() && !color.b();
}
