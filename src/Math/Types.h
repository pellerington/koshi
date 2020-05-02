#pragma once

#include <iostream>
#include <Math/Vec3f.h>
#include <Math/Vec2.h>
#include <Math/Box3f.h>

#define HALF_PI 1.57079632679f
#define PI 3.14159265359f
#define TWO_PI 6.28318530718f
#define FOUR_PI 12.5663706144f

#define INV_PI 0.31830988618f
#define INV_TWO_PI 0.15915494309f
#define INV_FOUR_PI 0.07957747154f

#define EPSILON_F 0.000001f
#define DELTA_PDF 1e9f

static const Vec3f VEC3F_ZERO = 0.f;
static const Vec3f VEC3F_ONES = 1.f;

static const Box3f BOX3F_UNIT = Box3f(VEC3F_ZERO, VEC3F_ONES);

typedef unsigned int uint;

struct VERT_DATA { float x, y, z, r; };
struct NORM_DATA { float x, y, z; };
struct UV_DATA { float u, v; };
struct TRI_DATA { uint v0, v1, v2; };
struct QUAD_DATA { uint v0, v1, v2, v3; };