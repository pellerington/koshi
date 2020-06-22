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

#define EPSILON_F 1e-16f
#define INV_EPSILON_F 1e16f

#define RAY_OFFSET 0.000001f


static const Vec3f VEC3F_ZERO = 0.f;
static const Vec3f VEC3F_ONES = 1.f;

static const Box3f BOX3F_UNIT = Box3f(VEC3F_ZERO, VEC3F_ONES);

typedef unsigned int uint;

struct float2 { float v0, v1; };
struct float3 { float v0, v1, v2; };
struct float4 { float v0, v1, v2, v3; };
struct uint3 { uint v0, v1, v2; };
struct uint4 { uint v0, v1, v2, v3; };
