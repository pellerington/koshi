#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "Vec3f.h"
#include "Vec2.h"
#include "Transform3f.h"
#include "Box3f.h"

#define HALF_PI 1.57079632679
#define PI 3.14159265359
#define TWO_PI 6.28318530718

#define INV_PI 0.31830988618
#define INV_TWO_PI 0.15915494309

#define EPSILON_F 0.000001f
#define DELTA_PDF 1e9

static const Vec3f VEC3F_ZERO = 0.f;
static const Vec3f VEC3F_ONES = 1.f;

static const Box3f BOX3F_UNIT = Box3f(Vec3f(0.f), Vec3f(1.f));

typedef unsigned int uint;

struct VERT_DATA { float x, y, z, r; };
struct NORM_DATA { float x, y, z; };
struct TRI_DATA { uint v0, v1, v2; };
struct QUAD_DATA { uint v0, v1, v2, v3; };
