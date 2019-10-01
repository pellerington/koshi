#pragma once

#include <vector>
#include <queue>
#include "../Math/Types.h"
#include "../Util/Surface.h"
class Surface;

#define UNIFORM_SAMPLE false

struct MaterialSample
{
    Vec3f wo;
    Vec3f fr;
    float pdf;
    float quality = 1.f;
};

class Material
{
public:
    virtual const Vec3f get_emission() { return VEC3F_ZERO; }
    // Global variables should be set in the construtor. Instanced variables should be set in the instance method (ie texture evalutation).
    virtual std::shared_ptr<Material> instance(const Surface &surface) { return std::shared_ptr<Material>(new Material()); }
    virtual bool sample_material(const Surface &surface, std::deque<MaterialSample> &samples, const float sample_reduction = 1.f) { return false; }
    virtual bool evaluate_material(const Surface &surface, MaterialSample &sample) { return false; }
private:

};
