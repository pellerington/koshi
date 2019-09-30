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
    virtual const Vec3f get_emission() = 0;
    // Global variables should be set in the construtor. Instanced variables should be set in the instance method (ie texture evalutation).
    virtual std::shared_ptr<Material> instance(const Surface &surface) = 0;
    virtual bool sample_material(const Surface &surface, std::deque<MaterialSample> &samples, const float sample_reduction = 1.f) = 0;
    virtual bool evaluate_material(const Surface &surface, MaterialSample &sample) = 0;
private:

};
