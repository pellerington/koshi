#pragma once

class Fresnel
{
public:
    enum Types { None, Metalic, Dielectric, Conductor, Diffuse };
    virtual Vec3f Fr(const float& cosi) = 0;
    virtual Vec3f Ft(const float& cosi) { return 1.f - Fr(cosi); }
};

class FresnelNone : public Fresnel
{
public:
    Vec3f Fr(const float& cosi) { return VEC3F_ONES; }
    Vec3f Ft(const float& cosi) { return VEC3F_ONES; }
};

class FresnelMetalic : public Fresnel
{
public:
    FresnelMetalic(const Vec3f& F0) : F0(F0) {}
    Vec3f Fr(const float& cosi)
    {
        return F0 + (1.f - F0) * std::pow(1.f - cosi, 5);
    }
private:
    const Vec3f F0;
};

class FresnelDielectric : public Fresnel
{
public:
    FresnelDielectric(const float& ior_in, const float& ior_out)
    : eta(ior_in / ior_out), F0(std::pow(std::abs((ior_out - ior_in) / (ior_out + ior_in)), 2.f)) {}
    Vec3f Fr(const float& cosi)
    {
        if(1.f - eta * eta * (1.f - cosi * cosi) < 0.f)
            return 1.f;
        return F0 + (1.f - F0) * std::pow(1.f - cosi, 5);
    }
private:
    const float eta;
    const float F0;
};
