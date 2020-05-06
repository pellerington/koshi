// Fill this out later AND! use it.

inline bool get_refraction(const GeometrySurface &surface, const float &eta, Vec3f &out)
{
    //Dont use surface pass in normal wi and ior_in
    float n_dot_wi = clamp(surface.normal.dot(surface.wi), -1.f, 1.f);
    float k = 1.f - eta * eta * (1.f - n_dot_wi * n_dot_wi);
    if(k < 0) return false;

    return eta * surface.wi + (eta * fabs(n_dot_wi) - sqrtf(k)) * ((surface.facing) ? surface.normal : -surface.normal);
}
