// #include <Volume/Volume.h>

// Volume::Volume(const Vec3f &_density, const std::shared_ptr<Texture> _density_texture, const Vec3f &_scattering, const float &g, const Vec3f &_emission)
// : density(_density), density_texture(_density_texture), scattering(_scattering), emission(_emission), g(g), g_sqr(g*g), g_inv(1.f/g), g_abs(fabs(g))
// {
//     // if(!is_heterogeneous())
//     // {
//         max_density = density;
//     // }
// }

// bool Volume::sample_volume(const Vec3f &wi, VolumeSample &sample, const Vec2f &rand)
// {
//     const float theta = TWO_PI * rand[0];

//     float cos_phi = (g_abs < EPSILON_F) ?  1.f - 2.f * rand[1] : (0.5f * g_inv) * (1.f + g_sqr - std::pow((1.f - g_sqr) / (1.f - g + 2.f * g * rand[1]), 2));
//     float sin_phi = sqrtf(std::max(EPSILON_F, 1.f - cos_phi * cos_phi));

//     const float x = sin_phi * cosf(theta), z = sin_phi * sinf(theta), y = cos_phi;
//     sample.wo = Transform3f::basis_transform(wi) * Vec3f(x, y, z);

//     // sample.pdf = INV_FOUR_PI * (1.f - g_sqr) / std::pow(1.f + g_sqr - 2.f * g * cos_phi, 1.5f);

//     return true;
// }

// bool Volume::evaluate_volume(const Vec3f &wi, VolumeSample &sample)
// {
//     return false;
// }

// Vec3f Volume::get_density(const Vec3f &uvw, Resources &resources)
// {
//     if(density_texture)
//         return density * density_texture->get_float(uvw.u, uvw.v, uvw.w, resources);
//     else
//         return density;
// }
// Vec3f Volume::get_scattering(const Vec3f &uvw, Resources &resources)
// {
//     return scattering;
// }
// Vec3f Volume::get_emission(const Vec3f &uvw, Resources &resources)
// {
//     return emission;
// }
