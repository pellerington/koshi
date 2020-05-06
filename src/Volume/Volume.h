// #pragma once

// #include <Math/Vec3f.h>
// #include <materials/Material.h>
// #include <Math/RNG.h>
// class Volume;

// struct VolumeSample : MaterialSample
// {
//     Vec3f pos;
//     const std::vector<Volume*> * passthrough_volumes;
// };

// class Volume
// {
// public:
//     Volume(const Vec3f &_density, const std::shared_ptr<Texture> _density_texture, const Vec3f &scattering = VEC3F_ZERO, const float &g = 0.f, const Vec3f &emission = VEC3F_ZERO);
//     Volume(const Volume &volume, const Transform3f * _world_to_obj) : Volume(volume) { world_to_obj = _world_to_obj; }

//     // This shouldn't be public and changable.
//     Vec3f max_density, min_density;

//     virtual bool is_heterogeneous() { return false; }
//     // is_multiscattering???
//     // is_exclusive???

//     virtual bool sample_volume(const Vec3f &wi, VolumeSample &sample, const Vec2f &rand); // UVW as well?
//     virtual bool evaluate_volume(const Vec3f &wi, VolumeSample &sample); // Needs wo // UVW as well?

//     virtual Vec3f get_density(const Vec3f &uvw, Resources &resources);
//     virtual Vec3f get_scattering(const Vec3f &uvw, Resources &resources);
//     virtual Vec3f get_emission(const Vec3f &uvw, Resources &resources);

//     const Transform3f * world_to_obj;

// private:

//     //MAYBE DON'T PASS IN TEXTURE AND JUST USE ATTRIBUTES
//     //BOUNDARIES WILL BE DECIDEd BY OBJECTVOLUME AND IT WOULD BE GOOD TO HAVE ATTRIBUTES FIND THERE MIN/MAX ANYAWY
//     const Vec3f density;
//     const std::shared_ptr<Texture> density_texture;

//     const Vec3f scattering;
//     const Vec3f emission;

//     const float g;
//     const float g_sqr;
//     const float g_inv;
//     const float g_abs;
// };
