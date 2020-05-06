// #pragma once

// #include <Volume/VolumeIntegrator.h>

// class ZeroScatVolumeIntegrator : public VolumeIntegrator
// {
// public:
//     ZeroScatVolumeIntegrator(Scene * scene, Ray &ray, const VolumeStack& volumes, Resources &resources) : VolumeIntegrator(scene, ray, volumes, resources) {}

//     Vec3f shadow(const float &t)
//     {
//         Vec3f tr = VEC3F_ONES;
//         for(auto curr_volume = volumes.begin(); curr_volume != volumes.end(); curr_volume++)
//         {
//             // TODO: Multiply shadow per volume ( so we can do homo easy and hetro using residual ratio )
//             if(t < curr_volume->tmax)
//             {
//                 tr *= Vec3f::exp(curr_volume->max_density * (curr_volume->tmin - t));
//                 break;
//             }
//             else
//                 tr *= Vec3f::exp(curr_volume->max_density * (curr_volume->tmin - curr_volume->tmax));
//         }
//         return tr;
//     }

//     Vec3f emission(/* float pdf for direct sampling???*/)
//     {
//         return VEC3F_ZERO;
//     }

//     void scatter(std::vector<VolumeSample> &samples)
//     {
//     }
// };
