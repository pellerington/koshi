// #pragma once

// #include <Volume/VolumeIntegrator.h>

// // Find a way to make ptrs so we don't need this.
// struct MultiScatData : public MaterialSample::Data {
//     Vec3f weight_history;
// };

// class MultiScatVolumeIntegrator : public VolumeIntegrator
// {
// public:
//     MultiScatVolumeIntegrator(Scene * scene, Ray& ray, const VolumeStack& volumes, const VolumeSample * in_sample, Resources& resources);

//     Vec3f shadow(const float& t);

//     Vec3f emission(/* float pdf for direct sampling???*/);

//     void scatter(std::vector<VolumeSample>& samples);

// private:
//     Vec3f weight;
//     float tmax;

//     Vec3f weighted_emission;

//     bool has_scatter;
//     VolumeSample sample;
//     MultiScatData data;
// };
