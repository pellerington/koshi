// #pragma once

// #include <Volume/Volume.h>
// #include <intersection/Ray.h>

// #include <vector>
// #include <map>
// #include <set>
// #include <algorithm>
// #include <iostream>
// #include <memory>

// struct VolumeIntersect
// {
//     float tmin, tmax;
//     float tlen, inv_tlen;
//     Vec3f max_density, min_density;
//     std::vector<Volume*> volumes;

//     std::vector<Vec3f> uvw_min;
//     std::vector<Vec3f> uvw_len;
// };

// class VolumeStack
// {
// public:
//     VolumeStack(const Ray &ray, const std::vector<Volume*> * _passthrough_volumes = nullptr)
//     : ray(ray)
//     {
//         if(_passthrough_volumes)
//         for(auto it = _passthrough_volumes->begin(); it != _passthrough_volumes->end(); it++)
//             hits[0.f].push_back(VolumeHit(true, *it));
//     }

//     inline void add_intersect(const float &t, const std::shared_ptr<Volume> &volume, bool end = false) {
//         if(!end)
//             hits[t].push_back(VolumeHit(true, volume.get()));
//         else
//             passthrough_transmission_volumes.push_back(volume.get());
//     }
//     inline void sub_intersect(const float &t, const std::shared_ptr<Volume> &volume) {
//         hits[t].push_back(VolumeHit(false, volume.get()));
//     }

//     inline const auto operator[](const float &t) const {
//         if(t < tmin || t > tmax)
//             return volumes.end();
//         return std::lower_bound(volumes.begin(), volumes.end(), t, [](const VolumeIntersect &v, const float &t) -> bool { return t < v.tmin; });
//     }
//     inline const auto begin() const { return volumes.begin(); }
//     inline const auto end() const { return volumes.end(); }
//     inline const uint num_volumes() const { return volumes.size(); }

//     void build(const float &tend);

//     inline const std::vector<Volume*> * get_passthrough_volumes() const { return &passthrough_volumes; }
//     inline const std::vector<Volume*> * get_passthrough_transmission_volumes() const { return &passthrough_transmission_volumes; }

//     const Ray ray;
//     float tmin, tmax; // should be private with accessors?

// private:
//     struct VolumeHit {
//         VolumeHit(const bool add, Volume * volume)
//         : add(add), volume(volume) {}
//         const bool add;
//         Volume * volume;
//     };
//     std::map<float, std::vector<VolumeHit>> hits;

//     std::vector<Volume*> passthrough_volumes;
//     std::vector<Volume*> passthrough_transmission_volumes;

//     std::vector<VolumeIntersect> volumes;
// };
