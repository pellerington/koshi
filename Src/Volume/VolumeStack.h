#pragma once

#include "Volume.h"

#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <memory>

struct VolumeIntersect
{
    float tmin, tmax;
    Vec3f max_density, min_density;
    std::vector<Volume*> volumes;
    //Vector<Vec3f> UVW_BEGIN, UVW_LEN
};

class VolumeStack
{
public:
    VolumeStack(const std::vector<Volume*> * entry_volumes = nullptr)
    : object_volume(nullptr)
    {
        if(entry_volumes)
        for(auto it = entry_volumes->begin(); it != entry_volumes->end(); it++)
            hits[0.f].push_back(VolumeHit(true, *it));
    }

    inline void add_intersect(const float &t, const std::shared_ptr<Volume> &volume, bool surface = false) {
        if(!surface)
            hits[t].push_back(VolumeHit(true, volume.get()));
        else
            object_volume = volume.get();
    }
    inline void sub_intersect(const float &t, const std::shared_ptr<Volume> &volume) {
        hits[t].push_back(VolumeHit(false, volume.get()));
    }

    inline const auto operator[](const float &t) const {
        if(t < tmin || t > tmax)
            return volumes.end();
        return std::lower_bound(volumes.begin(), volumes.end(), t, [](const VolumeIntersect &v, const float &t) -> bool { return t < v.tmin; });
    }
    inline const auto begin() const { return volumes.begin(); }
    inline const auto end() const { return volumes.end(); }
    inline const uint num_volumes() const { return volumes.size(); }

    void build(const float &tend);

    inline const std::vector<Volume*> * get_exit_volumes() const { return &exit_volumes; }
    inline const std::vector<Volume*> * get_enter_volumes() const { return &enter_volumes; }

    float tmin, tmax;

private:
    struct VolumeHit {
        VolumeHit(bool enter, Volume * volume) : enter(enter), volume(volume) {}
        bool enter;
        Volume * volume;
    };
    std::map<float, std::vector<VolumeHit>> hits;

    std::vector<Volume*> exit_volumes;

    Volume * object_volume;
    std::vector<Volume*> enter_volumes;

    std::vector<VolumeIntersect> volumes;

};
