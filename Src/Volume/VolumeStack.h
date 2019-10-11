#pragma once

#include "Volume.h"

#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <memory>

struct VolumeIntersect //VolumeIntersect??? Then make VolumeIntersect VolumeTemp??
{
    float tmin, tmax;
    Vec3f max_density, min_density;
    std::vector<Volume*> volume_prop;
    //Vector<Vec3f> UVW_BEGIN, UVW_LEN
};

class VolumeStack
{
public:
    VolumeStack(const std::unordered_set<Volume*> * entry_volumes = nullptr)
    {
        if(entry_volumes)
        for(auto it = entry_volumes->begin(); it != entry_volumes->end(); it++)
            hits[0.f].push_back(VolumeHit(true, *it));
    }

    inline void add_intersect(const float &t, const std::shared_ptr<Volume> &volume) {
        hits[t].push_back(VolumeHit(true, volume.get()));
    }
    inline void sub_intersect(const float &t, const std::shared_ptr<Volume> &volume) {
        hits[t].push_back(VolumeHit(false, volume.get()));
    }

    inline const VolumeIntersect& operator[](const int i) const { return volumes[i]; }
    inline const uint size() const { return volumes.size(); }
    inline const auto begin() const { return volumes.begin(); }
    inline const auto end() const { return volumes.end(); }

    void build(const float &tend);

    inline const std::unordered_set<Volume*> * exit_volumes() const { return &volume_tracker; }

private:
    struct VolumeHit {
        VolumeHit(bool enter, Volume * volume) : enter(enter), volume(volume) {}
        bool enter;
        Volume * volume;
    };
    std::map<float, std::vector<VolumeHit>> hits;

    std::unordered_set<Volume*> volume_tracker;

    std::vector<VolumeIntersect> volumes;
};
