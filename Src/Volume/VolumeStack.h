#pragma once

#include "Volume.h"

#include <vector>
#include <map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <memory>

class VolumeStack
{
public:
    VolumeStack(std::unordered_set<VolumeProperties*> * entry_volumes = nullptr)
    {
        if(entry_volumes)
        for(auto it = entry_volumes->begin(); it != entry_volumes->end(); it++)
            intersects[0.f].emplace_back(*it, true);
    }

    inline void add_intersect(const float &t, const std::shared_ptr<VolumeProperties> &volume_prop) { intersects[t].emplace_back(volume_prop, true); }
    inline void sub_intersect(const float &t, const std::shared_ptr<VolumeProperties> &volume_prop) { intersects[t].emplace_back(volume_prop, false); }

    inline const Volume& operator[](const int i) const { return volumes[i]; }
    inline const auto begin() const { return volumes.begin(); }
    inline const auto end() const { return volumes.end(); }
    inline const uint size() const { return volumes.size(); }

    void build(const float &tend);

    inline std::unordered_set<VolumeProperties*> * exit_volumes() { return &volume_prop_tracker; }

private:
    std::unordered_set<VolumeProperties*> volume_prop_tracker;

    struct VolumeIntersect
    {
        VolumeIntersect(const std::shared_ptr<VolumeProperties> &volume_prop, const bool add) : volume_prop(volume_prop.get()), add(add) {}
        VolumeIntersect(VolumeProperties * volume_prop, const bool add) : volume_prop(volume_prop), add(add) {}
        VolumeProperties * volume_prop;
        bool add;
    };
    std::map<float, std::vector<VolumeIntersect>> intersects;

    std::vector<Volume> volumes;
};
