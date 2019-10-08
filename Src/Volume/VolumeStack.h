#pragma once

#include "Volume.h"

#include <vector>
#include <map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <memory>

struct VolumeIntersect {
    VolumeIntersect(const std::shared_ptr<VolumeProperties> &volume_prop, const bool add) : volume_prop(volume_prop), add(add) {}
    const std::shared_ptr<VolumeProperties> volume_prop;
    const bool add;
};

class VolumeStack
{
public:
    inline void add_intersect(const float &t, const std::shared_ptr<VolumeProperties> &volume_prop) { intersects[t].emplace_back(volume_prop, true); }
    inline void sub_intersect(const float &t, const std::shared_ptr<VolumeProperties> &volume_prop) { intersects[t].emplace_back(volume_prop, false); }

    inline const Volume& operator[](const int i) const { return volumes[i]; }
    inline const auto begin() const { return volumes.begin(); }
    inline const auto end() const { return volumes.end(); }
    inline const uint size() const { return volumes.size(); }

    void build(const float &tend);

private:
    std::map<float, std::vector<VolumeIntersect>> intersects;
    std::vector<Volume> volumes;
};
