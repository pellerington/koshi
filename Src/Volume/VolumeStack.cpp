#include "VolumeStack.h"

void VolumeStack::build(const float &tend)
{
    std::unordered_set<const VolumeProperties*> curr_volume_props;
    for(auto isect = intersects.begin(); isect != intersects.end(); isect++)
    {
        if(isect->first > tend)
            break;

        // If we have atleast one volume and things on the stack, add the tmax to the last
        if(curr_volume_props.size() > 0 && volumes.size() > 0)
            volumes.back().tmax = isect->first;

        // Update current volumes
        for(auto isect_volume = isect->second.begin(); isect_volume != isect->second.end(); isect_volume++)
        {
            auto cv = std::find(curr_volume_props.begin(), curr_volume_props.end(), isect_volume->volume_prop.get());
            if(cv != curr_volume_props.end() && !isect_volume->add)
                curr_volume_props.erase(cv);
            else if(cv == curr_volume_props.end() && isect_volume->add)
                curr_volume_props.insert(isect_volume->volume_prop.get());
        }

        // If we have volumes on the stack start a new volume
        if(curr_volume_props.size() > 0)
        {
            volumes.push_back(Volume());
            volumes.back().tmin = isect->first;
            volumes.back().density = 0.f;
            for(auto vol_prop = curr_volume_props.begin(); vol_prop != curr_volume_props.end(); vol_prop++)
                volumes.back().density += (*vol_prop)->density;
        }
    }

    if(curr_volume_props.size() > 0)
        volumes.back().tmax = tend;
}
