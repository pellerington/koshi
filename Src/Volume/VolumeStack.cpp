#include "VolumeStack.h"

void VolumeStack::build(const float &tend)
{
    volume_prop_tracker.clear();

    for(auto isect = intersects.begin(); isect != intersects.end(); isect++)
    {
        if(isect->first > tend)
            break;

        // If we have atleast one volume and things on the stack, add the tmax to the last
        if(volume_prop_tracker.size() > 0 && volumes.size() > 0)
            volumes.back().tmax = isect->first;

        // Update current volumes
        for(auto isect_volume = isect->second.begin(); isect_volume != isect->second.end(); isect_volume++)
        {
            auto cv = std::find(volume_prop_tracker.begin(), volume_prop_tracker.end(), isect_volume->volume_prop);
            if(cv != volume_prop_tracker.end() && !isect_volume->add)
                volume_prop_tracker.erase(cv);
            else if(cv == volume_prop_tracker.end() && isect_volume->add)
                volume_prop_tracker.insert(isect_volume->volume_prop);
        }

        // If we have volumes on the stack start a new volume
        if(volume_prop_tracker.size() > 0)
        {
            volumes.push_back(Volume());
            volumes.back().tmin = isect->first;
            volumes.back().density = 0.f;
            for(auto vol_prop = volume_prop_tracker.begin(); vol_prop != volume_prop_tracker.end(); vol_prop++)
                volumes.back().density += (*vol_prop)->density;
        }
    }

    if(volume_prop_tracker.size() > 0)
        volumes.back().tmax = tend;
}
