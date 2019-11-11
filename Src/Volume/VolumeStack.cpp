#include "VolumeStack.h"

void VolumeStack::build(const float &tend)
{
    std::unordered_set<Volume*> volume_tracker;

    for(auto hit = hits.begin(); hit != hits.end(); hit++)
    {
        if(hit->first > tend)
            break;

        // If we have atleast one volume and things on the stack, add the tmax to the last
        if(volume_tracker.size() > 0 && volumes.size() > 0)
            volumes.back().tmax = hit->first;

        // Update current volumes
        for(auto hit_volume = hit->second.begin(); hit_volume != hit->second.end(); hit_volume++)
        {
            auto cv = std::find(volume_tracker.begin(), volume_tracker.end(), hit_volume->volume);
            if(cv != volume_tracker.end() && !hit_volume->add)
                volume_tracker.erase(cv);
            else if(cv == volume_tracker.end() && hit_volume->add)
                volume_tracker.insert(hit_volume->volume);
        }

        // If we have volumes on the stack start a new volume
        if(volume_tracker.size() > 0)
        {
            volumes.emplace_back();
            volumes.back().tmin = hit->first;
            volumes.back().max_density = 0.f;
            for(auto volume = volume_tracker.begin(); volume != volume_tracker.end(); volume++)
            {
                volumes.back().max_density += (*volume)->max_density;
                volumes.back().volumes.push_back(*volume);
            }
        }
    }

    if(volume_tracker.size() > 0)
    {
        volumes.back().tmax = tend;
        exit_volumes.insert(exit_volumes.end(), volume_tracker.begin(), volume_tracker.end());
        inside_object_volumes.insert(inside_object_volumes.end(), volume_tracker.begin(), volume_tracker.end());
    }

    tmin = tmax = 0.f;
    if(volumes.size() > 0)
    {
        tmin = volumes.front().tmin;
        tmax = volumes.back().tmax;
    }
}
