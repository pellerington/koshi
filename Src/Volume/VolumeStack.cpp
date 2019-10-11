#include "VolumeStack.h"

void VolumeStack::build(const float &tend)
{
    volume_tracker.clear();

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
            if(cv != volume_tracker.end() && !hit_volume->enter)
                volume_tracker.erase(cv);
            else if(cv == volume_tracker.end() && hit_volume->enter)
                volume_tracker.insert(hit_volume->volume);
        }

        // If we have volumes on the stack start a new volume
        if(volume_tracker.size() > 0)
        {
            volumes.push_back(VolumeIntersect());
            volumes.back().tmin = hit->first;
            volumes.back().max_density = 0.f;
            for(auto volume = volume_tracker.begin(); volume != volume_tracker.end(); volume++)
                volumes.back().max_density += (*volume)->max_density;
        }
    }

    if(volume_tracker.size() > 0)
        volumes.back().tmax = tend;
}
