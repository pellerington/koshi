// #include <Volume/VolumeStack.h>

// void VolumeStack::build(const float& tend)
// {
//     std::unordered_set<Volume*> volume_tracker;

//     for(auto hit = hits.begin(); hit != hits.end(); hit++)
//     {
//         if(hit->first > tend)
//             break;

//         // If we have atleast one volume and things on the stack, complete the previous volume_tracker
//         if(volume_tracker.size() > 0 && volumes.size() > 0)
//         {
//             volumes.back().tmax = hit->first;
//             volumes.back().tlen = volumes.back().tmax - volumes.back().tmin;
//             volumes.back().inv_tlen = 1.f / volumes.back().tlen;

//             const Vec3f pmin = ray.get_position(volumes.back().tmin);
//             const Vec3f pmax = ray.get_position(volumes.back().tmax);

//             volumes.back().uvw_min.resize(volumes.back().volumes.size());
//             volumes.back().uvw_len.resize(volumes.back().volumes.size());
//             for(uint i = 0; i < volumes.back().volumes.size(); i++)
//             {
//                 volumes.back().uvw_min[i] = volumes.back().volumes[i]->world_to_obj->multiply(pmin);
//                 volumes.back().uvw_len[i] = volumes.back().volumes[i]->world_to_obj->multiply(pmax) - volumes.back().uvw_min[i];
//             }
//         }

//         // Add or remove volumes we are tracking
//         for(auto hit_volume = hit->second.begin(); hit_volume != hit->second.end(); hit_volume++)
//         {
//             auto cv = std::find(volume_tracker.begin(), volume_tracker.end(), hit_volume->volume);
//             if(cv != volume_tracker.end() && !hit_volume->add)
//                 volume_tracker.erase(cv);
//             else if(cv == volume_tracker.end() && hit_volume->add)
//                 volume_tracker.insert(hit_volume->volume);
//         }

//         // If we have volumes to track, add a new volume
//         if(volume_tracker.size() > 0)
//         {
//             volumes.emplace_back();
//             volumes.back().tmin = hit->first;
//             volumes.back().max_density = 0.f;
//             for(auto volume = volume_tracker.begin(); volume != volume_tracker.end(); volume++)
//             {
//                 volumes.back().max_density += (*volume)->max_density;
//                 volumes.back().volumes.push_back(*volume);
//             }
//         }
//     }

//     // Finish the final volumes off if we have any left on the stack.
//     if(volume_tracker.size() > 0)
//     {
//         volumes.back().tmax = tend;
//         volumes.back().tlen = volumes.back().tmax - volumes.back().tmin;
//         volumes.back().inv_tlen = 1.f / volumes.back().tlen;

//         const Vec3f pmin = ray.get_position(volumes.back().tmin);
//         const Vec3f pmax = ray.get_position(volumes.back().tmax);

//         volumes.back().uvw_min.resize(volumes.back().volumes.size());
//         volumes.back().uvw_len.resize(volumes.back().volumes.size());
//         for(uint i = 0; i < volumes.back().volumes.size(); i++)
//         {
//             volumes.back().uvw_min[i] = volumes.back().volumes[i]->world_to_obj->multiply(pmin);
//             volumes.back().uvw_len[i] = volumes.back().volumes[i]->world_to_obj->multiply(pmax) - volumes.back().uvw_min[i];
//         }

//         passthrough_volumes.insert(passthrough_volumes.end(), volume_tracker.begin(), volume_tracker.end());
//         passthrough_transmission_volumes.insert(passthrough_transmission_volumes.end(), volume_tracker.begin(), volume_tracker.end());
//     }

//     tmin = tmax = 0.f;
//     if(volumes.size() > 0)
//     {
//         tmin = volumes.front().tmin;
//         tmax = volumes.back().tmax;
//     }
// }
