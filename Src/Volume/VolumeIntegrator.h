#pragma once

#include "VolumeStack.h"

class VolumeIntegrator
{
public:
    VolumeIntegrator(const VolumeStack& volumes) : volumes(volumes){}

    Vec3f shadow(const float &t)
    {
        Vec3f tr = VEC3F_ONES;
        for(auto curr_volume = volumes.begin(); curr_volume != volumes.end(); curr_volume++)
        {
            if(t < curr_volume->tmax)
            {
                tr *= Vec3f::exp(curr_volume->density * (curr_volume->tmin - t));
                break;
            }
            else
                tr *= Vec3f::exp(curr_volume->density * (curr_volume->tmin - curr_volume->tmax));
        }

        return tr;
    }

private:
    const VolumeStack& volumes;
};
