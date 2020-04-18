#pragma once

#include  <geometry/Geometry.h>

class LightCombiner : public Geometry
{
public:
    LightCombiner() : Geometry() {}

    void add_light(std::shared_ptr<Geometry> light)
    {
        lights.push_back(light);
    }

    bool evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources)
    {
        if(!lights.size())
            return false;

        for(size_t i = 0; i < lights.size(); i++)
        {
            LightSample isample;
            if(lights[i]->evaluate_light(intersect, pos, pfar, isample, resources))
            {
                light_sample.intensity += isample.intensity;
                light_sample.pdf += isample.pdf;
            }
        }
        return true;
    }

private:
    std::vector<std::shared_ptr<Geometry>> lights;
};
