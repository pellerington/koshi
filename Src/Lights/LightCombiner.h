#pragma once

#include "../Objects/Object.h"

class LightCombiner : public Object
{
public:
    LightCombiner() : Object() { set_null_rtc_geometry(); }

    Type get_type() { return Object::LightCombiner; }

    void add_light(std::shared_ptr<Object> light)
    {
        lights.push_back(light);
    }

    bool evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample)
    {
        if(!lights.size())
            return false;

        for(size_t i = 0; i < lights.size(); i++)
        {
            LightSample isample;
            if(lights[i]->evaluate_light(intersect, pos, pfar, isample))
            {
                light_sample.intensity += isample.intensity;
                light_sample.pdf += isample.pdf;
            }
        }
        return true;
    }

private:
    std::vector<std::shared_ptr<Object>> lights;
};
