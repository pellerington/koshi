#pragma once

#include <koshi/integrator/Integrator.h>

#include <koshi/geometry/Volume.h>
#include <koshi/material/MaterialVolume.h>

// TODO: Figure out how to merge this with the absorption medium/into interiors. Need enough information in volume to not need the material...

class VolumeSingleScatter : public Integrator
{
public:
    struct VolumeSingleScatterData { const Volume * volume; };
    struct VolumeSingleScatterHeterogenousData : public VolumeSingleScatterData 
    {
        VolumeSingleScatterHeterogenousData(Resources& resources) : samples(resources.memory /* TODO: More samples by default ??? */) {}
        struct TransmittanceSample { Vec3f transmittance; float t; };
        Array<TransmittanceSample> samples;
    };
    void * pre_integrate(const Intersect * intersect, Resources& resources);
    Vec3f integrate(const Intersect * intersect, void * data, Transmittance& transmittance, Resources& resources) const;
    Vec3f shadow(const float& t, const Intersect * intersect, void * data, Resources& resources) const;
};