#pragma once

struct Intersect
{
    Intersect(const std::shared_ptr<Object> &object, const Surface &surface, const VolumeStack &volumes)
    : object(object), surface(surface), volumes(volumes) {}
    const std::shared_ptr<Object> object;
    const Surface surface;
    const VolumeStack volumes;
    // Also incoming ray?
};
