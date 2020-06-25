#pragma once

#include <Util/Resources.h>
#include <base/Object.h>

#include <iostream>

class Texture : public Object
{
public:

    // TODO: Make this Vec4f
    // TODO: Make this evaluate private so we can pick uvw at a higher level?
    // Returns a value for the specific u, v and w.
    virtual Vec3f evaluate(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) const = 0;

    // The size of each discrete value in the texture. Zero if procedural, One if constant. Generally the inverse of the resolution.
    virtual Vec3f delta() const = 0;

    // Returns true if the texture will always return black.
    virtual bool null() const = 0;

    // Used to access specific return types.
    template<typename T>
    T evaluate(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) const;
};
