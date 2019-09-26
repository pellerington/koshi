#pragma once

#include <string>
#include <OpenImageIO/texture.h>

#include "Texture.h"

class Image : public Texture
{
public:
    Image(const std::string &filename, const bool smooth = true);
    const bool get_vec3f(const float u, const float v, Vec3f &out);
    inline const bool get_vec3f(const Surface &surface, Vec3f &out) { return get_vec3f(surface.u, surface.v, out); }

private:
    OIIO::ustring filename;
    OIIO::TextureSystem * texture_system;
    // OIIO::TextureSystem::TextureHandle * texture_handle;
    OIIO::TextureOpt options;
};
