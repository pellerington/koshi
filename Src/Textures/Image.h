#pragma once

#include <string>
#include <OpenImageIO/texture.h>

#include "Texture.h"

class Image : public Texture
{
public:
    Image(const std::string &filename, const bool smooth = true);
    const bool get_vec3f(const float &u, const float &v, const float &w, Vec3f &out);


private:
    OIIO::ustring filename;
    OIIO::TextureSystem * texture_system;
    // OIIO::TextureSystem::TextureHandle * texture_handle;
    OIIO::TextureOpt options;
};
