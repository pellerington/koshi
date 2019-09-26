#include "Image.h"

#include <iostream>

Image::Image(const std::string &_filename, const bool smooth) : filename(_filename)
{
    texture_system = OIIO::TextureSystem::create();
    //texture_system->attribute("max_memory_MB", 1000.0f);

    // texture_handle = texture_system->get_texture_handle(filename);

    options.mipmode = OIIO::TextureOpt::MipModeDefault;
    options.interpmode = smooth ? OIIO::TextureOpt::InterpSmartBicubic : OIIO::TextureOpt::InterpClosest;
}

const bool Image::get_vec3f(const float u, const float v, Vec3f &out)
{
    float result[3];
    if(!texture_system->texture(filename, options, u, v, 0.f, 0.f, 0.f, 0.f, 3, result))
        return false;

    out[0] = result[0];
    out[1] = result[1];
    out[2] = result[2];

    return true;
}
