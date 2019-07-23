#include "Image.h"

#include <iostream>

Image::Image(const std::string &filename, const bool smooth)
{
    texture_system = OpenImageIO::TextureSystem::create();
    texture_system->attribute("automip", 1);

    texture_handle = texture_system->get_texture_handle(OpenImageIO::ustring(filename));
    options.interpmode = smooth ? OpenImageIO::TextureOpt::InterpSmartBicubic : OpenImageIO::TextureOpt::InterpClosest;
}

const bool Image::get_vec3f(const float u, const float v, Vec3f &out)
{
    float result[3];
    if(!texture_system->texture(texture_handle, texture_system->get_perthread_info(), options, u, v, 0.f, 0.f, 0.f, 0.f, 3, result))
        return false;

    out[0] = result[0];
    out[1] = result[1];
    out[2] = result[2];

    return true;
}
