#pragma once

#include <string>
#include <OpenImageIO/texture.h>

#include <Textures/Texture.h>

class Image : public Texture
{
public:
    Image(const std::string &_filename, const bool smooth) : filename(_filename)
    {
        texture_system = OIIO::TextureSystem::create();
        // texture_system->attribute("max_memory_MB", 1000.0f);

        options.mipmode = OIIO::TextureOpt::MipModeDefault;
        options.interpmode = smooth ? OIIO::TextureOpt::InterpSmartBicubic : OIIO::TextureOpt::InterpClosest;
    }

    const Vec3f get_vec3f(const float &u, const float &v, const float &w, Resources &resources)
    {
        float result[3];
        if(!texture_system->texture(filename, options, u, v, 0.f, 0.f, 0.f, 0.f, 3, result))
            return Vec3f(0.0, 0.0, 0.0);

        return Vec3f(result[0], result[1], result[2]);
    }

private:
    OIIO::ustring filename;
    OIIO::TextureSystem * texture_system;
    OIIO::TextureOpt options;
};
