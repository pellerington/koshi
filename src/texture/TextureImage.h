#pragma once

#include <string>
#include <OpenImageIO/texture.h>

#include <texture/Texture.h>
 
// TODO: include more options such as MipMode and InterpMode
class TextureImage : public Texture
{
public:
    TextureImage(const std::string& filename, const bool& smooth) : filename(filename)
    {
        texture_system = OIIO::TextureSystem::create();
        // texture_system->attribute("max_memory_MB", 1000.0f);

        options.mipmode = OIIO::TextureOpt::MipModeDefault;
        options.interpmode = smooth ? OIIO::TextureOpt::InterpSmartBicubic : OIIO::TextureOpt::InterpClosest;
    
        resolution = VEC3F_ONES;
        inv_resolution = VEC3F_ONES;
        auto in = OIIO::ImageInput::open(filename);
        if (!in) return;
        const OIIO::ImageSpec &spec = in->spec();
        resolution.x = spec.width;
        resolution.y = spec.height;
        inv_resolution = 1.f / resolution;
    }

    Vec3f evaluate(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) const
    {
        float result[3];
        if(!texture_system->texture(filename, options, u, v, 0.f, 0.f, 0.f, 0.f, 3, result))
            return Vec3f(0.f, 0.f, 0.f);

        return Vec3f(result[0], result[1], result[2]);
    }

    Vec3f delta() const { return inv_resolution; }

    // TODO: Also include if we cannot read the file.
    bool null() const { return (filename == ""); }

private:
    OIIO::ustring filename;
    OIIO::TextureSystem * texture_system; // TODO: Have this be global
    mutable OIIO::TextureOpt options;

    Vec3f resolution;
    Vec3f inv_resolution;
};
