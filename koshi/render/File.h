#pragma once

#include <OpenImageIO/imageio.h>

#include <koshi/render/Render.h>

class File
{
public:
    static void Save(const Render * render, const std::string& aov_name, const std::string& filename)
    {
        const uint xres = render->get_image_resolution().x, yres = render->get_image_resolution().y;
        const uint channels = 3;
        float pixels[xres*yres*channels];
        for(uint x = 0; x < xres; x++)
        for(uint y = 0; y < yres; y++)
        {
            Vec3f color = render->get_pixel(aov_name, x, y);
            pixels[(x + y*xres)*3 + 0] = color.r;
            pixels[(x + y*xres)*3 + 1] = color.g;
            pixels[(x + y*xres)*3 + 2] = color.b;
        }

        OIIO::ImageOutput::unique_ptr out = OIIO::ImageOutput::create(filename);
        OIIO::ImageSpec spec (xres, yres, channels, OIIO::TypeDesc::FLOAT);
        out->open(filename, spec);
        out->write_image(OIIO::TypeDesc::FLOAT, pixels);
        out->close();
    }
};
