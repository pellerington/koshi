#pragma once

#include <OpenImageIO/imageio.h>

class OIIOViewer
{
public:
    static void FileOut(const Render * render, const char * filename)
    {
        std::vector<std::string> aov_list = render->get_aov_list();

        for(uint i = 0; i < aov_list.size(); i++)
        {
            std::string aov_filename = aov_list[i] + "_" + std::string(filename);

            const uint xres = render->get_image_resolution().x, yres = render->get_image_resolution().y;
            const uint channels = 3;
            float pixels[xres*yres*channels];
            for(uint x = 0; x < xres; x++)
            for(uint y = 0; y < yres; y++)
            {
                Vec3f color = render->get_pixel(aov_list[i], x, y);
                pixels[(x + y*xres)*3 + 0] = color.r;
                pixels[(x + y*xres)*3 + 1] = color.g;
                pixels[(x + y*xres)*3 + 2] = color.b;
            }

            OIIO::ImageOutput::unique_ptr out = OIIO::ImageOutput::create(aov_filename);
            OIIO::ImageSpec spec (xres, yres, channels, OIIO::TypeDesc::FLOAT);
            out->open(aov_filename, spec);
            out->write_image(OIIO::TypeDesc::FLOAT, pixels);
            out->close();
        }
    }
};
