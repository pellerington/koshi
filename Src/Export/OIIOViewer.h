#pragma once

class OIIOViewer
{
public:
    static void FileOut(const Render &render, const char * filename)
    {
        const uint xres = render.get_image_resolution().x, yres = render.get_image_resolution().y;
        const uint channels = 3;
        float pixels[xres*yres*channels];
        for(uint x = 0; x < xres; x++)
        for(uint y = 0; y < yres; y++)
        {
            Vec3f color = render.get_pixel_color(x, y);
            pixels[(x + y*xres)*3 + 0] = color.r;
            pixels[(x + y*xres)*3 + 1] = color.g;
            pixels[(x + y*xres)*3 + 2] = color.b;
        }

        OIIO::ImageOutput * out = OIIO::ImageOutput::create(filename);
        OIIO::ImageSpec spec (xres, yres, channels, OIIO::TypeDesc::FLOAT);
        out->open(filename, spec);
        out->write_image(OIIO::TypeDesc::FLOAT, pixels);
        out->close();
    }
};
