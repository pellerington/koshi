#include "Render.h"

#include <iostream>
#include <cfloat>
#include <chrono>
#include <thread>

#include "../Textures/Image.h"
#include "../Util/Color.h"

#define NEAREST_NEIGHBOUR

Render::Render(Scene * scene, const uint &num_workers)
: scene(scene), num_workers(num_workers), resolution(scene->camera.get_image_resolution())
{
    scene->pre_render();

    // This should be passed in or selected using a type (when we have multiple pixels)
    integrator = std::unique_ptr<Integrator>(new PathIntegrator(scene));
    integrator->pre_render();

    pixels = std::vector<std::vector<Pixel>>(resolution.x, std::vector<Pixel>(resolution.y));
    for(uint x = 0; x < resolution.x; x++)
    {
        for(uint y = 0; y < resolution.y; y++)
        {
            pixels[x][y].pixel = Vec2u(x, y);
            RNG::Rand2d(scene->camera.get_pixel_samples(pixels[x][y].pixel), pixels[x][y].rng);
            RNG::Shuffle<Vec2f>(pixels[x][y].rng);
            pixels[x][y].required_samples = pixels[x][y].rng.size();
        }
    }
}

void Render::start_render()
{
    const auto start = std::chrono::system_clock::now();

    std::vector<Vec2i> work;
    work.reserve(resolution.x * resolution.y);
    for(size_t x = 0; x < resolution.x; x++)
        for(size_t y = 0; y < resolution.y; y++)
            work.push_back(Vec2i(x,y));
    RNG::Shuffle<Vec2i>(work);

    std::thread workers[num_workers];
    for(uint i = 0; i < num_workers; i++)
        workers[i] = std::thread(&Render::render_worker, this, i, work);

    for(uint i = 0; i < num_workers; i++)
        workers[i].join();

    const auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Render Time: " << elapsed_seconds.count() << "\n";
}

void Render::render_worker(const uint id, const std::vector<Vec2i> &work)
{
    bool completed = false;
    while(!completed)
    {
        completed = true;
        for(size_t i = id * (work.size() / num_workers); i < (id + 1) * (work.size() / num_workers); i++)
        {
            const int &x = work[i][0], &y = work[i][1];
            if(pixels[x][y].current_sample < pixels[x][y].required_samples)
            {
                Ray ray = scene->camera.sample_pixel(pixels[x][y].pixel, &pixels[x][y].rng[pixels[x][y].current_sample]);
                pixels[x][y].color += integrator->integrate(ray);
                pixels[x][y].current_sample++;
                completed = false;
            }
        }
    }
}

Vec3f Render::get_pixel_color(uint x, uint y) const
{
    if(pixels[x][y].current_sample == 0.f)
    {
#ifdef NEAREST_NEIGHBOUR
        const int xstr = x, ystr = y;
        const int radmax = std::max(resolution.x, resolution.y);
        const int xmax = (int)resolution.x, ymax = (int)resolution.y;
        uint total = 0;
        int rad = 1;
        Vec3f color;
        while(!total && rad < radmax)
        {
            for(int ys = std::max(ystr - rad, 0); ys < ystr + rad && ys < ymax; ys++)
                for(int xs = std::max(xstr - rad, 0); xs < xstr + rad && xs < xmax; (ys > ystr-rad && ys < ystr+rad-1) ? xs += rad-1 : xs++)
                    if(pixels[xs][ys].current_sample > 0.f)
                    {
                        total++;
                        color += pixels[xs][ys].color / pixels[xs][ys].current_sample;
                    }
            rad++;
        }
        if(total > 0)
            return color / total;
#endif
        return VEC3F_ZERO;
    }
    return pixels[x][y].color / pixels[x][y].current_sample;
}
