#include "Render.h"

#include <iostream>
#include <cfloat>
#include <chrono>
#include <thread>

#include "../Textures/Image.h"

#include "../Util/Color.h"

Render::Render(Scene * scene, uint num_workers)
: scene(scene), num_workers(num_workers)
{
    Vec2i resolution = scene->camera.get_image_resolution();
    pixels = std::vector<std::vector<PixelCtx>>(resolution.x(), std::vector<PixelCtx>(resolution.y()));

    // This should be passed in or selected using a type (when we have multiple pixels)
    master_integrator = std::unique_ptr<Integrator>(new PathIntegrator(scene));
    master_integrator->pre_render();

    for(int x = 0; x < resolution.x(); x++)
    {
        for(int y = 0; y < resolution.y(); y++)
        {
            pixels[x][y].pixel = Vec2i(x, y);
            pixels[x][y].required_samples = scene->camera.get_pixel_samples(pixels[x][y].pixel);
            RNG::StratifiedRand(pixels[x][y].required_samples, pixels[x][y].rng);
            RNG::Shuffle<Vec2f>(pixels[x][y].rng);
        }
    }
}

void Render::start_render()
{
    auto start = std::chrono::system_clock::now();

    std::vector<Vec2i> work;
    work.reserve(pixels.size() * pixels[0].size());
    for(size_t x = 0; x < pixels.size(); x++)
        for(size_t y = 0; y < pixels[x].size(); y++)
            work.push_back(Vec2i(x,y));
    RNG::Shuffle<Vec2i>(work);

    std::thread workers[num_workers];
    for(uint i = 0; i < num_workers; i++)
        workers[i] = std::thread(&Render::render_worker, this, i, work);

    for(uint i = 0; i < num_workers; i++)
        workers[i].join();

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Render Time: " << elapsed_seconds.count() << "\n";
}

void Render::render_worker(const uint id, const std::vector<Vec2i> &work)
{
    std::shared_ptr<Integrator> integrator;

    bool completed = false;
    while(!completed)
    {
        completed = true;
        for(size_t i = id * (work.size() / num_workers); i < (id + 1) * (work.size() / num_workers); i++)
        {
            const int &x = work[i][0], &y = work[i][1];
            if(pixels[x][y].current_sample < pixels[x][y].required_samples)
            {
                Ray ray;
                scene->camera.sample_pixel(pixels[x][y].pixel, ray, &pixels[x][y].rng[pixels[x][y].current_sample]);
                integrator = master_integrator->create(ray);
                integrator->integrate(integrator->get_required_samples());

                pixels[x][y].color += integrator->get_color();
                pixels[x][y].current_sample++;

                completed = false;
            }
        }
    }
}

Vec3f Render::get_pixel(uint x, uint y)
{
    if(pixels[x][y].current_sample == 0.f)
        return Vec3f::Zero();
    return pixels[x][y].color / pixels[x][y].current_sample;
}
