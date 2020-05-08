#include <Render/Render.h>

#include <iostream>
#include <cfloat>
#include <chrono>
#include <thread>

#include <Util/Color.h>
#include <Util/Memory.h>

#include <integrators/Integrator.h>
#include <embree/EmbreeIntersector.h>

#define NEAREST_NEIGHBOUR

Render::Render(Scene& scene, Settings& settings)
: scene(scene), camera(scene.get_camera()), settings(settings), resolution(camera->get_image_resolution())
{
    // TODO: Perform checks (eg. does the camera exist.)

    scene.pre_render();

    // TODO: this should be passed in as an argument to the Render
    intersector = new EmbreeIntersector(&scene);

    RandomNumberService random_number_service;
    random_number_service.pre_render();

    std::mt19937 seed_generator;
    pixels = (Pixel ***)malloc(resolution.x * sizeof(Pixel**));
    for(uint x = 0; x < resolution.x; x++)
    {
        pixels[x] = (Pixel **)malloc(resolution.y * sizeof(Pixel*));
        for(uint y = 0; y < resolution.y; y++)
            pixels[x][y] = new Pixel(x, y, camera->get_pixel_samples(Vec2u(x, y)), seed_generator(), random_number_service.get_random_2D());
    }
}

void Render::start_render()
{
    const auto start = std::chrono::system_clock::now();

    RandomNumberService random_number_service;

    std::vector<Vec2i> work;
    work.reserve(resolution.x * resolution.y);
    for(size_t x = 0; x < resolution.x; x++)
        for(size_t y = 0; y < resolution.y; y++)
            work.push_back(Vec2i(x,y));
    random_number_service.shuffle<Vec2i>(work);

    std::thread workers[settings.num_threads];
    for(uint i = 0; i < settings.num_threads; i++)
        workers[i] = std::thread(&Render::render_worker, this, i, work);

    for(uint i = 0; i < settings.num_threads; i++)
        workers[i].join();

    const auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Render Time: " << elapsed_seconds.count() << "\n";
}

void Render::render_worker(const uint id, const std::vector<Vec2i> &work)
{
    Resources resources;
    resources.settings = &settings;
    resources.thread_id = id;
    resources.intersector = intersector;

    bool completed = false;
    while(!completed && !kill_signal)
    {
        completed = true;
        for(size_t i = id * (work.size() / settings.num_threads); i < (id + 1) * (work.size() / settings.num_threads) && !kill_signal; i++)
        {
            const int &x = work[i][0], &y = work[i][1];
            if(pixels[x][y]->current_sample < pixels[x][y]->required_samples)
            {
                Ray ray = camera->sample_pixel(pixels[x][y]->pixel, pixels[x][y]->rng.rand());
                resources.random_number_service = RandomNumberService(pixels[x][y]->seed());

                PathData path; // CAMERA
                path.depth = 0;
                path.quality = 1.f;
                path.prev_path = nullptr;

                IntersectList intersects = intersector->intersect(ray, &path);
                pixels[x][y]->color += Integrator::shade(intersects, resources);
                pixels[x][y]->current_sample++;

                resources.memory.clear();
                completed = false;
            }
        }
    }
}

Vec3f Render::get_pixel_color(const uint& x, const uint& y) const
{
    if(pixels[x][y]->current_sample == 0)
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
                    if(pixels[xs][ys]->current_sample > 0)
                    {
                        total++;
                        color += pixels[xs][ys]->color / (float)pixels[xs][ys]->current_sample;
                    }
            rad++;
        }
        if(total > 0)
            return color / total;
#endif
        return VEC3F_ZERO;
    }

    return pixels[x][y]->color / (float)pixels[x][y]->current_sample;
}
