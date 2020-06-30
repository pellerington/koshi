#include <Render/Render.h>

#include <iostream>
#include <cfloat>
#include <chrono>
#include <thread>

#include <Util/Color.h>
#include <Util/Memory.h>
#include <Math/Helpers.h>

#include <integrator/Integrator.h>
#include <embree/EmbreeIntersector.h>

Render::Render(Scene& scene, Settings& settings)
: scene(scene), camera(scene.get_camera()), settings(settings), resolution(camera->get_image_resolution())
{
    // TODO: Perform checks (eg. does the camera exist.)

    // Generate and seed our pixels.
    RandomService random_service;
    random_service.pre_render();
    std::mt19937 seed_generator;
    pixels = (Pixel ***)malloc(resolution.x * sizeof(Pixel**)); // TODO: Move to file output?
    for(uint x = 0; x < resolution.x; x++)
    {
        pixels[x] = (Pixel **)malloc(resolution.y * sizeof(Pixel*));
        for(uint y = 0; y < resolution.y; y++)
            pixels[x][y] = new Pixel(x, y, seed_generator(), random_service.get_random_2D());
    }
}

void Render::start_render()
{
    RandomService random_service;

    Resources resources;
    resources.thread_id = 0;
    resources.settings = &settings;
    resources.scene = &scene;
    resources.random_service = &random_service;

    // TODO: pre_render on demand!
    scene.pre_render(resources);

    // TODO: this should be passed in as an argument to the Render
    intersector = new EmbreeIntersector(&scene);

    // Start Rendering
    const auto start = std::chrono::system_clock::now();

    pass_resolution = std::vector<uint>(settings.num_threads, 1);
    preview = true;

    std::thread workers[settings.num_threads];
    for(uint i = 0; i < settings.num_threads; i++)
        workers[i] = std::thread(&Render::render_worker, this, i);

    for(uint i = 0; i < settings.num_threads; i++)
        workers[i].join();

    const auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Render Time: " << elapsed_seconds.count() << "\n";
}

void Render::render_worker(const uint& id)
{
    Resources resources;
    resources.settings = &settings;
    resources.thread_id = id;
    resources.intersector = intersector;
    resources.memory = new Memory;

    bool rendered = false;

    auto render_pixel_sample = [&](const uint& x, const uint& y)
    {
        if (pixels[x][y]->samples < settings.max_samples_per_pixel 
        && (pixels[x][y]->samples < settings.min_samples_per_pixel 
        ||  pixels[x][y]->variance > 0.05f*0.05f))
        {
            Ray ray = camera->sample_pixel(Vec2u(x, y), pixels[x][y]->rng.rand());
            RandomService random_service(pixels[x][y]->seed());
            resources.random_service = &random_service;

            PathData path;
            path.depth = 0;
            path.quality = 1.f;
            path.prev_path = nullptr;

            IntersectList * intersects = intersector->intersect(ray, &path, resources);
            const Vec3f color = Integrator::shade(intersects, resources);

            pixels[x][y]->color += color;
            pixels[x][y]->samples++;

            pixels[x][y]->color_sqr += color*color;
            pixels[x][y]->variance = variance(pixels[x][y]->color, pixels[x][y]->color_sqr, pixels[x][y]->samples);

            resources.memory->clear();
            rendered = false;
        }
    };

    uint max_resolution = pow(2, ceil(log2(resolution.max())));

    while(!rendered && !kill_render)
    {
        rendered = true;
        if(id == 0) render_pixel_sample(0, 0);
        uint index = 1;
        for(uint curr_resolution = 1; curr_resolution < max_resolution; curr_resolution *= 2)
        {
            uint m = max_resolution / (curr_resolution * 2);
            for(uint yr = 0; yr < curr_resolution && yr*max_resolution/curr_resolution < resolution.y; yr++)
            {
                for(uint xr = 0; xr < curr_resolution && xr*max_resolution/curr_resolution < resolution.x; xr++)
                {
                    {
                        uint xi = 1, yi = 0;
                        for(uint i = 0; i < 3; i++)
                        {
                            index++;
                            uint x = (xr*2 + xi) * m, y = (yr*2 + yi) * m;
                            if(index % settings.num_threads == id)
                                render_pixel_sample(x, y);
                            yi = !yi ? 1 : 1;
                            xi = !xi ? 1 : 0;
                        }
                    }
                }
            }

            if(preview)
            {
                pass_resolution[id] = std::max(curr_resolution*2, pass_resolution[id]);
                uint preview_resolution = pass_resolution[0];
                for(uint i = 0; i < pass_resolution.size(); i++)
                    preview_resolution = std::min(preview_resolution, pass_resolution[i]);
                preview = (preview_resolution < max_resolution);
            }
        }
    }
}

Vec3f Render::get_pixel_color(const uint& x, const uint& y) const
{
    uint xi = x, yi = y;
    if(preview)
    {
        uint max_resolution = pow(2, ceil(log2(resolution.max())));
        uint min_resolution = max_resolution;
        for(uint i = 0; i < pass_resolution.size(); i++)
            min_resolution = std::min(min_resolution, pass_resolution[i]);
        uint m = (max_resolution / min_resolution);
        xi = m * (x / m), yi = m * (y / m);
    }
    return (!pixels[xi][yi]->samples) ? VEC3F_ZERO : pixels[xi][yi]->color / pixels[xi][yi]->samples;
}
