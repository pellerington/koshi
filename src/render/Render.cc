#include <render/Render.h>

#include <iostream>
#include <cfloat>

#include <Util/Color.h>
#include <Util/Memory.h>
#include <math/Helpers.h>

#include <integrator/Integrator.h>
#include <embree/EmbreeIntersector.h>

Render::Render(Scene * scene, Settings * settings)
: scene(scene), settings(settings), resolution(scene->get_camera()->get_image_resolution())
{
    RandomService random_service;
    random_service.pre_render();

    aovs["color"].mode = AOV::AVERAGE;
    aovs["color"].buffer = new Buffer(resolution.x * resolution.y, 3);
    aovs["color_sqr"].mode = AOV::AVERAGE;
    aovs["color_sqr"].buffer = new Buffer(resolution.x * resolution.y, 3);
    aovs["samples"].mode = AOV::SUM;
    aovs["samples"].buffer = new Buffer(resolution.x * resolution.y, 1);

    // TODO: Move this type of aov into a user callback or function.
    aovs["normals"].mode = AOV::AVERAGE;
    aovs["normals"].buffer = new Buffer(resolution.x * resolution.y, 3);

    // TODO: Make this an aov buffer so we can do checkpointing!
    rng = new Random<2>[resolution.x * resolution.y];

    for(uint i = 0; i < resolution.x * resolution.y; i++)
    {
        aovs["color"].buffer->set(i, VEC3F_ZERO);
        aovs["color_sqr"].buffer->set(i, VEC3F_ZERO);
        aovs["samples"].buffer->set(i, VEC3F_ZERO);
        aovs["normals"].buffer->set(i, VEC3F_ZERO);
        rng[i] = random_service.get_random<2>();
    }
}

void Render::start()
{
    RandomService random_service;
    Resources resources;
    resources.thread_id = 0;
    resources.settings = settings;
    resources.scene = scene;
    resources.random_service = &random_service;

    // TODO: pre-render on demand!
    scene->pre_render(resources);

    // TODO: this should be passed in as an argument to the Render? Or included in the scene.
    ObjectGroup render_group;
    for(auto object = scene->begin(); object != scene->end(); ++object)
    {
        Geometry * geometry = dynamic_cast<Geometry*>(object->second);
        if(geometry) render_group.push(geometry);
    }
    intersector = new EmbreeIntersector(&render_group);

    start_time = std::chrono::system_clock::now();

    thread_state.resize(settings->num_threads, false);
    threads.resize(settings->num_threads);
    for(uint i = 0; i < threads.size(); i++)
        threads[i] = std::thread(&Render::spawn_thread, this, i);
}

void Render::resume()
{
    for(uint i = 0; i < threads.size(); i++)
        threads[i] = std::thread(&Render::spawn_thread, this, i);
}

void Render::pause()
{
    pause_signal = true;
    for(uint i = 0; i < threads.size(); i++)
        threads[i].join();
}

void Render::wait()
{
    for(uint i = 0; i < settings->num_threads; i++)
        threads[i].join();
}

bool Render::sample_pixel(const uint& x, const uint& y, Resources& resources)
{
    const uint index = y*resolution.x + x;
    const float samples = aovs.at("samples").buffer->get(index)[0];
    const Vec3f color = aovs.at("color").buffer->get(index);
    const Vec3f color_sqr = aovs.at("color_sqr").buffer->get(index);

    if (samples < settings->max_samples_per_pixel && (samples < settings->min_samples_per_pixel || variance(color, color_sqr, samples) > 0.05f*0.05f))
    {
        Ray ray = scene->get_camera()->sample_pixel(x, y, rng[index].rand());
        RandomService random_service(index + samples*resolution.x*resolution.y);
        resources.random_service = &random_service;

        PathData path;
        path.depth = 0;
        path.quality = 1.f;
        path.prev_path = nullptr;
        IntersectList * intersects = intersector->intersect(ray, &path, resources);
        const Vec3f output = Integrator::shade(intersects, resources);

        aovs.at("color").buffer->add(index, output);
        aovs.at("color_sqr").buffer->add(index, output*output);
        aovs.at("samples").buffer->add(index, VEC3F_ONES);

        // TODO: Move me somewhere else later.
        if(intersects->hit())
        {
            const Surface * surface = dynamic_cast<Surface*>(intersects->get_front()->geometry_data);
            if(surface) aovs.at("normals").buffer->add(index, surface->normal);
        }

        resources.memory->clear();

        return true;
    }
    return false;
}

void Render::spawn_thread(const uint& id)
{
    Resources resources;
    resources.settings = settings;
    resources.scene = scene;
    resources.thread_id = id;
    resources.intersector = intersector;
    resources.memory = new Memory;

    uint max_resolution = pow(2, ceil(log2(resolution.max())));
    bool thread_completed = false;
    while(!thread_completed && !pause_signal)
    {
        thread_completed = true;
        if(id == 0) thread_completed = !sample_pixel(0, 0, resources) && thread_completed;
        uint index = 1;
        for(uint curr_resolution = 1; curr_resolution < max_resolution; curr_resolution *= 2)
        {
            uint n = max_resolution / curr_resolution;
            uint m = max_resolution / (curr_resolution * 2);
            for(uint i = 0; i < 3; i++)
            {
                for(uint y = !(i & 1u) * m; y < resolution.y; y += n)
                {
                    for(uint x = !(i & 2) * m; x < resolution.x; x += n)
                    {
                        if((index++ % settings->num_threads) == id)
                            thread_completed = !sample_pixel(x, y, resources) && thread_completed;
                    }
                }
            }
        }
    }

    thread_state[id] = thread_completed;
    end_time = std::chrono::system_clock::now();

    for(uint i = 0; i < thread_state.size(); i++)
        if(!thread_state[i])
            return;

    // TODO: Access this by a function instead of printing it out.
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::cout << "Render Time: " << elapsed_seconds.count() << "\n";

}

const std::vector<std::string> Render::get_aov_list() const
{
    std::vector<std::string> aov_list;
    for(auto aov = aovs.begin(); aov != aovs.end(); ++aov)
        aov_list.push_back(aov->first);
    return aov_list;
}

Vec3f Render::get_pixel(const std::string& aov_name, const uint& x, const uint& y) const
{
    float samples = aovs.at("samples").buffer->get(y*resolution.x + x)[0];

    if(!samples)
    {
        uint xi = x, yi = y; 
        while(xi > 0 || yi >0)
        {
            xi /= 2; yi /= 2; xi *= 2; yi *= 2;
            float samples = aovs.at("samples").buffer->get(yi*resolution.x + xi)[0];
            if(samples > 0)
            {
                const Vec3f value = aovs.at(aov_name).buffer->get(yi*resolution.x + xi);
                return value / ((aovs.at(aov_name).mode == AOV::AVERAGE) ? samples : 1.f);
            }
        }
    }

    const Vec3f value = aovs.at(aov_name).buffer->get(y*resolution.x + x);
    return (!samples) ? VEC3F_ZERO : value / ((aovs.at(aov_name).mode == AOV::AVERAGE) ? samples : 1.f);
}