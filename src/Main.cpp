#include <iostream>
#include <thread>

#include <Math/Types.h>
#include <Import/SceneFile.h>
#include <Export/SFMLViewer.h>
#include <Export/OIIOViewer.h>
#include <Export/DebugObj.h>

#include <cmath>

int main(int argc, char *argv[])
{
    std::cout << "Started Render." << '\n';

    int threads = 1;
    bool sfml_window = false;
    std::string filename;
    for(int i = 1; i < argc-1; i++)
    {
        if(std::string(argv[i]) == "-threads")
        {
            threads = std::atoi(argv[i+1]);
        }
        if(std::string(argv[i]) == "-file")
        {
            filename = std::string(argv[i+1]);
        }
        if(std::string(argv[i]) == "-sfml")
        {
            sfml_window = true;
        }
    }

    // std::vector<Vec3f> points;
    // for(int i = 0; i < num_samples; i++)
    // {
    //     points.push_back(point);
    // }
    // DebugObj::Points(points);

    // Memory memory;
    // for(uint i = 0; i < 200; i++)
    // {
    //     Ray * ray = resources.memory.create<Ray>(Vec3f(0,i,0), Vec3f(0,0,i));
    //     std::cout << ray->dir << '\n';
    // }
    // memory.clear();
    // std::cout << "Cleared!" << '\n';
    // for(uint i = 0; i < 200; i++)
    // {
    //     Ray * ray = resources.memory.create<Ray>(Vec3f(0,i,0), Vec3f(0,0,i));
    //     std::cout << ray->dir << '\n';
    // }


    std::cout << "Threads: " << threads << '\n';
    std::cout << "File: " << filename << '\n';

    Scene scene = SceneFile::Import(filename, threads);
    Render render(&scene, threads);

    std::cout << "Imported Scene" << '\n';

    // TODO: Create and return threads, so we dont need to do thread -> thread
    std::thread render_thread(&Render::start_render, &render);

    std::cout << "Render Thread Started." << '\n';

    if(sfml_window)
    {
        std::thread view_thread(SFMLViewer::RenderWindow, 1024, &render);
        view_thread.join();
        render.kill_render();
    }

    render_thread.join();

    OIIOViewer::FileOut(render, "output.png");

    std::cout << "Writing Render." << '\n';

    return 0;
}
