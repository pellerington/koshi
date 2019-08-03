#include <iostream>
#include <thread>

#include "Math/Types.h"
#include "Import/SceneFile.h"
#include "Export/SFMLViewer.h"
#include "Export/DebugObj.h"

#include <cmath>

int main(int argc, char *argv[])
{
    int threads = 1;
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
    }

    // std::vector<Vec3f> points;
    // int num_samples = 25000;
    // std::vector<Vec2f> rnd;
    // RNG::Rand2d(num_samples, rnd);
    // for(int i = 0; i < num_samples; i++)
    // {
    //     Vec3f point;
    //     points.push_back(point);
    // }
    // DebugObj::Points(points);

    std::cout << "File: " << filename << '\n';

    Scene scene = SceneFile::Import(filename);
    Render render(&scene, threads);

    std::cout << "Imported Scene" << '\n';

    std::thread render_thread(&Render::start_render, &render);

    std::cout << "Render Thread Started." << '\n';

    std::thread view_thread(SFMLViewer::RenderWindow, 1024, &render);
    view_thread.join();

    std::cout << "End Render." << '\n';

    return 0;
}
