#include <iostream>
#include <thread>

#include <math/Types.h>
#include <import/SceneFile.h>
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
    for(int i = 0; i < argc; i++)
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

    std::cout << "Threads: " << threads << '\n';
    std::cout << "File: " << filename << '\n';

    Settings settings;
    settings.num_threads = threads;
    Scene scene;

    SceneFile scene_file;    
    scene_file.Import(filename, scene, settings);

    for(auto it = scene.begin(); it != scene.end(); ++it)
        std::cout << it->first << " " << it->second << " " << (it->second != nullptr) << "\n";

    std::cout << "Imported Scene" << '\n';

    Render render(&scene, &settings);
    render.start();

    std::cout << "Render Started." << '\n';

    if(sfml_window)
    {
        std::thread view_thread(SFMLViewer::RenderWindow, 1024, &render);
        view_thread.join();
        render.pause();
    }
    else
    {
        render.wait();
    }

    OIIOViewer::FileOut(&render, "output.png");

    std::cout << "Writing Render." << '\n';

    return 0;
}
