#pragma once

#include <vector>
#include <iostream>
#include <chrono>
#include <thread>
#include <koshi/dependency/robin_hood.h>

#include <koshi/math/Types.h>
#include <koshi/base/Scene.h>
#include <koshi/intersection/Intersector.h>
#include <koshi/base/Settings.h>
#include <koshi/random/Random.h>
#include <koshi/camera/Camera.h>

#include <koshi/render/Buffer.h>

class Render
{
public:
    Render(Scene * scene, Settings * settings);

    void start();
    void pause();
    void resume();
    void wait();
    // void reset();

    inline const Vec2u& get_image_resolution() const { return resolution; }

    const std::vector<std::string> get_aov_list() const;
    // TODO: This should get value from any buffer/aov.
    Vec3f get_pixel(const std::string& aov_name, const uint& x, const uint& y) const;

private:
    Scene * scene;
    Settings * settings;
    Intersector * intersector;
    const Vec2u resolution;

    // TODO: Need to free buffers at the end.
    struct AOV 
    {
        enum AOVMode { AVERAGE, SUM };
        AOVMode mode;
        Buffer * buffer;
    };
    robin_hood::unordered_map<std::string, AOV> aovs;
    Random<2> * rng;

    std::vector<std::thread> threads;
    std::vector<bool> thread_state;
    std::chrono::time_point<std::chrono::system_clock> start_time;
    std::chrono::time_point<std::chrono::system_clock> end_time;
    bool pause_signal = false;

    bool sample_pixel(const uint& x, const uint& y, Resources& resources);
    void spawn_thread(const uint& id);
};
