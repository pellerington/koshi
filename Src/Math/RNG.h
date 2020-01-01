#pragma once

#include <random>
#include <deque>
#include <fstream>
#include "../Math/Types.h"

class RNG_UTIL
{
public:
    template <class T>
    static void Shuffle(std::vector<T> &samples) { std::shuffle(samples.begin(), samples.end(), random_generator); }
    template <class T>
    static void Shuffle(std::deque<T> &samples) { std::shuffle(samples.begin(), samples.end(), random_generator); }

    static std::uniform_real_distribution<float> distribution;

private:
    static std::mt19937 random_generator;
};

class RANDOM
{
public:
    RANDOM(const uint seed) : generator(seed)
    {
    }

    inline float Rand()
    {
        return RNG_UTIL::distribution(generator);
    }

    inline Vec2f Rand2D()
    {
        return Vec2f(RNG_UTIL::distribution(generator), RNG_UTIL::distribution(generator));
    }

    inline void Reset() {}
    inline void Reset2D() {}

private:
    std::mt19937 generator;
};

class RSEQ
{
public:
    RSEQ(const uint seed)
    : n_1D(seed), n_2D(seed)
    {
    }

    inline float Rand()
    {
        n_1D++;
        return fmod(shift + a1_1D * n_1D, 1);
    }

    inline Vec2f Rand2D()
    {
        n_2D++;
        return Vec2f(fmod(shift + a1_2D * n_2D, 1), fmod(shift + a2_2D * n_2D, 1));
    }

    inline void Reset() {}
    inline void Reset2D() {}

private:
    double n_1D;
    static constexpr double g_1D = 1.61803398874989484820458683436563;
    static constexpr double a1_1D = 1.0 / g_1D;

    double n_2D;
    static constexpr double g_2D = 1.32471795724474602596090885447809;
    static constexpr double a1_2D = 1.0 / g_2D;
    static constexpr double a2_2D = 1.0 / (g_2D*g_2D);

    static constexpr double shift = 0.5;
};

class BLUE_NOISE
{
public:

    BLUE_NOISE(const uint seed)
    : n_1D(0), map_1D(seed%num_maps), n_2D(0), map_2D(map_1D)
    {
    }

    inline float Rand()
    {
        if(n_1D >= num_points)
        {
            n_1D = 0;
            map_1D = (map_1D + 1) % num_maps;
        }
        return maps_1D[map_1D][n_1D++];
    }

    inline Vec2f Rand2D()
    {
        if(n_2D >= num_points)
        {
            n_2D = 0;
            map_2D = (map_2D + 1) % num_maps;
        }
        return maps_2D[map_2D][n_2D++];
    }

    inline void Reset()
    {
        n_1D = 0;
        map_1D = (map_1D + 1) % num_maps;
    }

    inline void Reset2D()
    {
        n_2D = 0;
        map_2D = (map_2D + 1) % num_maps;
    }

    static bool GenerateCache()
    {
        maps_1D.reserve(num_maps);
        maps_2D.reserve(num_maps);

        if(!LoadCache())
        {
            std::cout << "Generating RNG Cache..." << '\n';
            for(uint i = 0; i < num_maps; i++)
                maps_1D.push_back(Generate1DMap());

            for(uint i = 0; i < num_maps; i++)
                maps_2D.push_back(Generate2DMap());

            SaveCache();

            return false;
        }
        return true;
    }
    static bool loaded_cache;

private:

    static std::vector<float> Generate1DMap()
    {
        std::vector<float> points;
        points.reserve(num_points);

        points.push_back(RNG_UTIL::distribution(generator));

        for(uint i = 1; i < num_points; i++)
        {
            float max_dist = 0.f;
            float max_point = 0.f;

            const uint num_candidates = i * 10 + 1;
            for(uint j = 0; j < num_candidates; j++)
            {
                const float candidate = RNG_UTIL::distribution(generator);

                float min_dist = 2.f;
                for(uint k = 0; k < points.size(); k++)
                {
                    float dx = fabs(candidate - points[k]);
                    if (dx > 0.5f) dx = 1.0f - dx;
                    min_dist = std::min(min_dist, dx);
                }

                if(min_dist > max_dist)
                {
                    max_dist = min_dist;
                    max_point = candidate;
                }
            }

            points.push_back(max_point);
        }

        return points;
    }

    static std::vector<Vec2f> Generate2DMap()
    {
        std::vector<Vec2f> points;
        points.reserve(num_points);

        points.emplace_back(RNG_UTIL::distribution(generator), RNG_UTIL::distribution(generator));

        for(uint i = 1; i < num_points; i++)
        {
            float max_dist = 0.f;
            Vec2f max_point;

            const uint num_candidates = i * 10 + 1;
            for(uint j = 0; j < num_candidates; j++)
            {
                const Vec2f candidate(RNG_UTIL::distribution(generator), RNG_UTIL::distribution(generator));

                float min_dist = 2.f;
                for(uint k = 0; k < points.size(); k++)
                {
                    float dx = fabs(candidate.x - points[k].x);
                    float dy = fabs(candidate.y - points[k].y);
                    if (dx > 0.5f) dx = 1.0f - dx;
                    if (dy > 0.5f) dy = 1.0f - dy;
                    min_dist = std::min(min_dist, sqrtf(dx*dx + dy*dy));
                }

                if(min_dist > max_dist)
                {
                    max_dist = min_dist;
                    max_point = candidate;
                }
            }

            points.push_back(max_point);
        }

        return points;
    }

    static void SaveCache()
    {
        std::ofstream cache_file("blue_noise_cache");
        if (cache_file.is_open())
        {
            cache_file << num_maps << " " << num_points << "\n";
            for(uint i = 0; i < num_maps; i++)
                for(uint j = 0; j < num_points; j++)
                    cache_file << maps_1D[i][j] << "\n";
            for(uint i = 0; i < num_maps; i++)
                for(uint j = 0; j < num_points; j++)
                    cache_file << maps_2D[i][j][0] << " " << maps_2D[i][j][1] << "\n";
            cache_file.close();
        }
    }

    static bool LoadCache()
    {
        std::ifstream cache_file("blue_noise_cache");
        if (cache_file.is_open())
        {
            std::string line;

            // Check head
            std::getline(cache_file, line);
            std::istringstream ss(line);
            int n, count = 0;
            while( ss >> n )
            {
                if(count == 0 && n != num_maps)
                    return false;
                if(count == 1 && n != num_points)
                    return false;
                count++;
            }
            if(count != 2)
                return false;

            // Get 1D contents
            uint nm = 0, np = 0;
            while(nm < num_maps && std::getline(cache_file, line))
            {
                if(np == 0)
                    maps_1D.push_back(std::vector<float>(num_points));

                std::istringstream ss(line);
                float f;
                uint count = 0;
                while( ss >> f )
                {
                    maps_1D[nm][np] = f;
                    count++;
                }
                if(count != 1)
                    return false;

                np++;
                if(np == num_points)
                {
                    nm++;
                    np = 0;
                }
            }

            // Get 2D contents
            nm = 0, np = 0;
            while(nm < num_maps && std::getline(cache_file, line))
            {
                if(np == 0)
                    maps_2D.push_back(std::vector<Vec2f>(num_points));

                std::istringstream ss(line);
                float f;
                uint count = 0;
                while( ss >> f )
                {
                    maps_2D[nm][np][count] = f;
                    count++;
                }
                if(count != 2)
                    return false;

                np++;
                if(np == num_points)
                {
                    nm++;
                    np = 0;
                }
            }

            cache_file.close();
            return true;
        }

        return false;
    }

    static constexpr uint num_maps = 2048;
    static constexpr uint num_points = 128;

    static std::mt19937 generator;
    static std::vector<std::vector<float>> maps_1D;
    static std::vector<std::vector<Vec2f>> maps_2D;

    uint n_1D;
    uint map_1D;

    uint n_2D;
    uint map_2D;
};

// typedef RANDOM RNG;
// typedef RSEQ RNG;
typedef BLUE_NOISE RNG;
