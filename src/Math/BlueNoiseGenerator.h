#pragma once

#include <random>
#include <iostream>
#include <fstream>
#include <Math/Types.h>

class BlueNoiseGenerator
{
public:

    static Vec2f ** GenerateMaps2D(const uint& num_points, const uint& num_maps)
    {
        Vec2f ** maps = new Vec2f*[num_maps];

        if(!LoadCache2D(maps, num_points, num_maps))
        {
            std::mt19937 generator;

            std::cout << "Generating Blue Noise 2D Cache..." << '\n';
            for(uint i = 0; i < num_maps; i++)
                maps[i] = GenerateMap2D(generator, num_points, num_maps);

            SaveCache2D(maps, num_points, num_maps);
        }

        return maps;
    }

    static Vec2f * GenerateMap2D(std::mt19937& generator, const uint& num_points, const uint& num_maps)
    {
        Vec2f * map = new Vec2f[num_points];

        std::uniform_real_distribution<float> distribution(0.f, 1.f);
        map[0] = Vec2f(distribution(generator), distribution(generator));

        for(uint i = 1; i < num_points; i++)
        {
            float max_dist = 0.f;
            Vec2f max_point;

            const uint num_candidates = i * 10 + 1;
            for(uint j = 0; j < num_candidates; j++)
            {
                const Vec2f candidate(distribution(generator), distribution(generator));

                float min_dist = 2.f;
                for(uint k = 0; k < i; k++)
                {
                    float dx = fabs(candidate.x - map[k].x);
                    float dy = fabs(candidate.y - map[k].y);
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

            map[i] = max_point;
        }

        return map;
    }

    static void SaveCache2D(Vec2f ** maps, const uint& num_points, const uint& num_maps)
    {
        std::ofstream file("bluenoise_cache_2D");
        if (file.is_open())
        {
            file << num_maps << " " << num_points << "\n";
            for(uint i = 0; i < num_maps; i++)
                for(uint j = 0; j < num_points; j++)
                    file << maps[i][j][0] << " " << maps[i][j][1] << "\n";
            file.close();
        }
    }

    static bool LoadCache2D(Vec2f ** maps, const uint& num_points, const uint& num_maps)
    {
        std::ifstream file("bluenoise_cache_2D");
        if (file.is_open())
        {
            std::string line;

            // Check head
            std::getline(file, line);
            std::istringstream ss(line);
            uint n, count = 0;
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

            // Get 2D contents
            uint nm = 0, np = 0;
            while(nm < num_maps && std::getline(file, line))
            {
                if(np == 0) maps[nm] = new Vec2f[num_points];

                std::istringstream ss(line);
                float f;
                uint count = 0;
                while( ss >> f )
                {
                    maps[nm][np][count] = f;
                    count++;
                }

                if(count != 2) return false;

                if(++np == num_points)
                {
                    nm++;
                    np = 0;
                }
            }

            file.close();
            return true;
        }

        return false;
    }

};

// static std::vector<float> Generate1DMap()
// {
//     std::vector<float> points;
//     points.reserve(num_points);

//     points.push_back(RNG_UTIL::distribution(generator));

//     for(uint i = 1; i < num_points; i++)
//     {
//         float max_dist = 0.f;
//         float max_point = 0.f;

//         const uint num_candidates = i * 10 + 1;
//         for(uint j = 0; j < num_candidates; j++)
//         {
//             const float candidate = RNG_UTIL::distribution(generator);

//             float min_dist = 2.f;
//             for(uint k = 0; k < points.size(); k++)
//             {
//                 float dx = fabs(candidate - points[k]);
//                 if (dx > 0.5f) dx = 1.0f - dx;
//                 min_dist = std::min(min_dist, dx);
//             }

//             if(min_dist > max_dist)
//             {
//                 max_dist = min_dist;
//                 max_point = candidate;
//             }
//         }

//         points.push_back(max_point);
//     }

//     return points;
// }