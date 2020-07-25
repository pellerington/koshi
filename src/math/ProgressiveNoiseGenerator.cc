#include <math/ProgressiveNoiseGenerator.h>

#include <iostream>
#include <fstream>
#include <sstream>

template<uint N>
float * ProgressiveNoiseGenerator<N>::GetData(const uint& sequences, const uint& sequence_size)
{
    float * data = new float[sequences * sequence_size * N];
    if(!LoadData(data, sequences, sequence_size))
    {
        CreateData(data, sequences, sequence_size);
        SaveData(data, sequences, sequence_size);
    }
    return data;
}

template<uint N>
void ProgressiveNoiseGenerator<N>::CreateData(float * data, const uint& sequences, const uint& sequence_size)
{
    std::cout << "Creating Progressive Noise " << N << "D Cache..." << '\n';

    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    std::mt19937 generator;

    for(uint offset = 0; offset < sequences * sequence_size * N; offset += sequence_size * N)
    {
        // TODO: Also take into account per axis multi-jittering!
        // TODO: Cleanup code.
        // TOOD: Check the progression is good on python?
        // TODO: Make this cleaner using simd and only set positions for N ?

        // Sample randomly,
        // Empty square furthest from any (including wrap around).
        // Repeat until all squares are filled.
        // Double resolution.

        uint count = 0;
        int resolution = 1;
        float delta = 1.f / resolution;
        struct item {
            bool added = false;
            int index[N] = { 0 }; 
        };
        std::vector<item> samples;

        while(count < sequence_size)
        {
            samples = std::vector<item>(std::pow(resolution, N));
            for(uint i = 0; i < samples.size(); i++)
                for(uint n = 0; n < N; n++)
                    samples[i].index[n] = (uint)(i / std::pow(resolution, n)) % resolution;

            // Fill existing points.
            for(uint i = 0; i < count; i++)
            {
                uint pos = 0;
                for(uint n = 0; n < N; n++)
                {
                    pos += std::pow(resolution, n) * std::floor(data[offset + i*N + n] * resolution);
                }
                samples[pos].added = true;
            }

            uint max_count = std::pow(resolution, N);
            for(;count < max_count && count < sequence_size; count++)
            {
                uint max_i = 0;
                float max_d = 0.f;

                for(uint i = 0; i < samples.size(); i++)
                {
                    if(!samples[i].added)
                    {
                        // Closest added point.
                        float min_d = FLT_MAX;
                        for(uint j = 0; j < samples.size(); j++)
                            if(samples[j].added)
                            {
                                float dist = 0.f;
                                for(uint n = 0; n < N; n++)
                                {
                                    int d = abs(samples[i].index[n] - samples[j].index[n]);
                                    d = std::min(d, abs(samples[i].index[n] - samples[j].index[n] + resolution));
                                    d = std::min(d, abs(samples[i].index[n] - samples[j].index[n] - resolution));
                                    dist += d * d;
                                }
                                dist = sqrtf(dist);
                                min_d = std::min(dist, min_d);
                            }
                        
                        if(min_d > max_d)
                        {
                            max_i = i;
                            max_d = min_d;
                        }
                    }
                }

                // Fill it in.
                samples[max_i].added = true;
                for(uint n = 0; n < N; n++)
                    data[offset + count*N + n] = distribution(generator) * delta + ((float)samples[max_i].index[n] / resolution);
            }

            resolution *= 2;
            delta = 1.f / resolution;
        }
    }
}

template<uint N>
void ProgressiveNoiseGenerator<N>::SaveData(float * data, const uint& sequences, const uint& sequence_size)
{
    std::ofstream file("RandomCacheProgressiveNoise" + std::to_string(N) + "D");
    if (file.is_open())
    {
        for(uint i = 0; i < sequences * sequence_size * N; i++)
            file << data[i] << "\n";
        file.close();
    }
}

template<uint N>
bool ProgressiveNoiseGenerator<N>::LoadData(float * data, const uint& sequences, const uint& sequence_size)
{
    std::ifstream file("RandomCacheProgressiveNoise" + std::to_string(N) + "D");
    if (file.is_open())
    {
        std::string line;
        uint i = 0;
        while(std::getline(file, line))
        {
            std::istringstream ss(line);
            ss >> data[i];
            i++;
        }
        file.close();
        return true;
    }

    return false;
}