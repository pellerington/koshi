#include <math/BlueNoiseGenerator.h>

#include <iostream>
#include <fstream>
#include <sstream>

template<uint N>
float * BlueNoiseGenerator<N>::GetData(const uint& sequences, const uint& sequence_size)
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
void BlueNoiseGenerator<N>::CreateData(float * data, const uint& sequences, const uint& sequence_size)
{
    std::cout << "Creating Blue Noise " << N << "D Cache..." << '\n';

    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    std::mt19937 generator;

    for(uint offset = 0; offset < sequences * sequence_size * N; offset += sequence_size * N)
    {
        for(uint n = 0; n < N; n++)
            data[offset+n] = distribution(generator);

        for(uint i = 1; i < sequence_size; i++)
        {
            float max_distance = 0.f;
            float max_candidate[N] = {0.f};

            const uint num_candidates = i * 10 + 1;
            for(uint j = 0; j < num_candidates; j++)
            {
                float candidate[N] = {0.f};
                for(uint n = 0; n < N; n++)
                    candidate[n] = distribution(generator);

                float min_distance = 2.f;
                for(uint k = 0; k < i; k++)
                {
                    float distance = 0.f;
                    for(uint n = 0; n < N; n++)
                    {
                        float d = fabs(candidate[n] - data[offset + k*N + n]);
                        if (d > 0.5f) d = 1.0f - d;
                        distance += d * d;
                    }
                    distance = sqrtf(distance);
                    min_distance = std::min(min_distance, distance);
                }

                if(min_distance > max_distance)
                {
                    max_distance = min_distance;
                    for(uint n = 0; n < N; n++)
                        max_candidate[n] = candidate[n];
                }
            }

            for(uint n = 0; n < N; n++)
                data[offset + i*N + n] = max_candidate[n];
        }
    }
}

template<uint N>
void BlueNoiseGenerator<N>::SaveData(float * data, const uint& sequences, const uint& sequence_size)
{
    std::ofstream file("RandomCacheBlueNoise" + std::to_string(N) + "D");
    if (file.is_open())
    {
        for(uint i = 0; i < sequences * sequence_size * N; i++)
            file << data[i] << "\n";
        file.close();
    }
}

template<uint N>
bool BlueNoiseGenerator<N>::LoadData(float * data, const uint& sequences, const uint& sequence_size)
{
    std::ifstream file("RandomCacheBlueNoise" + std::to_string(N) + "D");
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