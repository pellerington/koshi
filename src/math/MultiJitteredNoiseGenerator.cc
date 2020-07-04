#include <math/MultiJitteredNoiseGenerator.h>

#include <iostream>
#include <fstream>
#include <sstream>

template<uint N>
float * MultiJitteredNoiseGenerator<N>::GetData(const uint& sequences, const uint& sequence_size)
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
void MultiJitteredNoiseGenerator<N>::CreateData(float * data, const uint& sequences, const uint& sequence_size)
{
    std::cout << "Creating Multi-Jittered Noise " << N << "D Cache..." << '\n';

    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    std::mt19937 generator;

    for(uint offset = 0; offset < sequences * sequence_size * N; offset += sequence_size * N)
    {

        // TODO: Fill me in!

        // uint count = 0;
        // uint resolution = 1;

        // std::vector<bool> added(std::pow(resolution, N), false);
        // std::vector<float[N]> positions(std::pow(resolution, N));

        // for(uint i = 0; i < count; i++)
        // {
        //     uint position = 0;
        //     for(uint n = 0; n < N; n++)
        //         position += std::floor(data[offset + i*N + n] * resolution) * std::pow(resolution, n-1);
        //     added[position] = true;
        // }

        // for(uint i = 0; i < )

    }
}

template<uint N>
void MultiJitteredNoiseGenerator<N>::SaveData(float * data, const uint& sequences, const uint& sequence_size)
{
    std::ofstream file("RandomCacheMultiJitteredNoise" + std::to_string(N) + "D");
    if (file.is_open())
    {
        for(uint i = 0; i < sequences * sequence_size * N; i++)
            file << data[i] << "\n";
        file.close();
    }
}

template<uint N>
bool MultiJitteredNoiseGenerator<N>::LoadData(float * data, const uint& sequences, const uint& sequence_size)
{
    std::ifstream file("RandomCacheMultiJitteredNoise" + std::to_string(N) + "D");
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