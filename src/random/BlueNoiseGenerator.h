#pragma once

#include <random>

// Uses a best candidate approach.

template<uint N>
class BlueNoiseGenerator
{
public:
    static float * GetData(const uint& sequences, const uint& sequence_size);
    static void CreateData(float * data, const uint& sequences, const uint& sequence_size);
    static void SaveData(float * data, const uint& sequences, const uint& sequence_size);
    static bool LoadData(float * data, const uint& sequences, const uint& sequence_size);
};

template class BlueNoiseGenerator<1>;
template class BlueNoiseGenerator<2>;
