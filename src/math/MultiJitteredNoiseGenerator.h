#pragma once

#include <random>
#include <math/Types.h>

// Based on Progressive Multi-Jittered Sample Sequences
// https://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/paper.pdf

template<uint N>
class MultiJitteredNoiseGenerator
{
public:
    static float * GetData(const uint& sequences, const uint& sequence_size);
    static void CreateData(float * data, const uint& sequences, const uint& sequence_size);
    static void SaveData(float * data, const uint& sequences, const uint& sequence_size);
    static bool LoadData(float * data, const uint& sequences, const uint& sequence_size);
};

template class MultiJitteredNoiseGenerator<1>;
template class MultiJitteredNoiseGenerator<2>;