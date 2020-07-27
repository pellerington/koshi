#pragma once

#include <random>
#include <koshi/math/Types.h>

// Based on Loosely on Progressive Sample Sequences
// https://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/paper.pdf

template<uint N>
class ProgressiveNoiseGenerator
{
public:
    static float * GetData(const uint& sequences, const uint& sequence_size);
    static void CreateData(float * data, const uint& sequences, const uint& sequence_size);
    static void SaveData(float * data, const uint& sequences, const uint& sequence_size);
    static bool LoadData(float * data, const uint& sequences, const uint& sequence_size);
};

template class ProgressiveNoiseGenerator<1>;
template class ProgressiveNoiseGenerator<2>;