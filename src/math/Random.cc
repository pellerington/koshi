#include <math/Random.h>

template<>
float * Random<1>::data = nullptr;

template<>
float * Random<2>::data = nullptr;

template<uint N>
float * RandomGenerator(const uint& seqences, const uint& sequence_size)
{
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    std::mt19937 generator;

    float * data = new float[SEQUENCES * SEQUENCE_SIZE * N];
    for(uint i = 0; i < SEQUENCES * SEQUENCE_SIZE * N; i++)
        data[i] = distribution(generator);

    return data;
}

void RandomService::pre_render()
{
    // Random<1>::data = RandomGenerator<1>(SEQUENCES, SEQUENCE_SIZE);
    // Random<2>::data = RandomGenerator<2>(SEQUENCES, SEQUENCE_SIZE);

    Random<1>::data = BlueNoiseGenerator<1>::GetData(SEQUENCES, SEQUENCE_SIZE);
    Random<2>::data = BlueNoiseGenerator<2>::GetData(SEQUENCES, SEQUENCE_SIZE);
}