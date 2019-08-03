#include "RNG.h"

std::uniform_real_distribution<float> RNG::distribution = std::uniform_real_distribution<float>(0,1);

std::random_device rd;
std::mt19937 RNG::generator = std::mt19937(rd());
