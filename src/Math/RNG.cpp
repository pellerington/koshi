#include <Math/RNG.h>

std::uniform_real_distribution<float> RNG_UTIL::distribution = std::uniform_real_distribution<float>(0.f, 1.f);
std::mt19937 RNG_UTIL::random_generator = std::mt19937();

std::mt19937 BLUE_NOISE::generator = std::mt19937();
std::vector<std::vector<float>> BLUE_NOISE::maps_1D = std::vector<std::vector<float>>();
std::vector<std::vector<Vec2f>> BLUE_NOISE::maps_2D = std::vector<std::vector<Vec2f>>();
bool BLUE_NOISE::loaded_cache = BLUE_NOISE::GenerateCache();
