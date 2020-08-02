#include <koshi/camera/PixelSampler.h>

const std::vector<float> GaussianFilterSampler::cdf = {
    0.0510946, 0.102090, 0.152787, 0.202991, 0.252513, 0.301173, 0.348799, 0.395230,
    0.4403210, 0.483939, 0.525968, 0.566309, 0.604877, 0.641607, 0.676451, 0.709376,
    0.7403660, 0.769422, 0.796558, 0.821802, 0.845195, 0.866788, 0.886641, 0.904824,
    0.9214120, 0.936486, 0.950131, 0.962434, 0.973484, 0.983370, 0.992180, 1.000000
};

Vec2f GaussianFilterSampler::sample(const float rng[2])
{
    const uint i = std::lower_bound(cdf.begin(), cdf.end(), rng[0]) - cdf.begin();
    const float cdf_min = (i > 0) ? cdf[i-1] : 0.f;
    const float r = (i + (rng[0] - cdf_min) / (cdf[i] - cdf_min)) / cdf.size();
    const float theta = TWO_PI * rng[1];
    return Vec2f(r*cosf(theta) + 0.5f, r*sinf(theta) + 0.5f);
}