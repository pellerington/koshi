#include <koshi/material/Material.h>

struct MaterialLobeRandomWalk : MaterialLobe
{
    Vec3f density;
    Vec3f scatter;
    float g;
    float g_sqr;
    float g_inv;
    Intersector * intersector;
    Material * exit_material;

    bool sample(MaterialSample& sample, Resources& resources) const { return false; }
    Vec3f weight(const Vec3f& wo, Resources& resources) const { return VEC3F_ZERO; }
    float pdf(const Vec3f& wo, Resources& resources) const { return 0.f; }
    ScatterType get_scatter_type() const { return ScatterType::DIFFUSE; }
    Hemisphere get_hemisphere() const { return Hemisphere::FRONT; }
};

class MaterialRandomWalk: public Material
{
public:
    MaterialRandomWalk(const Texture * color_texture, const Texture * density_texture, const float& anistropy, 
                       const Texture * normal_texture, const Texture * opacity_texture);
    MaterialLobes instance(const Surface * surface, const Intersect * intersect, Resources& resources);
private:
    const Texture * color_texture;
    const Texture * density_texture;
    const float anistropy;
    Material * exit_material;
};
