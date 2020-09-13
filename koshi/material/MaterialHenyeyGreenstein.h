#include <koshi/material/Material.h>

struct MaterialLobeHenyeyGreenstein : public MaterialLobe
{
    bool sample(MaterialSample& sample, Resources& resources) const;
    bool evaluate(MaterialSample& sample, Resources& resources) const;
    
    ScatterType get_scatter_type() const { return ScatterType::VOLUME; }
    Hemisphere get_hemisphere() const { return Hemisphere::SPHERE; }

    float g, g_sqr, g_inv;
};

class MaterialHenyeyGreenstein : public Material
{
public:
    MaterialHenyeyGreenstein(const Texture * anistropy_texture) 
    : Material(), anistropy_texture(anistropy_texture)
    {
    }

    MaterialLobes instance(const Surface * surface, const Intersect * intersect, Resources& resources);

private:
    const Texture * anistropy_texture;
};
