#include <koshi/material/Material.h>
#include <koshi/intersection/IntersectCallbacks.h>

struct MaterialLobeRandomWalk : MaterialLobe
{
    Vec3f density;
    float max_density;
    float inv_max_density;
    Vec3f null;
    Vec3f scatter;
    float g;
    float g_sqr;
    float g_inv;

    Intersector * intersector;
    Material * exit_material;
    PathData next_path;

    bool sample(MaterialSample& sample, Resources& resources) const;
    bool evaluate(MaterialSample& sample, Resources& resources) const { return false; }
    ScatterType get_scatter_type() const { return ScatterType::SUBSURFACE; }
    Hemisphere get_hemisphere() const { return Hemisphere::FRONT; }

private:
    static void post_intersection_callback(IntersectList * intersects, void * data, Resources& resources);
    static IntersectionCallbacks intersection_callback;

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
