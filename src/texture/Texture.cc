#include <texture/Texture.h>

template<>
float Texture::evaluate<float>(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) const { 
    return evaluate(u,v,w,intersect,resources)[0]; 
}

template<>
Vec2f Texture::evaluate<Vec2f>(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) const { 
    Vec3f out = evaluate(u,v,w,intersect,resources);
    return Vec2f(out[0], out[1]);
}

template<>
Vec3f Texture::evaluate<Vec3f>(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) const { 
    return evaluate(u,v,w,intersect,resources); 
}