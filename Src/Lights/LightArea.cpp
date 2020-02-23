#include "LightArea.h"

static Vec3f quad_vertices[8] = {
    Vec3f( 1.f,  1.f, 0.f),
    Vec3f( 1.f, -1.f, 0.f),
    Vec3f(-1.f, -1.f, 0.f),
    Vec3f(-1.f,  1.f, 0.f)
};

LightArea::LightArea(const Transform3f &obj_to_world, std::shared_ptr<Light> light, const bool double_sided, const bool hide_camera)
: Object(obj_to_world, light ? light : std::shared_ptr<Light>(new Light(VEC3F_ZERO)), nullptr, nullptr, hide_camera), double_sided(double_sided)
{
    geom = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_QUAD);

    VERT_DATA * vertices = (VERT_DATA*) rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(VERT_DATA), 4);
    for(uint i = 0; i < 4; i++)
    {
        const Vec3f v = obj_to_world * quad_vertices[i];
        vertices[i].x = v.x; vertices[i].y = v.y; vertices[i].z = v.z;
        bbox.extend(v);
    }

    QUAD_DATA * quad = (QUAD_DATA*) rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, sizeof(QUAD_DATA), 1);
    quad[0].v0 = 0; quad[0].v1 = 1; quad[0].v2 = 2; quad[0].v3 = 3;

    normal = obj_to_world.multiply(Vec3f(0.f, 0.f, -1.f), false);
    normal.normalize();
    area = obj_to_world.multiply(Vec3f(2.f, 0.f, 0.f), false).length() * obj_to_world.multiply(Vec3f(0.f, 2.f, 0.f), false).length();
}

bool LightArea::sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources)
{
    //CHECK IF WE ARE ABOVE THE LIGHT AND !DOUBLE SIDED THEN RETURN FALSE
    RNG &rng = resources.rng; rng.Reset2D();
    for(uint i = 0; i < num_samples; i++)
    {
        const Vec2f rnd = rng.Rand2D();

        const Vec3f light_pos = obj_to_world * Vec3f(rnd[0]*2.f-1.f, rnd[1]*2.f-1.f, 0.f);
        const Vec3f dir = *pos - light_pos;
        const float sqr_len = dir.sqr_length();
        const float cos_theta = normal.dot(dir / sqrtf(sqr_len));
        if(cos_theta < 0.f && !double_sided)
            continue;

        light_samples.emplace_back();
        LightSample &light_sample = light_samples.back();

        light_sample.position = light_pos;
        light_sample.intensity = light->get_emission();
        light_sample.pdf = sqr_len / (area * (fabs(cos_theta) + EPSILON_F));
    }

    return true;
}

bool LightArea::evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources)
{
    if(!intersect.front && !double_sided)
        return false;

    light_sample.position = intersect.position;
    light_sample.intensity = light->get_emission();

    const Vec3f dir = *pos - light_sample.position;
    const float sqr_len = dir.sqr_length();
    const float cos_theta = fabs(normal.dot(dir / sqrtf(sqr_len)));

    light_sample.pdf = sqr_len / (area * cos_theta);

    return true;
}
