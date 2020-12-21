#include <koshi/geometry/GeometryVolume.h>

#include <koshi/material/MaterialVolume.h>

#include <fstream>

const Box3f GeometryVolume::bbox = Box3f(Vec3f(-VOLUME_LENGTH*0.5f), Vec3f(VOLUME_LENGTH*0.5f));

GeometryVolume::GeometryVolume(const Transform3f& obj_to_world, const GeometryVisibility& visibility)
: Geometry(obj_to_world, visibility)
{
    obj_bbox = bbox;
    world_bbox = obj_to_world * obj_bbox;
}

void GeometryVolume::pre_render(Resources& resources)
{
    MaterialVolume * material = get_attribute<MaterialVolume>("material");

    if(!material)
        return;

    // TODO: These BVHs are dependent on the material being integrated...
    if(material->homogenous())
    {
        VolumeBox3f b;
        b.bbox = obj_bbox;
        b.max_density = b.min_density = material->get_density(VEC3F_ZERO, nullptr, resources);
        acceleration_structure.push_back(b);
    }
    else
    {
        Vec3f delta = material->get_density_texture()->delta();
        if(delta <= 0.f) delta = 0.03125f; // Delta for procedurals.

        // Setup dim x dim x dim sized grid to read points into.
        Vec3f max_density[VOLUME_BVH_SPLITS][VOLUME_BVH_SPLITS][VOLUME_BVH_SPLITS];
        Vec3f min_density[VOLUME_BVH_SPLITS][VOLUME_BVH_SPLITS][VOLUME_BVH_SPLITS];
        for(uint x = 0; x < VOLUME_BVH_SPLITS; x++)
        for(uint y = 0; y < VOLUME_BVH_SPLITS; y++)
        for(uint z = 0; z < VOLUME_BVH_SPLITS; z++)
        {
            max_density[x][y][z] = FLT_MIN;
            min_density[x][y][z] = FLT_MAX;
        }

        // Read all points.
        Vec3f uvw;
        for(uvw.u = delta.u * 0.5f; uvw.u < 1.f; uvw.u += delta.u)
        for(uvw.v = delta.v * 0.5f; uvw.v < 1.f; uvw.v += delta.v)
        for(uvw.z = delta.w * 0.5f; uvw.w < 1.f; uvw.w += delta.w)
        {
            const Vec3f density = material->get_density(uvw, nullptr, resources);
            // TODO: Each point could fit into multiple grids.
            const Vec3u i = uvw * VOLUME_BVH_SPLITS;
            max_density[i.u][i.v][i.w].max(density);
            min_density[i.u][i.v][i.w].min(density); 
        }

        split_acceleration_structure(max_density, min_density, Vec3u(0,0,0), Vec3u(VOLUME_BVH_SPLITS, VOLUME_BVH_SPLITS, VOLUME_BVH_SPLITS));
    

        // // DEBUG:
        std::cout << "bbox: " << acceleration_structure.size() << "\n";
        std::ofstream out("debug.obj");
        for(uint i = 0; i < acceleration_structure.size(); i++)
        {
            const Vec3f& min = acceleration_structure[i].bbox.min();
            const Vec3f& max = acceleration_structure[i].bbox.max();
            const Vec3u l[24][2] = {
                { Vec3u(0,0,0), Vec3u(1,0,0) }, { Vec3u(0,0,0), Vec3u(0,1,0) }, { Vec3u(0,0,0), Vec3u(0,0,1) },
                { Vec3u(1,0,0), Vec3u(0,0,0) }, { Vec3u(1,0,0), Vec3u(1,1,0) }, { Vec3u(1,0,0), Vec3u(1,0,1) },
                { Vec3u(0,1,0), Vec3u(1,1,0) }, { Vec3u(0,1,0), Vec3u(0,0,0) }, { Vec3u(0,1,0), Vec3u(0,1,1) },
                { Vec3u(1,1,0), Vec3u(0,1,0) }, { Vec3u(1,1,0), Vec3u(1,0,0) }, { Vec3u(1,1,0), Vec3u(1,1,1) },
                { Vec3u(0,0,1), Vec3u(1,0,1) }, { Vec3u(0,0,1), Vec3u(0,1,1) }, { Vec3u(0,0,1), Vec3u(0,0,0) },
                { Vec3u(1,0,1), Vec3u(0,0,1) }, { Vec3u(1,0,1), Vec3u(1,1,1) }, { Vec3u(1,0,1), Vec3u(1,0,0) },
                { Vec3u(0,1,1), Vec3u(1,1,1) }, { Vec3u(0,1,1), Vec3u(0,0,1) }, { Vec3u(0,1,1), Vec3u(0,1,0) },
                { Vec3u(1,1,1), Vec3u(0,1,1) }, { Vec3u(1,1,1), Vec3u(1,0,1) }, { Vec3u(1,1,1), Vec3u(1,1,0) }
            };
            for(uint j = 0; j < 24; j++)
            {
                out << "v " << (!l[j][0][0] ? min[0] : max[0]) << " " << (!l[j][0][1] ? min[1] : max[1]) << " " << (!l[j][0][2] ? min[2] : max[2]) << "\n";
                out << "v " << (!l[j][1][0] ? min[0] : max[0]) << " " << (!l[j][1][1] ? min[1] : max[1]) << " " << (!l[j][1][2] ? min[2] : max[2]) << "\n";
                out << "v " << (!l[j][0][0] ? min[0] : max[0]) << " " << (!l[j][0][1] ? min[1] : max[1]) << " " << (!l[j][0][2] ? min[2] : max[2]) << "\n";
            }
        }
        for(uint i = 0; i < acceleration_structure.size(); i++)
        {
            for(uint j = 0; j < 24; j++)
            {
                out << "f " << (24*i*3) + j*3 + 1 << " " << (24*i*3) + j*3 + 2 << " " << (24*i*3) + j*3 + 3 << "\n";
            }
        }

    }

    // Display acceleration structure somehow...
}

void GeometryVolume::split_acceleration_structure(Vec3f max_density[VOLUME_BVH_SPLITS][VOLUME_BVH_SPLITS][VOLUME_BVH_SPLITS], Vec3f min_density[VOLUME_BVH_SPLITS][VOLUME_BVH_SPLITS][VOLUME_BVH_SPLITS], const Vec3u& n0, const Vec3u& n1)
{
    Vec3f curr_max_density, curr_min_density;
    auto calculate_cost = [&](const Vec3u& s0, const Vec3u& s1)
    {
        curr_max_density = FLT_MIN, curr_min_density = FLT_MAX;
        for(uint x = s0.x; x < s1.x; x++)
        for(uint y = s0.y; y < s1.y; y++)
        for(uint z = s0.z; z < s1.z; z++)
        {
            curr_max_density.max(max_density[x][y][z]);
            curr_min_density.min(min_density[x][y][z]);
        }

        // TODO: World or obj size???? If density scales we should use obj???

        Vec3f world_size = world_bbox.length() * (Vec3f(s1 - s0) / VOLUME_BVH_SPLITS);
        float volume = world_size[0] * world_size[1] * world_size[2];
        return ceilf(volume * (curr_max_density /*- curr_min_density*/).max());

        // Vec3f world_size = world_bbox.length() * (Vec3f(s1 - s0) / VOLUME_BVH_SPLITS);
        // const float diagonal = world_size.length();
        // const float surface_area = 2*(world_size[0]*world_size[1] + world_size[0]*world_size[2] + world_size[1]*world_size[2]);
        // return surface_area * diagonal * (curr_max_density /*- curr_min_density*/).max();
    };

    uint split[2] = {0, 0};
    float cost = calculate_cost(n0, n1);
    const Vec3f total_max_density = curr_max_density;
    const Vec3f total_min_density = curr_min_density;

    // std::cout << "\ninital cost: " << cost << "\n"; 
    if(total_max_density.null()) return;

    for(uint axis = 0; axis < 3; axis++)
    {
        for(uint n = n0[axis]+1; n < n1[axis]; n++)
        {
            Vec3f n0_split = n1; n0_split[axis] = n;
            Vec3f n1_split = n0; n1_split[axis] = n;
            float curr_cost = calculate_cost(n0, n0_split) + calculate_cost(n1_split, n1);
            if(curr_cost < cost)
            {
                cost = curr_cost;
                split[0] = axis; split[1] = n;
            }
            // std::cout << cost << " " << curr_cost << "\n";
        }
    }

    if(split[1] == 0)
    {
        VolumeBox3f b;
        b.bbox = Box3f(obj_bbox.min() + (obj_bbox.length() * n0) / VOLUME_BVH_SPLITS, 
                       obj_bbox.min() + (obj_bbox.length() * n1) / VOLUME_BVH_SPLITS);
        b.max_density = total_max_density;
        b.min_density = total_min_density;
        acceleration_structure.push_back(b);
    }
    else
    {
        Vec3f n0_split = n1; n0_split[split[0]] = split[1];
        Vec3f n1_split = n0; n1_split[split[0]] = split[1];
        split_acceleration_structure(max_density, min_density, n0, n0_split);
        split_acceleration_structure(max_density, min_density, n1_split, n1);
    }
}