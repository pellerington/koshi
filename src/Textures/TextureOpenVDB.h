#pragma once

#include <Textures/Texture.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

class TexutreOpenVDB : public Texture
{
public:
    TexutreOpenVDB(const std::string& filename, const std::string& gridname)
    {
        openvdb::initialize(); // call this static somehow?

        // Store common pointers in a static thing? or maybe openvdb does that anyway?
        openvdb::io::File file(filename);
        file.open();
        openvdb::GridBase::Ptr base_grid;
        for (openvdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName(); ++nameIter)
        {
            if(nameIter.gridName() == gridname)
            {
                base_grid = file.readGrid(nameIter.gridName());
                grid_ptr = openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid);
            }
        }
        file.close();

        openvdb::CoordBBox bbox = grid_ptr->evalActiveVoxelBoundingBox();
        for(uint i = 0; i < 3; i++)
        {
            min[i] = bbox.min()[i];
            len[i] = bbox.max()[i] - bbox.min()[i];
        }

        openvdb::Coord coord = grid_ptr->evalActiveVoxelDim();
        std::cout << "openvdb size: "  << " " << len[0] << " " << len[1] << " " << len[2] << "\n";

        inv_len = 1.f / len;

    }

    void pre_render(Resources& resources)
    {
        num_threads = resources.settings->num_threads;
    }

    float get_float(const float& u, const float& v, const float& w, Resources& resources)
    {
        // TODO: Use threads to do this fasterer.
        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*grid_ptr);
        return sampler.isSample(openvdb::Vec3f(u * len.x + min.x, v * len.y + min.y, w * len.z + min.z));
    }

    Vec3f get_vec3f(const float &u, const float &v, const float &w, Resources &resources)
    {
        return get_float(u, v, w, resources);
    }

    virtual Vec3f delta() const { return inv_len; }

private:
    uint num_threads;
    openvdb::FloatGrid::Ptr grid_ptr;
    Vec3f min;
    Vec3f len;
    Vec3f inv_len;
};
