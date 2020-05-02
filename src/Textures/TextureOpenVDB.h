#pragma once

#include <Textures/Texture.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

class TexutreOpenVDB : public Texture
{
public:
    TexutreOpenVDB(const std::string filename, const std::string gridname, const uint num_threads) : num_threads(num_threads)
    {
        openvdb::initialize(); // call this static somehow?

        // Store common pointers in a static thing? or maybe openvdb does that anyway?
        openvdb::io::File file(filename);
        file.open();
        openvdb::GridBase::Ptr base_grid;
        for (openvdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName(); ++nameIter)
            if(nameIter.gridName() == gridname)
            {
                base_grid = file.readGrid(nameIter.gridName());
                grid_ptr = openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid);
            }
        file.close();

        openvdb::CoordBBox bbox = grid_ptr->evalActiveVoxelBoundingBox();
        for(uint i = 0; i < 3; i++)
        {
            min[i] = bbox.min()[i];
            len[i] = bbox.max()[i] - bbox.min()[i];
        }

        openvdb::Coord coord = grid_ptr->evalActiveVoxelDim();
        std::cout << coord  << " " << len[0] << " " << len[1] << " " << len[2] << "\n";

    }

    const float get_float(const float &u, const float &v, const float &w, Resources &resources)
    {
        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*grid_ptr);
        return sampler.isSample(openvdb::Vec3f(u * len[0] + min[0], v * len[1] + min[1], w * len[2] + min[2]));
    }

private:
    const uint num_threads;
    openvdb::FloatGrid::Ptr grid_ptr;
    float min[3];
    float len[3];
};
