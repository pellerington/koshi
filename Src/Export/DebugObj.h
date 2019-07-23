#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <Eigen/Core>

class DebugObj
{
public:
    static void Points(std::vector<Vec3f> &pts)
    {
        std::ofstream out("debug.obj");

        for(size_t i = 0; i < pts.size(); i++)
        {
            out << "v " << pts[i][0] << " " << pts[i][1] << " " << pts[i][2] << std::endl;
            out << "v " << pts[i][0] << " " << pts[i][1] << " " << pts[i][2] << std::endl;
            out << "v " << pts[i][0] << " " << pts[i][1] << " " << pts[i][2] << std::endl;
        }

        for(size_t i = 0; i < pts.size(); i++)
            out << "f " << i*3 + 1 << " " << i*3 + 2 << " " << i*3 + 3 << "\n";

        out.close();
    }

    static void Lines(std::vector<Vec3f> &pts)
    {
        std::ofstream out("debug.obj");

        for(size_t i = 0; i < pts.size(); i=i+2)
        {
            out << "v " << pts[i][0] << " " << pts[i][1] << " " << pts[i][2] << std::endl;
            out << "v " << pts[i][0] << " " << pts[i][1] << " " << pts[i][2] << std::endl;
            out << "v " << pts[i+1][0] << " " << pts[i+1][1] << " " << pts[i+1][2] << std::endl;
        }

        for(size_t i = 0; i < pts.size()/2; i++)
            out << "f " << i*3 + 1 << " " << i*3 + 2 << " " << i*3 + 3 << "\n";

        out.close();
    }
};
