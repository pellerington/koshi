#pragma once

#include <vector>
#include <memory>
#include "../Objects/Object.h"

class Accelerator
{
public:
    struct Node
    {
        bool leaf;
        Box3f bbox;
        std::shared_ptr<Node> l;
        std::shared_ptr<Node> r;
        std::vector<std::shared_ptr<Object>> objects;
    };
    Accelerator() { initialized = false; }
    Accelerator(std::vector<std::shared_ptr<Object>> &objects);
    bool intersect(Ray &ray, Surface &surface, const std::shared_ptr<Accelerator::Node> &node = nullptr);
    bool is_initialized() { return initialized; }
private:
    bool initialized = false;
    struct Split
    {
        uint axis = 0;
        float position = 0.f;
        float cost = FLT_MAX;
    };
    std::shared_ptr<Accelerator::Node> build(std::vector<std::shared_ptr<Object>> &objects);
    std::shared_ptr<Accelerator::Node> root;
    void debug_accelerator(std::shared_ptr<Accelerator::Node> node, std::vector<Vec3f> &lines);
};
