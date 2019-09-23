#include "Accelerator.h"

#include "../Math/Helpers.h"
#include "../Math/Types.h"
#include "../Export/DebugObj.h"

Accelerator::Accelerator(std::vector<std::shared_ptr<Object>> &objects)
{
    std::cout << "Creating accelerator..." << '\n';
    if(!objects.empty())
        root = build(objects);
    initialized = true;
    std::cout << "Accelerator Completed." << '\n';

    // std::vector<Vec3f> lines;
    // debug_accelerator(root, lines);
    // DebugObj::Lines(lines);
}

bool Accelerator::intersect(Ray &ray, Surface &surface, const std::shared_ptr<Accelerator::Node> &node)
{
    if(!node)
    {
        ray.inv_dir = 1.f / ray.dir; // This should be done in a constructor
        return intersect(ray, surface, root);
    }

    if(node->leaf)
    {
        for(size_t i = 0; i < node->objects.size(); i++)
            node->objects[i]->intersect(ray, surface);
        return ray.hit;
    }
    else
    {
        bool hit = false;
        if(node->l->bbox.intersect(ray))
            hit = intersect(ray, surface, node->l) || hit;
        if(node->r->bbox.intersect(ray))
            hit = intersect(ray, surface, node->r) || hit;
        return ray.hit;
    }

    return false;
}

std::shared_ptr<Accelerator::Node> Accelerator::build(std::vector<std::shared_ptr<Object>> &objects)
{
    const float object_cost = 1.f;
    const float node_cost = 0.125f;

    std::shared_ptr<Accelerator::Node> node(new Accelerator::Node);

    for(size_t i = 0; i < objects.size(); i++)
        node->bbox.extend(objects[i]->get_bbox());
    const float total_surface_area = node->bbox.surface_area();

    Split min_split;
    const uint num_splits = 16;

    for(uint axis = 0; axis < 3; axis++)
    {
        const float width = node->bbox.length()[axis] / (1.f + num_splits);
        for(uint s = 0; s < num_splits; s++)
        {
            Split split;
            split.axis = axis;
            split.position = (width * (s + 1.f)) + node->bbox.min()[axis];

            Box3f l_bbox, r_bbox;
            float l_cost = 0.f, r_cost = 0.f;
            for(size_t i = 0; i < objects.size(); i++)
            {
                if(objects[i]->get_bbox().center()[axis] < split.position)
                {
                    l_cost += object_cost;
                    l_bbox.extend(objects[i]->get_bbox());
                }
                else if(objects[i]->get_bbox().center()[axis] >= split.position)
                {
                    r_cost += object_cost;
                    r_bbox.extend(objects[i]->get_bbox());
                }
            }

            // No split
            if(r_cost == 0.f || l_cost == 0.f)
                continue;

            split.cost = node_cost + (l_bbox.surface_area()/total_surface_area * l_cost) + (r_bbox.surface_area()/total_surface_area * r_cost);
            min_split = (split.cost < min_split.cost) ? split : min_split;
        }
    }

    const float leaf_cost = object_cost * objects.size();
    if(leaf_cost <= min_split.cost + EPSILON_F)
    {
        //Create a leaf
        node->leaf = true;
        node->objects = objects;
    }
    else
    {
        //Recursive split
        node->leaf = false;
        std::vector<std::shared_ptr<Object>> l_objects, r_objects;
        for(size_t i = 0; i < objects.size(); i++)
        {
            if(objects[i]->get_bbox().center()[min_split.axis] < min_split.position)
                l_objects.push_back(objects[i]);
            else if(objects[i]->get_bbox().center()[min_split.axis] >= min_split.position)
                r_objects.push_back(objects[i]);
        }
        node->l = build(l_objects);
        node->r = build(r_objects);
    }

    return node;
}

void Accelerator::debug_accelerator(std::shared_ptr<Accelerator::Node> node, std::vector<Vec3f> &lines)
{
    if(!node->leaf)
    {
        debug_accelerator(node->l, lines);
        debug_accelerator(node->r, lines);
    }
    else
    {
        Vec3f p0(node->bbox.min()[0], node->bbox.min()[1], node->bbox.min()[2]);
        Vec3f p1(node->bbox.min()[0], node->bbox.min()[1], node->bbox.max()[2]);
        Vec3f p2(node->bbox.min()[0], node->bbox.max()[1], node->bbox.max()[2]);
        Vec3f p3(node->bbox.min()[0], node->bbox.max()[1], node->bbox.min()[2]);

        Vec3f t0(node->bbox.max()[0], node->bbox.min()[1], node->bbox.min()[2]);
        Vec3f t1(node->bbox.max()[0], node->bbox.min()[1], node->bbox.max()[2]);
        Vec3f t2(node->bbox.max()[0], node->bbox.max()[1], node->bbox.max()[2]);
        Vec3f t3(node->bbox.max()[0], node->bbox.max()[1], node->bbox.min()[2]);

        lines.push_back(p0); lines.push_back(p1);
        lines.push_back(p1); lines.push_back(p2);
        lines.push_back(p2); lines.push_back(p3);
        lines.push_back(p3); lines.push_back(p0);

        lines.push_back(t0); lines.push_back(t1);
        lines.push_back(t1); lines.push_back(t2);
        lines.push_back(t2); lines.push_back(t3);
        lines.push_back(t3); lines.push_back(t0);

        lines.push_back(p0); lines.push_back(t0);
        lines.push_back(p1); lines.push_back(t1);
        lines.push_back(p2); lines.push_back(t2);
        lines.push_back(p3); lines.push_back(t3);
    }
}
