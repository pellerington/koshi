#pragma once

#include <intersection/Intersect.h>

typedef void (PreIntersectionCallback)(IntersectList * intersects, void * data, Resources& resources);

// ObjectIntersectCallback?

// HitIntersectionCallback?

typedef void (PostIntersectionCallback)(IntersectList * intersects, void * data, Resources& resources);

struct IntersectionCallbacks : public Object
{
    IntersectionCallbacks()
    : pre_intersection_cb(nullptr),  pre_intersection_data(nullptr),
      post_intersection_cb(nullptr), post_intersection_data(nullptr)
    {
    }

    IntersectionCallbacks
    (
        PreIntersectionCallback * pre_intersection_cb,   void * pre_intersection_data,
        PostIntersectionCallback * post_intersection_cb, void * post_intersection_data
    ) 
    : pre_intersection_cb(pre_intersection_cb),   pre_intersection_data(pre_intersection_data),
      post_intersection_cb(post_intersection_cb), post_intersection_data(post_intersection_data)
    {
    }

    PreIntersectionCallback * pre_intersection_cb;
    void * pre_intersection_data;

    PostIntersectionCallback * post_intersection_cb;
    void * post_intersection_data;
};