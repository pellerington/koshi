#pragma once

#include <koshi/base/Object.h>
#include <vector>
#include <iostream>

class ObjectGroup : public Object
{
public:
    inline bool empty() { return group.empty(); }
    inline size_t size() { return group.size(); }

    inline Object * get(const size_t& i) { return group[i]; }
    inline Object * operator[](const size_t& i) { return group[i]; }

    template<class T>
    inline T * get(const size_t& i) { return dynamic_cast<T*>(group[i]); }
    template<class T>
    inline T * operator[](const size_t& i) { return dynamic_cast<T*>(group[i]); }

    void push(Object * object) { group.push_back(object); }

private:
    std::vector<Object*> group;
};