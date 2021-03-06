#pragma once

#include <string>
#include <unordered_map>
#include <koshi/dependency/robin_hood.h>
class Resources;
class Intersect;

class Object
{
public:

    virtual void pre_render(Resources& resources) {}

    void set_attribute(const std::string& attribute_name, Object * object)
    {
        if(!attributes)
            attributes = new robin_hood::unordered_map<std::string, Object *>();
        (*attributes)[attribute_name] = object;
    }

    template<class T>
    T * get_attribute(const std::string& attribute_name)
    {
        if(!attributes)
            return nullptr;
        auto attribute = attributes->find(attribute_name);
        if(attribute == attributes->end())
            return nullptr;
        // Use static cast?
        return dynamic_cast<T*>(attribute->second);
    }

    virtual ~Object() { if(attributes) delete attributes; }
    
private:
    robin_hood::unordered_map<std::string, Object *> * attributes = nullptr;
};