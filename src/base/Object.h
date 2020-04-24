#pragma once

#include <string>
#include <unordered_map>

class Object
{
public:
    void set_attribute(const std::string &attribute_name, Object * object)
    {
        attributes[attribute_name] = object;
    }

    template<class T>
    T * get_attribute(const std::string &attribute_name)
    {
        auto attribute = attributes.find(attribute_name);
        if(attribute == attributes.end())
            return nullptr;
        // Maybe use static cast?
        return dynamic_cast<T*>(attribute->second);
    }

    virtual ~Object() = default;
    
private:
    std::unordered_map<std::string, Object *> attributes;
};