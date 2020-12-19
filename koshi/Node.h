#pragma once

#include <string>

#include <koshi/Koshi.h>

KOSHI_OPEN_NAMESPACE

class Node
{
public:
    bool setInput(const std::string& input, const std::string& name, Node * node);
    // Don't store input and just have scene->readTexture(this, "color", intersect)??? How will that work with MaterialX???
    void dirtyInput(const std::string& input, const uint& flag);
private:
    
};

KOSHI_CLOSE_NAMESPACE