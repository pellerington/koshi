#pragma once

#include <koshi/Vec4.h>
#include <koshi/Intersect.h>

KOSHI_OPEN_NAMESPACE

/*
DEVICE_FUNCTION Vec4f evaluate(intersect, void * data)
{
    TextureConstantData * cdata = (TextureConstantData *)data;

    float x = evaluate<float>(Inputs::COLOR, intersect, resources);

    return cdata->color;
}

enum TextureConstantInputs {
    COLOR, ROUGHNESS,
}

scene->setInput(Inputs::COLOR, Texture *, Texture *)

during scene->prepare() set all node inputs[i] to what they were set to.

Node
{
inputs:
    uint inputs[MAX_INPUTS];
}
*/

#define MAX_INPUTS 16

enum TextureType
{
    NONE
};

class Texture
{
    Texture() : type(TextureType::NONE) {}

    // TODO: Will this virtual be inlined on GPU?
    virtual DEVICE_FUNCTION Vec4f evaluate(Intersect * intersect) = 0;

protected:
    const TextureType type;

private:
    // TOOD: this should be in node class not here
    uint inputs[MAX_INPUTS];
};

KOSHI_CLOSE_NAMESPACE