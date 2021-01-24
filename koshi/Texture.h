#pragma once

#include <koshi/Koshi.h>

KOSHI_OPEN_NAMESPACE

// Host/Device
// Texture -> Static Function + Inputs???
// Material -> Static Function + Inputs???

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

KOSHI_CLOSE_NAMESPACE