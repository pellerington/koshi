#pragma once

#include <koshi/Vec4.h>
#include <koshi/Intersect.h>

KOSHI_OPEN_NAMESPACE

#define MAX_INPUTS 16


// TODO: Instead runtime generate a code based on an XML (MaterialX) file for the material.


// enum TextureType
// {
//     NONE
// };

// class Texture
// {
//     Texture() : type(TextureType::NONE) {}

//     // TODO: Will this virtual be inlined on GPU?
//     virtual DEVICE_FUNCTION Vec4f evaluate(Intersect * intersect) = 0;

// protected:
//     const TextureType type;

// private:
//     // TOOD: this should be in node class not here
//     uint inputs[MAX_INPUTS];
// };

KOSHI_CLOSE_NAMESPACE