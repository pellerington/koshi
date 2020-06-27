#include <embree/Embree.h>
#include <embree/EmbreeGeometryArea.h>
#include <embree/EmbreeGeometryBox.h>

RTCDevice Embree::rtc_device = rtcNewDevice("");

// Area arrays
const float EmbreeGeometryArea::vertices[4][4] = {
    { AREA_LENGTH*0.5f,  AREA_LENGTH*0.5f, 0.f, 0.f},
    { AREA_LENGTH*0.5f, -AREA_LENGTH*0.5f, 0.f, 0.f},
    {-AREA_LENGTH*0.5f, -AREA_LENGTH*0.5f, 0.f, 0.f},
    {-AREA_LENGTH*0.5f,  AREA_LENGTH*0.5f, 0.f, 0.f}
};
const uint EmbreeGeometryArea::indices[1][4] = {{0,1,2,3}};

// Box arrays
const float EmbreeGeometryBox::vertices[8][4] = {
    {-BOX_LENGTH*0.5f, -BOX_LENGTH*0.5f, -BOX_LENGTH*0.5f, 0.f},
    { BOX_LENGTH*0.5f, -BOX_LENGTH*0.5f, -BOX_LENGTH*0.5f, 0.f},
    { BOX_LENGTH*0.5f, -BOX_LENGTH*0.5f,  BOX_LENGTH*0.5f, 0.f},
    {-BOX_LENGTH*0.5f, -BOX_LENGTH*0.5f,  BOX_LENGTH*0.5f, 0.f},
    {-BOX_LENGTH*0.5f,  BOX_LENGTH*0.5f, -BOX_LENGTH*0.5f, 0.f},
    { BOX_LENGTH*0.5f,  BOX_LENGTH*0.5f, -BOX_LENGTH*0.5f, 0.f},
    { BOX_LENGTH*0.5f,  BOX_LENGTH*0.5f,  BOX_LENGTH*0.5f, 0.f},
    {-BOX_LENGTH*0.5f,  BOX_LENGTH*0.5f,  BOX_LENGTH*0.5f, 0.f}
};
const uint EmbreeGeometryBox::indices[6][4] = {
    {0, 4, 5, 1},
    {1, 5, 6, 2},
    {2, 6, 7, 3},
    {0, 3, 7, 4},
    {4, 7, 6, 5},
    {0, 1, 2, 3},
};