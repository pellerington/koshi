#pragma once

#include <cstdint>

#include <koshi/Koshi.h>

KOSHI_OPEN_NAMESPACE

enum Format { INT8, INT16, INT32, UINT8, UINT16, UINT32, FLOAT16, FLOAT32, DOUBLE, };

DEVICE_FUNCTION uint sizeofFormat(const Format& format)
{
    switch(format)
    {
    case INT8:    return sizeof(int8_t);
    case INT16:   return sizeof(int16_t);
    case INT32:   return sizeof(int32_t);
    case UINT8:   return sizeof(uint8_t);
    case UINT16:  return sizeof(uint16_t);
    case UINT32:  return sizeof(uint32_t);
    // case FLOAT16: return sizeof(half);
    case FLOAT32: return sizeof(float);
    case DOUBLE: return sizeof(double);
    default:      return 0;
    }
}

KOSHI_CLOSE_NAMESPACE