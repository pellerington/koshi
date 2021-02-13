#pragma once

#include <string>

#include <koshi/Koshi.h>

#define MAX_STRING_LENGTH 64u

KOSHI_OPEN_NAMESPACE

class String 
{
public:
    String() {}

    String(const std::string& str)
    {
        uint size = std::min(MAX_STRING_LENGTH, (uint)str.size());
        str.copy(data, size);
        data[size] = '\0';
    }

    DEVICE_FUNCTION bool operator==(const char * str) const
    {
        for(uint i = 0; i < MAX_STRING_LENGTH; i++)
        {
            if(str[i] != data[i]) return false;
            if(str[i] == '\0') return true;
        }
        return true;
    }

    bool operator==(const std::string& str) const
    {
        return std::string(data) == str;
    }

    operator std::string() const { return std::string(data); }


private:
    char data[MAX_STRING_LENGTH];
};

KOSHI_CLOSE_NAMESPACE