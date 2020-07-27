#pragma once

#include <koshi/base/Resources.h>

#define DEFAULT_SIZE 8

// TODO: Have this be constructed by the Memory object!
template <class T>
class Array
{
public:
    Array(Memory * memory, const uint& init_size = DEFAULT_SIZE) 
    : memory(memory), curr_size(0), max_size(init_size) 
    {
        array = memory->create_array<T>(max_size);
    }

    inline void push(const T& object) 
    { 
        if(curr_size == max_size)
        {
            max_size *= 2;
            T * new_array = memory->create_array<T>(max_size);
            for(uint i = 0; i < curr_size; i++)
                new_array[i] = array[i];
            array = new_array;
        }
        array[curr_size++] = object;
    }

    inline T& push()
    {
        if(curr_size == max_size)
        {
            max_size *= 2;
            T * new_array = memory->create_array<T>(max_size);
            for(uint i = 0; i < curr_size; i++)
                new_array[i] = array[i];
            array = new_array;
        }
        return array[curr_size++];
    }

    inline void resize(const uint& new_size) 
    {
        if(new_size > max_size)
        {
            while(max_size < new_size)
                 max_size *= 2;
            T * new_array = memory->create_array<T>(max_size);
            for(uint i = 0; i < curr_size; i++)
                new_array[i] = array[i];
            array = new_array;
        }

        curr_size = new_size;
    }

    inline const uint& size() const { return curr_size; }

    T& operator[](const uint& i) { return array[i]; }
    const T& operator[](const uint& i) const { return array[i]; }

private:
    Memory * memory;
    uint curr_size, max_size;
    T * array;
};
