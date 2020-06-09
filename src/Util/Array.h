#pragma once

#include <Util/Resources.h>

#define DEFAULT_SIZE 16

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
        // If max size expand and copy.
        array[curr_size++] = object; 
    }

    void resize(const uint& new_size) 
    {
        if(new_size < curr_size)
            curr_size = new_size;
        // else resize max_size
    }

    inline const uint& size() const { return curr_size; }

    T& operator[](const uint& i) { return array[i]; }
    const T& operator[](const uint& i) const { return array[i]; }

private:
    Memory * memory;
    uint curr_size, max_size;
    T * array;
};
