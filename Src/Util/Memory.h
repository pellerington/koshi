#pragma once

class Memory
{
public:
    Memory()
    {
        curr_memory = 0;
        max_memory = sizeof(char)*1024;
        arena = (char*)malloc(max_memory);
    }

    template <class T, typename... Args>
    T * create(Args&&... args)
    {
        const uint obj_memory = curr_memory;
        curr_memory += sizeof(T);
        if(curr_memory > max_memory)
        {
            max_memory = curr_memory * 2;
            arena = (char*)realloc(arena, max_memory);
        }
        return new(arena + obj_memory) T(std::forward<Args>(args)...);
    }

    void clear() { curr_memory = 0; }

private:
    uint curr_memory, max_memory;
    char * arena;
};
