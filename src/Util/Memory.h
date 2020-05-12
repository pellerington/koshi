#pragma once

#include <iostream>
#include <vector>

//TODO: Compare performance when changing this.
#define MIN_PAGE_SIZE 262144u

#include <malloc.h>

class Memory
{
public:
    Memory()
    {
        pages.emplace_back();
        curr_page = 0;
        curr_memory = 0;
        pages[curr_page].memory_address = (uint8_t*)malloc(MIN_PAGE_SIZE);
        pages[curr_page].max_memory = MIN_PAGE_SIZE;
    }

    template <class T, typename... Args>
    T * create(Args&&... args)
    {
        uint size = 2 * ((sizeof(T) + 15) & (~15));
        uint object_address = curr_memory;
        curr_memory = object_address + size;

        if(curr_memory >= pages[curr_page].max_memory)
        {
            curr_page++;
            if(curr_page >= pages.size())
            {
                pages.emplace_back();
                uint page_size = std::max(MIN_PAGE_SIZE, size);
                pages[curr_page].memory_address = (uint8_t*)malloc(page_size);
                pages[curr_page].max_memory = page_size;
            }
            else if(size > pages[curr_page].max_memory)
            {
                free(pages[curr_page].memory_address);
                pages[curr_page].memory_address = (uint8_t*)malloc(size);
                pages[curr_page].max_memory = size;
            }
            object_address = 0;
            curr_memory = size;
        }
        
        return new(pages[curr_page].memory_address + object_address) T(std::forward<Args>(args)...);
    }

    void clear() { curr_page = 0; curr_memory = 0; }

private:
    uint curr_page, curr_memory;
    struct Page {
        uint8_t * memory_address;
        uint max_memory;
    };
    std::vector<Page> pages;
};
