#pragma once

#include <iostream>
#include <vector>

//TODO: Compare performance when changing this.
#define MIN_PAGE_SIZE 1024ul

class Memory
{
public:
    Memory()
    {
        pages.emplace_back();
        curr_page = 0;
        curr_memory = 0;
        pages[curr_page].memory = (char*)malloc(MIN_PAGE_SIZE);
        pages[curr_page].max_memory = MIN_PAGE_SIZE;
    }

    template <class T, typename... Args>
    T * create(Args&&... args)
    {
        uint object_address = curr_memory;

        curr_memory += sizeof(T);

        if(curr_memory >= pages[curr_page].max_memory)
        {
            curr_page++;
            if(curr_page >= pages.size())
            {
                pages.emplace_back();
                uint page_size = std::max(MIN_PAGE_SIZE, sizeof(T));
                pages[curr_page].memory = (char*)malloc(page_size);
                pages[curr_page].max_memory = page_size;
            }
            else if(sizeof(T) > pages[curr_page].max_memory)
            {
                free(pages[curr_page].memory);
                pages[curr_page].memory = (char*)malloc(sizeof(T));
                pages[curr_page].max_memory = sizeof(T);
            }
            object_address = 0;
            curr_memory = sizeof(T);
        }
        
        return new(pages[curr_page].memory + object_address) T(std::forward<Args>(args)...);
    }

    void clear() { curr_page = 0; curr_memory = 0; }

private:
    uint curr_page, curr_memory;
    
    struct Page {
        char * memory;
        uint max_memory;
    };
    std::vector<Page> pages;
};
