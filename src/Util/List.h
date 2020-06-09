#pragma once



template <class T>
class List
{
public:

    List(Resources& resource) : memory(resource.memory), size(0) {}

    // Better name? Itertator?
    class Item
    {
        public:
            Item * next() { return next; }
            T * get() { return object; }
            const T * get() const { return object; }
        private:
            T * object;
            Item * next = nullptr;
            Item * prev = nullptr;
    };

    void push(T * object)
    {
        Item * item = memory->create<Item>();
        item->object = object;
        item->next = start->next;
        if(item->next)
            item->next->prev = item;
        start->next = item;
    }

    void pop(Item * item)
    {
        if(item->next) item->next->prev = item->prev;
        if(item->prev) item->prev->next = item->next;
        size--;
    }

    inline Item * begin() { return start->next; }
    inline const Item * begin() const { return start->next; }

private:
    Memory& memory;
    uint size;
    Item start;
};