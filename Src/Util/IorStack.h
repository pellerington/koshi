#pragma once

class IorStack
{
public:
    IorStack(const float curr_ior = 1.f, const IorStack * prev = nullptr) : curr_ior(curr_ior), prev(prev) {}

    inline const IorStack * get_prev_stack() const { return prev ? prev : this; }
    inline float get_curr_ior() const { return curr_ior; }
    inline float get_prev_ior() const { return prev ? prev->get_curr_ior() : curr_ior; }

private:
    float curr_ior;
    const IorStack * prev;
};
