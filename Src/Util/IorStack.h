#pragma once

struct IorStack {
    IorStack(const float curr_ior = 1.f, const IorStack * prev = nullptr) : curr_ior(curr_ior), prev(prev) {}
    float curr_ior;
    const IorStack * prev;
};
