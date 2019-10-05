#pragma once

#include <smmintrin.h> // SSE4.1

class Vec3f
{
public:
    union {
        struct { float x, y, z; };
        struct { float r, g, b; };
        struct { float u, v, w; };
        __m128 data;
    };

    Vec3f() : data(_mm_setzero_ps()) {}
    Vec3f(const float &x, const float &y, const float &z, const float &t = 0.f) : data(_mm_setr_ps(x, y, z, t)) {}
    Vec3f(const float &n) : data(_mm_setr_ps(n, n, n, 0.f)) {}
    Vec3f(const __m128 &data) : data(data) {}

    inline float& operator[](const int i) { return data[i]; }
    inline const float& operator[](const int i) const { return data[i]; }

    // Assignement
    inline Vec3f& operator= (const Vec3f& other) { data = other.data; return *this; }
    inline Vec3f& operator= (const float &n) { data = _mm_setr_ps(n, n, n, 0.f); return *this; }

    // Negate
    inline Vec3f operator-() const { return Vec3f(_mm_sub_ps(_mm_setzero_ps(), data)); }

    // Addition
    inline Vec3f& operator+= (const Vec3f &other) { data = _mm_add_ps(data, other.data); return *this; }
    inline Vec3f operator+ (const Vec3f &other) const { return Vec3f(_mm_add_ps(data, other.data)); }
    inline Vec3f& operator+= (const float &n) { data = _mm_add_ps(data, _mm_set_ps1(n)); return *this; }
    inline Vec3f operator+ (const float &n) const { return Vec3f(_mm_add_ps(data, _mm_set_ps1(n))); }
    friend inline Vec3f operator+ (const float &n, const Vec3f &other) { return Vec3f(_mm_add_ps(other.data, _mm_set_ps1(n))); }

    // Multiply
    inline Vec3f& operator*= (const Vec3f &other) { data = _mm_mul_ps(data, other.data); return *this; }
    inline Vec3f operator* (const Vec3f &other) const { return Vec3f(_mm_mul_ps(data, other.data)); }
    inline Vec3f& operator*= (const float &n) { data = _mm_mul_ps(data, _mm_set_ps1(n)); return *this; }
    inline Vec3f operator* (const float &n) const { return Vec3f(_mm_mul_ps(data, _mm_set_ps1(n))); }
    friend inline Vec3f operator* (const float &n, const Vec3f &other) { return Vec3f(_mm_mul_ps(other.data, _mm_set_ps1(n))); }

    // Subtract
    inline Vec3f& operator-= (const Vec3f &other) { data = _mm_sub_ps(data, other.data); return *this; }
    inline Vec3f operator- (const Vec3f &other) const { return Vec3f(_mm_sub_ps(data, other.data)); }
    inline Vec3f& operator-= (const float &n) { data = _mm_sub_ps(data, _mm_set_ps1(n)); return *this; }
    inline Vec3f operator- (const float &n) const { return Vec3f(_mm_sub_ps(data, _mm_set_ps1(n))); }
    friend inline Vec3f operator- (const float &n, const Vec3f &other) { return Vec3f(_mm_sub_ps(_mm_set_ps1(n), other.data)); }

    // Divide
    inline Vec3f& operator/= (const Vec3f &other) { data = _mm_div_ps(data, other.data); return *this; }
    inline Vec3f operator/ (const Vec3f &other) const { return Vec3f(_mm_div_ps(data, other.data)); }
    inline Vec3f& operator/= (const float &n) { data = _mm_div_ps(data, _mm_set_ps1(n)); return *this; }
    inline Vec3f operator/ (const float &n) const { return Vec3f(_mm_div_ps(data, _mm_set_ps1(n))); }
    friend inline Vec3f operator/ (const float &n, const Vec3f &other) { return Vec3f(_mm_div_ps(_mm_set_ps1(n), other.data)); }

    // Replace these
    inline float length() const {
        __m128 sqr = _mm_mul_ps(data, data);
        return sqrt(sqr[0] + sqr[1] + sqr[2]);
    }
    inline float sqr_length() const {
        __m128 sqr = _mm_mul_ps(data, data);
        return sqr[0] + sqr[1] + sqr[2];
    }

    inline float dot(const Vec3f& other) const {
        return _mm_dp_ps(data, other.data, 0xff)[0]; // We should make sure that both [3] = 0.f or this could fail.
    }

    inline Vec3f cross(const Vec3f& other) const {
        return Vec3f(_mm_sub_ps(
            _mm_mul_ps(_mm_shuffle_ps(data, data, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(other.data, other.data, _MM_SHUFFLE(3, 1, 0, 2))),
            _mm_mul_ps(_mm_shuffle_ps(data, data, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(other.data, other.data, _MM_SHUFFLE(3, 0, 2, 1)))
        ));
    }

    inline void normalize() {
        __m128 sqr = _mm_mul_ps(data, data);
        float mag = sqrt(sqr[0] + sqr[1] + sqr[2]);
        data = _mm_div_ps(data, _mm_set_ps1(mag));
    }

    inline Vec3f normalized() const {
        Vec3f v = *this;
        v.normalize();
        return v;
    }

    inline static Vec3f min(const Vec3f &v0, const Vec3f &v1) { return Vec3f(_mm_min_ps(v0.data, v1.data)); }
    inline void min(const Vec3f &other) { data = _mm_min_ps(data, other.data); }
    inline float min() const { return (x < y && x < z) ? x : ((y < z) ? y : z); }

    inline static Vec3f max(const Vec3f &v0, const Vec3f &v1) { return Vec3f(_mm_max_ps(v0.data, v1.data)); }
    inline void max(const Vec3f &other) { data = _mm_max_ps(data, other.data); }
    inline float max() const { return (x > y && x > z) ? x : ((y > z) ? y : z); }

    friend std::ostream& operator<<(std::ostream& os, const Vec3f& v)
    {
        os << "(" << v[0] << " " << v[1] << " " << v[2] << ")";
        return os;
    }
};
