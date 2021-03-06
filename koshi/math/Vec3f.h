#pragma once

#include <smmintrin.h> // SSE4.1
#include <ostream>
#include <cmath>
#include <iostream>
#include <koshi/math/Vec3.h>

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
    Vec3f(const float& x, const float& y, const float& z, const float& t = 0.f) : data(_mm_setr_ps(x, y, z, t)) {}
    Vec3f(const float& n) : data(_mm_setr_ps(n, n, n, 0.f)) {}
    Vec3f(const __m128& data) : data(data) {}
    Vec3f(const float * n) : data(_mm_setr_ps(n[0], n[1], n[2], 0.f)) {}
    template<typename T>
    Vec3f(const Vec3<T>& n) : data(_mm_setr_ps(n[0], n[1], n[2], 0.f)) {}

    inline float& operator[](const int& i) { return data[i]; }
    inline const float& operator[](const int& i) const { return data[i]; }

    // Assignement
    inline Vec3f& operator= (const Vec3f& other) { data = other.data; return *this; }
    inline Vec3f& operator= (const float& n) { data = _mm_setr_ps(n, n, n, 0.f); return *this; }

    // Casting
    template<typename T>
    inline operator Vec3<T>() { return Vec3<T>(x, y, z); }

    // Negate
    inline Vec3f operator-() const { return Vec3f(_mm_sub_ps(_mm_setzero_ps(), data)); }

    // Addition
    inline Vec3f& operator+= (const Vec3f& other) { data = _mm_add_ps(data, other.data); return *this; }
    inline Vec3f operator+ (const Vec3f& other) const { return Vec3f(_mm_add_ps(data, other.data)); }
    inline Vec3f& operator+= (const float& n) { data = _mm_add_ps(data, _mm_set_ps1(n)); return *this; }
    inline Vec3f operator+ (const float& n) const { return Vec3f(_mm_add_ps(data, _mm_set_ps1(n))); }
    friend inline Vec3f operator+ (const float& n, const Vec3f& other) { return Vec3f(_mm_add_ps(other.data, _mm_set_ps1(n))); }

    // Multiply
    inline Vec3f& operator*= (const Vec3f& other) { data = _mm_mul_ps(data, other.data); return *this; }
    inline Vec3f operator* (const Vec3f& other) const { return Vec3f(_mm_mul_ps(data, other.data)); }
    inline Vec3f& operator*= (const float& n) { data = _mm_mul_ps(data, _mm_set_ps1(n)); return *this; }
    inline Vec3f operator* (const float& n) const { return Vec3f(_mm_mul_ps(data, _mm_set_ps1(n))); }
    friend inline Vec3f operator* (const float& n, const Vec3f& other) { return Vec3f(_mm_mul_ps(other.data, _mm_set_ps1(n))); }
    template<typename T>
    friend inline Vec3<T> operator* (const Vec3<T>& n, const Vec3f& m) { return Vec3<T>(n.x*m.x, n.y*m.y, n.z*m.z); }
    template<typename T>
    friend inline Vec3f operator* (const Vec3f& n, const Vec3<T>& m) { return Vec3f(n.x*m.x, n.y*m.y, n.z*m.z); }

    // Subtract
    inline Vec3f& operator-= (const Vec3f& other) { data = _mm_sub_ps(data, other.data); return *this; }
    inline Vec3f operator- (const Vec3f& other) const { return Vec3f(_mm_sub_ps(data, other.data)); }
    inline Vec3f& operator-= (const float& n) { data = _mm_sub_ps(data, _mm_set_ps1(n)); return *this; }
    inline Vec3f operator- (const float& n) const { return Vec3f(_mm_sub_ps(data, _mm_set_ps1(n))); }
    friend inline Vec3f operator- (const float& n, const Vec3f& other) { return Vec3f(_mm_sub_ps(_mm_set_ps1(n), other.data)); }

    // Divide
    inline Vec3f& operator/= (const Vec3f& other) { data = _mm_div_ps(data, other.data); return *this; }
    inline Vec3f operator/ (const Vec3f& other) const { return Vec3f(_mm_div_ps(data, other.data)); }
    inline Vec3f& operator/= (const float& n) { data = _mm_div_ps(data, _mm_set_ps1(n)); return *this; }
    inline Vec3f operator/ (const float& n) const { return Vec3f(_mm_div_ps(data, _mm_set_ps1(n))); }
    friend inline Vec3f operator/ (const float& n, const Vec3f& other) { return Vec3f(_mm_div_ps(_mm_set_ps1(n), other.data)); }

    // Logic
    inline bool operator== (const Vec3f& v) const { return x == v.x && y == v.y && z == v.z; }
    inline bool operator< (const float& n) const { return x < n && y < n && z < n; }
    inline bool operator<= (const float& n) const { return x <= n && y <= n && z <= n; }
    inline bool operator> (const float& n) const { return x > n && y > n && z > n; }
    inline bool operator>= (const float& n) const { return x >= n && y >= n && z >= n; }

    inline bool null() const { return x == 0.f && y == 0.f && z == 0.f; }
    inline bool operator!() const { return x == 0.f && y == 0.f && z == 0.f; }

    inline float length() const {
        __m128 sqr = _mm_mul_ps(data, data);
        return sqrtf(sqr[0] + sqr[1] + sqr[2]);
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
        float mag = sqrtf(sqr[0] + sqr[1] + sqr[2]);
        data = _mm_div_ps(data, _mm_set_ps1(mag));
    }

    inline Vec3f normalized() const {
        Vec3f v = *this;
        v.normalize();
        return v;
    }

    static inline Vec3f normalize(const Vec3f& v) {
        return v.normalized();
    }

    inline static Vec3f clamp(const Vec3f& _v, const float& min, const float& max) {
        Vec3f v = _v; v.min(max); v.max(min); return v;
    }
    inline void clamp(const float& min, const float& max) {
        this->min(max); this->max(min);
    }

    inline static Vec3f min(const Vec3f& v0, const Vec3f& v1) { return Vec3f(_mm_min_ps(v0.data, v1.data)); }
    inline void min(const Vec3f& other) { data = _mm_min_ps(data, other.data); }
    inline float min() const { return (x < y && x < z) ? x : ((y < z) ? y : z); }

    inline static Vec3f max(const Vec3f& v0, const Vec3f& v1) { return Vec3f(_mm_max_ps(v0.data, v1.data)); }
    inline void max(const Vec3f& other) { data = _mm_max_ps(data, other.data); }
    inline float max() const { return (x > y && x > z) ? x : ((y > z) ? y : z); }

    inline float avg() const { return (x + y + z) * 0.333333333f; }

    inline void abs() { x = fabs(x); y = fabs(y); z = fabs(z); }

    // This is an approximation, switch to actual std::exp and move this into fast math?
    inline static Vec3f exp(const Vec3f& v) {
        Vec3f x(v);
        x = 1.f + x / 256.f;
        x *= x; x *= x; x *= x; x *= x;
        x *= x; x *= x; x *= x; x *= x;
        return x;
    }

    inline static Vec3f log(const Vec3f& v) {
        return Vec3f(logf(v.x), logf(v.y), logf(v.z));
    }

    inline void floor() {
        x = std::floor(x); y = std::floor(y); z = std::floor(z);
    }
    inline static Vec3f floor(const Vec3f& v) {
        return Vec3f(std::floor(v.x), std::floor(v.y), std::floor(v.z));
    }

    inline Vec3f lambda(float (*f)(const float)) {
        return Vec3f(f(x), f(y), f(z));
    }

    friend std::ostream& operator<<(std::ostream& os, const Vec3f& v)
    {
        os << "(" << v[0] << " " << v[1] << " " << v[2] << ")";
        return os;
    }
};
