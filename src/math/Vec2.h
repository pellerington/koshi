#pragma once

template<typename T>
class Vec2
{
public:
    union {
        struct { T x, y; };
        struct { T u, v; };
        T data[2];
    };

    Vec2() : data{0, 0} {}
    Vec2(const T& x, const T& y) : data{x, y}  {}
    Vec2(const T& n) : data{n, n}  {}

    inline T& operator[](const int i) { return data[i]; }
    inline const T& operator[](const int i) const { return data[i]; }

    // Assignement
    inline Vec2& operator= (const Vec2& other) { x = other.x; y = other.y; return *this; }
    inline Vec2& operator= (const T& n) { x = n; y = n; return *this; }

    // Negate
    inline Vec2 operator-() const { return Vec2(-x, -y); }

    // Addition
    inline Vec2& operator+= (const Vec2& other) { x += other.x; y += other.y; return *this; }
    inline Vec2 operator+ (const Vec2& other) const { return Vec2(x+other.x, y+other.y); }
    inline Vec2& operator+= (const T& n) { x += n; y += n; return *this; }
    inline Vec2 operator+ (const T& n) const { return Vec2(x+n, y+n); }
    friend inline Vec2 operator+ (const T& n, const Vec2& other) { return Vec2(n+other.x, n+other.y); }

    // Multiply
    inline Vec2& operator*= (const Vec2& other) { x *= other.x; y *= other.y; return *this; }
    inline Vec2 operator* (const Vec2& other) const { return Vec2(x*other.x, y*other.y); }
    inline Vec2& operator*= (const T& n) { x *= n; y *= n; return *this; }
    inline Vec2 operator* (const T& n) const { return Vec2(x*n, y*n); }
    friend inline Vec2 operator* (const T& n, const Vec2& other) { return Vec2(n*other.x, n*other.y); }

    // Subtract
    inline Vec2& operator-= (const Vec2& other) { x -= other.x; y -= other.y; return *this; }
    inline Vec2 operator- (const Vec2& other) const { return Vec2(x-other.x, y-other.y); }
    inline Vec2& operator-= (const T& n) { x -= n; y -= n; return *this; }
    inline Vec2 operator- (const T& n) const { return Vec2(x-n, y-n); }
    friend inline Vec2 operator- (const T& n, const Vec2& other) { return Vec2(n-other.x, n-other.y); }

    // Divide
    inline Vec2& operator/= (const Vec2& other) { x /= other.x; y /= other.y; return *this; }
    inline Vec2 operator/ (const Vec2& other) const { return Vec2(x/other.x, y/other.y); }
    inline Vec2& operator/= (const T& n) { x /= n; y /= n; return *this; }
    inline Vec2 operator/ (const T& n) const { return Vec2(x/n, y/n); }
    friend inline Vec2 operator/ (const T& n, const Vec2& other) { return Vec2(n/other.x, n/other.y); }

    // Max
    inline T max() const { return (x > y) ? x : y; }

    friend std::ostream& operator<<(std::ostream& os, const Vec2& v)
    {
        os << "(" << v.x << " " << v.y << ")";
        return os;
    }
};

typedef Vec2<float> Vec2f;
typedef Vec2<int> Vec2i;
typedef Vec2<uint> Vec2u;
