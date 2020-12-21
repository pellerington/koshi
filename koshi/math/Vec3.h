#pragma once

template<typename T>
class Vec3
{
public:
    union {
        struct { T x, y, z; };
        struct { T u, v, w; };
        T data[3];
    };

    Vec3() : data{0, 0, 0} {}
    Vec3(const T& x, const T& y, const T& z) : data{x, y, z}  {}
    Vec3(const T& n) : data{n, n, n}  {}

    inline T& operator[](const int i) { return data[i]; }
    inline const T& operator[](const int i) const { return data[i]; }

    // Assignment
    inline Vec3& operator= (const Vec3& other) { x = other.x; y = other.y; z = other.z; return *this; }

    // Subtract
    inline Vec3& operator-= (const Vec3& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }
    inline Vec3 operator- (const Vec3& other) const { return Vec3(x-other.x, y-other.y, z-other.z); }
    inline Vec3& operator-= (const T& n) { x -= n; y -= n; z -= n; return *this; }
    inline Vec3 operator- (const T& n) const { return Vec3(x-n, y-n, z-n); }
    friend inline Vec3 operator- (const T& n, const Vec3& other) { return Vec3(n-other.x, n-other.y, n-other.z); }

    // Logic
    inline bool operator== (const Vec3<T>& v) const { return x == v.x && y == v.y && z == v.z; }

    friend std::ostream& operator<<(std::ostream& os, const Vec3<T>& v)
    {
        os << "(" << v[0] << " " << v[1] << " " << v[2] << ")";
        return os;
    }

};

typedef Vec3<int> Vec3i;
typedef Vec3<unsigned int> Vec3u;
