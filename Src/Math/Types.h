#pragma once

#include <iostream>
#include <Eigen/Core>

#define HALF_PI 1.57079632679
#define PI 3.14159265359
#define TWO_PI 6.28318530718

#define INV_PI 0.31830988618
#define INV_TWO_PI 0.15915494309

#define EPSILON 0.0000001f

typedef unsigned int uint;

template<typename T>
class Vec3 : public Eigen::Matrix<T, 3, 1>
{
public:
    Vec3() : Eigen::Matrix<T, 3, 1>(0, 0, 0) {}
    Vec3(const T &x, const T &y, const T &z) : Eigen::Matrix<T, 3, 1>(x, y, z) {}
    Vec3(const T &n) : Eigen::Matrix<T, 3, 1>(n, n, n)  {}

    inline T& x() { return (*this)[0]; }
    inline T& y() { return (*this)[1]; }
    inline T& z() { return (*this)[2]; }

    inline T& r() { return (*this)[0]; }
    inline T& g() { return (*this)[1]; }
    inline T& b() { return (*this)[2]; }

    inline const T &x() const { return (*this)[0]; }
    inline const T &y() const { return (*this)[1]; }
    inline const T &z() const { return (*this)[2]; }

    inline const T &r() const { return (*this)[0]; }
    inline const T &g() const { return (*this)[1]; }
    inline const T &b() const { return (*this)[2]; }

    inline Vec3& operator= (const Vec3& other) { (*this)[0] = other[0]; (*this)[1] = other[1]; (*this)[2] = other[2]; return *this; }
    inline Vec3& operator= (const T &n) { (*this)[0] = n; (*this)[1] = n; (*this)[2] = n; return *this; }

    inline Vec3 operator* (const Vec3& other) { return Vec3((*this)[0] * other[0], (*this)[1] * other[1], (*this)[2] * other[2]); }

    inline T length() { return this->norm(); }
    inline T sqr_length() { return this->squaredNorm(); }

    inline Vec3<T> cross(const Vec3& other) const { return Eigen::Matrix<T, 3, 1>::cross(other); }

    static Vec3 Zero() { return Vec3(0, 0, 0); }

    //Default Eigen
    template<typename OtherDerived>
    Vec3(const Eigen::MatrixBase<OtherDerived>& other) : Eigen::Matrix<T, 3, 1>(other)  {}
    template<typename OtherDerived>
    inline Vec3& operator= (const Eigen::MatrixBase <OtherDerived>& other)
    {
        this->Eigen::Matrix<T, 3, 1>::operator=(other);
        return *this;
    }
};

typedef Vec3<float> Vec3f;

template<typename T>
class Vec2 : public Eigen::Matrix<T, 2, 1>
{
public:
    Vec2() : Eigen::Matrix<T, 2, 1>(0, 0) {}
    Vec2(const T &x, const T &y) : Eigen::Matrix<T, 2, 1>(x, y) {}
    Vec2(const T &n) : Eigen::Matrix<T, 2, 1>(n, n)  {}

    inline T &x() { return (*this)[0]; }
    inline T &y() { return (*this)[1]; }

    inline T &u() { return (*this)[0]; }
    inline T &v() { return (*this)[1]; }

    inline const T &x() const { return (*this)[0]; }
    inline const T &y() const { return (*this)[1]; }

    inline const T &u() const { return (*this)[0]; }
    inline const T &v() const { return (*this)[1]; }

    inline Vec2& operator= (const Vec2& other) { (*this)[0] = other[0]; (*this)[1] = other[1]; return *this; }
    inline Vec2& operator= (const T &n) { (*this)[0] = n; (*this)[1] = n; return *this; }

    inline Vec2 operator* (const Vec2& other) { return Vec2((*this)[0] * other[0], (*this)[1] * other[1]); }

    inline T length() { return this->norm(); }
    inline T sqr_length() { return this->squaredNorm(); }

    inline Vec2<T> cross(const Vec2& other) const { return Eigen::Matrix<T, 2, 1>::cross(other); }

    static Vec2 Zero() { return Vec2(0, 0); }

    //Default Eigen
    template<typename OtherDerived>
    Vec2(const Eigen::MatrixBase<OtherDerived>& other) : Eigen::Matrix<T, 2, 1>(other)  {}
    template<typename OtherDerived>
    inline Vec2& operator= (const Eigen::MatrixBase <OtherDerived>& other)
    {
        this->Eigen::Matrix<T, 2, 1>::operator=(other);
        return *this;
    }
};

typedef Vec2<float> Vec2f;
typedef Vec2<int> Vec2i;
