#pragma once

#include <cstdio>

#include <koshi/Vec3.h>
#include <koshi/Ray.h>

KOSHI_OPEN_NAMESPACE

// TODO: REMOVE LAST ROW...

class Transform
{
public:
    DEVICE_FUNCTION Transform() 
    : data{ 1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f }
    {
    }

    DEVICE_FUNCTION static Transform fromData(const float * d)
    {
        Transform t;
        t.data[0] = d[0];  t.data[1] = d[1];  t.data[2] = d[2];  t.data[3] = d[3];
        t.data[4] = d[4];  t.data[5] = d[5];  t.data[6] = d[6];  t.data[7] = d[7];
        t.data[8] = d[8];  t.data[9] = d[9];  t.data[10] = d[10];  t.data[11] = d[11];
        return t;
    }

    DEVICE_FUNCTION static Transform fromColumnFirstData(const float * d)
    {
        // Assume size 16 input if columf first.
        Transform t;
        t.data[0] = d[0];  t.data[1] = d[4];  t.data[2] = d[8];  t.data[3] = d[12];
        t.data[4] = d[1];  t.data[5] = d[5];  t.data[6] = d[9];  t.data[7] = d[13];
        t.data[8] = d[2];  t.data[9] = d[6];  t.data[10] = d[10];  t.data[11] = d[14];
        return t;
    }

    DEVICE_FUNCTION static Transform fromData(const double * d)
    {
        Transform t;
        t.data[0] = d[0];  t.data[1] = d[1];  t.data[2] = d[2];  t.data[3] = d[3];
        t.data[4] = d[4];  t.data[5] = d[5];  t.data[6] = d[6];  t.data[7] = d[7];
        t.data[8] = d[8];  t.data[9] = d[9];  t.data[10] = d[10];  t.data[11] = d[11];
        return t;
    }

    DEVICE_FUNCTION static Transform fromColumnFirstData(const double * d)
    {
        // Assume size 16 input if columf first.
        Transform t;
        t.data[0] = d[0];  t.data[1] = d[4];  t.data[2] = d[8];  t.data[3] = d[12];
        t.data[4] = d[1];  t.data[5] = d[5];  t.data[6] = d[9];  t.data[7] = d[13];
        t.data[8] = d[2];  t.data[9] = d[6];  t.data[10] = d[10];  t.data[11] = d[14];
        return t;
    }

    DEVICE_FUNCTION void copy(float * d) const
    {
        d[0] = data[0]; d[1] = data[1]; d[2] = data[2]; d[3] = data[3];
        d[4] = data[4]; d[5] = data[5]; d[6] = data[6]; d[7] = data[7];
        d[8] = data[8]; d[9] = data[9]; d[10] = data[10]; d[11] = data[11];
    }

    DEVICE_FUNCTION Transform inverse() const
    {
        Transform inv;
        inv.data[0] = data[5] * data[10] - data[9] * data[6];
        inv.data[4] = -data[4] * data[10] + data[8] * data[6];
        inv.data[8] = data[4] * data[9] - data[8] * data[5];
        inv.data[1] = -data[1] * data[10] + data[9] * data[2];
        inv.data[5] = data[0] * data[10] - data[8] * data[2];
        inv.data[9] = -data[0] * data[9] + data[8] * data[1];
        inv.data[2] = data[1] * data[6] - data[5] * data[2];
        inv.data[6] = -data[0] * data[6] + data[4] * data[2];
        inv.data[10] = data[0] * data[5] - data[4] * data[1];
        inv.data[3] = -data[1] * data[6] * data[11] + data[1] * data[7] * data[10] + data[5] * data[2] * data[11] - data[5] * data[3] * data[10] - data[9] * data[2] * data[7] + data[9] * data[3] * data[6];
        inv.data[7] = data[0] * data[6] * data[11] - data[0] * data[7] * data[10] - data[4] * data[2] * data[11] + data[4] * data[3] * data[10] + data[8] * data[2] * data[7] - data[8] * data[3] * data[6];
        inv.data[11] = -data[0] * data[5] * data[11] + data[0] * data[7] * data[9] + data[4] * data[1] * data[11] - data[4] * data[3] * data[9] - data[8] * data[1] * data[7] + data[8] * data[3] * data[5];
        float inv_det = 1.f / (data[0] * inv.data[0] + data[1] * inv.data[4] + data[2] * inv.data[8]);
        for (int i = 0; i < 12; i++)
            inv.data[i] = inv.data[i] * inv_det;
        return inv;
    }

    template<bool translate>
    DEVICE_FUNCTION Vec3f multiply(const Vec3f& v) const
    {
        if(translate)
        {
            return Vec3f(
                data[0] * v.x + data[1] * v.y + data[2] * v.z + data[3],
                data[4] * v.x + data[5] * v.y + data[6] * v.z + data[7],
                data[8] * v.x + data[9] * v.y + data[10] * v.z + data[11]
            );
        }
        else
        {
            return Vec3f(
                data[0] * v.x + data[1] * v.y + data[2] * v.z,
                data[4] * v.x + data[5] * v.y + data[6] * v.z,
                data[8] * v.x + data[9] * v.y + data[10] * v.z
            );
        }
    }

    DEVICE_FUNCTION Vec3f operator*(const Vec3f& v) const
    {
        return Vec3f(
            data[0] * v.x + data[1] * v.y + data[2] * v.z + data[3],
            data[4] * v.x + data[5] * v.y + data[6] * v.z + data[7],
            data[8] * v.x + data[9] * v.y + data[10] * v.z + data[11]
        );
    }

    DEVICE_FUNCTION Ray operator*(const Ray& ray) const
    {
        Ray transformed_ray = ray;
        transformed_ray.origin = multiply<true>(ray.origin);
        transformed_ray.direction = multiply<false>(ray.direction);
        return transformed_ray;
    }

    DEVICE_FUNCTION friend Ray& operator*=(Ray& ray, const Transform& transform)
    {
        ray.origin = transform.multiply<true>(ray.origin);
        ray.direction = transform.multiply<false>(ray.direction);
        return ray;
    }

    // inline Box3f operator* (const Box3f& box) const
    // {
    //     Vec3f min = FLT_MAX, max = FLT_MIN;
    //     const Vec3f vs[2] = { box.min(), box.max() };

    //     for(uint x = 0; x < 2; x++)
    //     for(uint y = 0; y < 2; y++)
    //     for(uint z = 0; z < 2; z++)
    //     {
    //         const Vec3f tv = *this * Vec3f(vs[x].x, vs[y].y, vs[z].z);
    //         min.min(tv); max.max(tv);
    //     }

    //     return Box3f(min, max);
    // }

    // inline Transform3f operator* (const Transform3f& v) const
    // {
    //     Transform3f transform;

    //     __m128 cols[4] = {
    //         _mm_setr_ps(v.rows[0][0], v.rows[1][0], v.rows[2][0], v.rows[3][0]),
    //         _mm_setr_ps(v.rows[0][1], v.rows[1][1], v.rows[2][1], v.rows[3][1]),
    //         _mm_setr_ps(v.rows[0][2], v.rows[1][2], v.rows[2][2], v.rows[3][2]),
    //         _mm_setr_ps(v.rows[0][3], v.rows[1][3], v.rows[2][3], v.rows[3][3])
    //     };

    //     transform.rows[0] = _mm_setr_ps(_mm_dp_ps(rows[0], cols[0], 0xff)[0], _mm_dp_ps(rows[0], cols[1], 0xff)[0], _mm_dp_ps(rows[0], cols[2], 0xff)[0], _mm_dp_ps(rows[0], cols[3], 0xff)[0]);
    //     transform.rows[1] = _mm_setr_ps(_mm_dp_ps(rows[1], cols[0], 0xff)[0], _mm_dp_ps(rows[1], cols[1], 0xff)[0], _mm_dp_ps(rows[1], cols[2], 0xff)[0], _mm_dp_ps(rows[1], cols[3], 0xff)[0]);
    //     transform.rows[2] = _mm_setr_ps(_mm_dp_ps(rows[2], cols[0], 0xff)[0], _mm_dp_ps(rows[2], cols[1], 0xff)[0], _mm_dp_ps(rows[2], cols[2], 0xff)[0], _mm_dp_ps(rows[2], cols[3], 0xff)[0]);
    //     transform.rows[3] = _mm_setr_ps(_mm_dp_ps(rows[3], cols[0], 0xff)[0], _mm_dp_ps(rows[3], cols[1], 0xff)[0], _mm_dp_ps(rows[3], cols[2], 0xff)[0], _mm_dp_ps(rows[3], cols[3], 0xff)[0]);

    //     return transform;
    // }

    // static const Transform3f basis_transform(const Vec3f& n)
    // {
    //     Transform3f transform;
    //     const Vec3f nu = (std::fabs(n[0]) > std::fabs(n[1]))
    //     ? Vec3f(n[2], 0, -n[0]) / sqrtf(n[0] * n[0] + n[2] * n[2])
    //     : Vec3f(0, -n[2], n[1]) / sqrtf(n[1] * n[1] + n[2] * n[2]);
    //     const Vec3f nv = nu.cross(n);
    //     transform.rows[0] = _mm_setr_ps(nv[0], n[0], nu[0], 0.f);
    //     transform.rows[1] = _mm_setr_ps(nv[1], n[1], nu[1], 0.f);
    //     transform.rows[2] = _mm_setr_ps(nv[2], n[2], nu[2], 0.f);
    //     transform.rows[3] = _mm_setr_ps(0.f,   0.f,  0.f,   0.f);
    //     return transform;
    // }

    // static const Transform3f translation(const Vec3f& t)
    // {
    //     Transform3f transform;
    //     transform.rows[0][3] = t[0];
    //     transform.rows[1][3] = t[1];
    //     transform.rows[2][3] = t[2];
    //     return transform;
    // }

    // static const Transform3f scale(const Vec3f& s)
    // {
    //     Transform3f transform;
    //     transform.rows[0][0] = s[0];
    //     transform.rows[1][1] = s[1];
    //     transform.rows[2][2] = s[2];
    //     return transform;
    // }

    // static const Transform3f x_rotation(const float r)
    // {
    //     Transform3f transform;
    //     transform.rows[1][1] = cosf(r);
    //     transform.rows[1][2] = sinf(r);
    //     transform.rows[2][1] = -sinf(r);
    //     transform.rows[2][2] = cosf(r);
    //     return transform;
    // }

    // static const Transform3f y_rotation(const float r)
    // {
    //     Transform3f transform;
    //     transform.rows[0][0] = cosf(r);
    //     transform.rows[0][2] = -sinf(r);
    //     transform.rows[2][0] = sinf(r);
    //     transform.rows[2][2] = cosf(r);
    //     return transform;
    // }

    // static const Transform3f z_rotation(const float r)
    // {
    //     Transform3f transform;
    //     transform.rows[0][0] = cosf(r);
    //     transform.rows[0][1] = sinf(r);
    //     transform.rows[1][0] = -sinf(r);
    //     transform.rows[1][1] = cosf(r);
    //     return transform;
    // }

    // static const Transform3f inverse(const Transform3f& v)
    // {
    //     auto get_det = [&](const uint& i, const uint& j)
    //     {
    //         static const uint t[4][3] = { {1,2,3}, {0,2,3}, {0,1,3}, {0,1,2} };
    //         __m128 p =        _mm_setr_ps(v.rows[t[i][0]][t[j][0]], v.rows[t[i][1]][t[j][0]], v.rows[t[i][2]][t[j][0]], 0.f);
    //         p = _mm_mul_ps(p, _mm_setr_ps(v.rows[t[i][1]][t[j][1]], v.rows[t[i][2]][t[j][1]], v.rows[t[i][0]][t[j][1]], 0.f));
    //         p = _mm_mul_ps(p, _mm_setr_ps(v.rows[t[i][2]][t[j][2]], v.rows[t[i][0]][t[j][2]], v.rows[t[i][1]][t[j][2]], 0.f));

    //         __m128 n =        _mm_setr_ps(v.rows[t[i][0]][t[j][2]], v.rows[t[i][0]][t[j][1]], v.rows[t[i][0]][t[j][0]], 0.f);
    //         n = _mm_mul_ps(n, _mm_setr_ps(v.rows[t[i][1]][t[j][1]], v.rows[t[i][1]][t[j][0]], v.rows[t[i][1]][t[j][2]], 0.f));
    //         n = _mm_mul_ps(n, _mm_setr_ps(v.rows[t[i][2]][t[j][0]], v.rows[t[i][2]][t[j][2]], v.rows[t[i][2]][t[j][1]], 0.f));

    //         return p[0] + p[1] + p[2] - n[0] - n[1] - n[2];
    //     };

    //     Transform3f inverse;
    //     inverse.rows[0] = inverse.rows[2] = _mm_setr_ps(1.f, -1.f, 1.f, -1.f);
    //     inverse.rows[1] = inverse.rows[3] = _mm_setr_ps(-1.f, 1.f, -1.f, 1.f);

    //     for (uint y = 0; y < 4; y++)
    //         for (uint x = 0; x < 4; x++)
    //             inverse.rows[y][x] *= get_det(x, y);

    //     float det = v.rows[0][0] * inverse.rows[0][0] + v.rows[0][1] * inverse.rows[1][0]
    //               + v.rows[0][2] * inverse.rows[2][0] + v.rows[0][3] * inverse.rows[3][0];

    //     if (det == 0) return Transform3f();

    //     const __m128 inv_det = _mm_set1_ps(1.f / det);
    //     for(uint i = 0; i < 4; i++)
    //         inverse.rows[i] = _mm_mul_ps(inverse.rows[i], inv_det);

    //     return inverse;
    // }

    // bool is_identity() const
    // {
    //     for(uint y = 0; y < 4; y++)
    //         for(uint x = 0; x < 4; x++)
    //             if(rows[y][x] != (y == x) ? 1.f : 0.f)
    //                 return false;

    //     return true;
    // }

    DEVICE_FUNCTION void print()
    {
        printf("%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n\n", 
        data[0], data[1], data[2], data[3], 
        data[4], data[5], data[6], data[7], 
        data[8], data[9], data[10], data[11]);
    }

private:
    float data[12];

};

KOSHI_CLOSE_NAMESPACE