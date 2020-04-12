#pragma once

#include "Vec3f.h"
#include "Box3f.h"

class Transform3f
{
public:
    Transform3f() : rows{_mm_setr_ps(1.f, 0.f, 0.f, 0.f),
                         _mm_setr_ps(0.f, 1.f, 0.f, 0.f),
                         _mm_setr_ps(0.f, 0.f, 1.f, 0.f),
                         _mm_setr_ps(0.f, 0.f, 0.f, 1.f)} {}

    inline Vec3f operator* (const Vec3f &v) const
    {
        __m128 vt = v.data; vt[3] = 1.f;
        return Vec3f(_mm_dp_ps(rows[0], vt, 0xff)[0], _mm_dp_ps(rows[1], vt, 0xff)[0], _mm_dp_ps(rows[2], vt, 0xff)[0]);
    }

    inline Vec3f multiply(const Vec3f &v, const bool translate = true) const
    {
        __m128 vt = v.data; vt[3] = (translate) ? 1.f : 0.f;
        return Vec3f(_mm_dp_ps(rows[0], vt, 0xff)[0], _mm_dp_ps(rows[1], vt, 0xff)[0], _mm_dp_ps(rows[2], vt, 0xff)[0]);
    }

    inline Box3f operator* (const Box3f &box) const
    {
        Vec3f min = FLT_MAX, max = FLT_MIN;
        const Vec3f vs[2] = { box.min(), box.max() };

        for(uint x = 0; x < 2; x++)
        for(uint y = 0; y < 2; y++)
        for(uint z = 0; z < 2; z++)
        {
            const Vec3f tv = *this * Vec3f(vs[x].x, vs[y].y, vs[z].z);
            min.min(tv); max.max(tv);
        }

        return Box3f(min, max);
    }

    inline Transform3f operator* (const Transform3f &v) const
    {
        Transform3f transform;

        __m128 cols[4] = {
            _mm_setr_ps(v.rows[0][0], v.rows[1][0], v.rows[2][0], v.rows[3][0]),
            _mm_setr_ps(v.rows[0][1], v.rows[1][1], v.rows[2][1], v.rows[3][1]),
            _mm_setr_ps(v.rows[0][2], v.rows[1][2], v.rows[2][2], v.rows[3][2]),
            _mm_setr_ps(v.rows[0][3], v.rows[1][3], v.rows[2][3], v.rows[3][3])
        };

        transform.rows[0] = _mm_setr_ps(_mm_dp_ps(rows[0], cols[0], 0xff)[0], _mm_dp_ps(rows[0], cols[1], 0xff)[0], _mm_dp_ps(rows[0], cols[2], 0xff)[0], _mm_dp_ps(rows[0], cols[3], 0xff)[0]);
        transform.rows[1] = _mm_setr_ps(_mm_dp_ps(rows[1], cols[0], 0xff)[0], _mm_dp_ps(rows[1], cols[1], 0xff)[0], _mm_dp_ps(rows[1], cols[2], 0xff)[0], _mm_dp_ps(rows[1], cols[3], 0xff)[0]);
        transform.rows[2] = _mm_setr_ps(_mm_dp_ps(rows[2], cols[0], 0xff)[0], _mm_dp_ps(rows[2], cols[1], 0xff)[0], _mm_dp_ps(rows[2], cols[2], 0xff)[0], _mm_dp_ps(rows[2], cols[3], 0xff)[0]);
        transform.rows[3] = _mm_setr_ps(_mm_dp_ps(rows[3], cols[0], 0xff)[0], _mm_dp_ps(rows[3], cols[1], 0xff)[0], _mm_dp_ps(rows[3], cols[2], 0xff)[0], _mm_dp_ps(rows[3], cols[3], 0xff)[0]);

        return transform;
    }

    static const Transform3f basis_transform(const Vec3f &n)
    {
        Transform3f transform;
        const Vec3f nu = (std::fabs(n[0]) > std::fabs(n[1]))
        ? Vec3f(n[2], 0, -n[0]) / sqrtf(n[0] * n[0] + n[2] * n[2])
        : Vec3f(0, -n[2], n[1]) / sqrtf(n[1] * n[1] + n[2] * n[2]);
        const Vec3f nv = nu.cross(n);
        transform.rows[0] = _mm_setr_ps(nv[0], n[0], nu[0], 0.f);
        transform.rows[1] = _mm_setr_ps(nv[1], n[1], nu[1], 0.f);
        transform.rows[2] = _mm_setr_ps(nv[2], n[2], nu[2], 0.f);
        transform.rows[3] = _mm_setr_ps(0.f,   0.f,  0.f,   0.f);
        return transform;
    }

    static const Transform3f translation(const Vec3f &t)
    {
        Transform3f transform;
        transform.rows[0][3] = t[0];
        transform.rows[1][3] = t[1];
        transform.rows[2][3] = t[2];
        return transform;
    }

    static const Transform3f scale(const Vec3f &s)
    {
        Transform3f transform;
        transform.rows[0][0] = s[0];
        transform.rows[1][1] = s[1];
        transform.rows[2][2] = s[2];
        return transform;
    }

    static const Transform3f x_rotation(const float r)
    {
        Transform3f transform;
        transform.rows[1][1] = cosf(r);
        transform.rows[1][2] = sinf(r);
        transform.rows[2][1] = -sinf(r);
        transform.rows[2][2] = cosf(r);
        return transform;
    }

    static const Transform3f y_rotation(const float r)
    {
        Transform3f transform;
        transform.rows[0][0] = cosf(r);
        transform.rows[0][2] = -sinf(r);
        transform.rows[2][0] = sinf(r);
        transform.rows[2][2] = cosf(r);
        return transform;
    }

    static const Transform3f z_rotation(const float r)
    {
        Transform3f transform;
        transform.rows[0][0] = cosf(r);
        transform.rows[0][1] = sinf(r);
        transform.rows[1][0] = -sinf(r);
        transform.rows[1][1] = cosf(r);
        return transform;
    }

    static const Transform3f inverse(const Transform3f &v)
    {
        auto get_det = [&](const uint &i, const uint &j)
        {
            static const uint t[4][3] = { {1,2,3}, {0,2,3}, {0,1,3}, {0,1,2} };
            __m128 p =        _mm_setr_ps(v.rows[t[i][0]][t[j][0]], v.rows[t[i][1]][t[j][0]], v.rows[t[i][2]][t[j][0]], 0.f);
            p = _mm_mul_ps(p, _mm_setr_ps(v.rows[t[i][1]][t[j][1]], v.rows[t[i][2]][t[j][1]], v.rows[t[i][0]][t[j][1]], 0.f));
            p = _mm_mul_ps(p, _mm_setr_ps(v.rows[t[i][2]][t[j][2]], v.rows[t[i][0]][t[j][2]], v.rows[t[i][1]][t[j][2]], 0.f));

            __m128 n =        _mm_setr_ps(v.rows[t[i][0]][t[j][2]], v.rows[t[i][0]][t[j][1]], v.rows[t[i][0]][t[j][0]], 0.f);
            n = _mm_mul_ps(n, _mm_setr_ps(v.rows[t[i][1]][t[j][1]], v.rows[t[i][1]][t[j][0]], v.rows[t[i][1]][t[j][2]], 0.f));
            n = _mm_mul_ps(n, _mm_setr_ps(v.rows[t[i][2]][t[j][0]], v.rows[t[i][2]][t[j][2]], v.rows[t[i][2]][t[j][1]], 0.f));

            return p[0] + p[1] + p[2] - n[0] - n[1] - n[2];
        };

        Transform3f inverse;
        inverse.rows[0] = inverse.rows[2] = _mm_setr_ps(1.f, -1.f, 1.f, -1.f);
        inverse.rows[1] = inverse.rows[3] = _mm_setr_ps(-1.f, 1.f, -1.f, 1.f);

        for (uint y = 0; y < 4; y++)
            for (uint x = 0; x < 4; x++)
                inverse.rows[y][x] *= get_det(x, y);

        float det = v.rows[0][0] * inverse.rows[0][0] + v.rows[0][1] * inverse.rows[1][0]
                  + v.rows[0][2] * inverse.rows[2][0] + v.rows[0][3] * inverse.rows[3][0];

        if (det == 0) return Transform3f();

        const __m128 inv_det = _mm_set1_ps(1.f / det);
        for(uint i = 0; i < 4; i++)
            inverse.rows[i] = _mm_mul_ps(inverse.rows[i], inv_det);

        return inverse;
    }

    bool is_identity() const
    {
        for(uint y = 0; y < 4; y++)
            for(uint x = 0; x < 4; x++)
                if(rows[y][x] != (y == x) ? 1.f : 0.f)
                    return false;

        return true;
    }

    friend std::ostream& operator<<(std::ostream& os, const Transform3f& t)
    {
        os << "{" << t.rows[0][0] << " " << t.rows[0][1] << " " << t.rows[0][2] << " " << t.rows[0][3] << "\n";
        os << " " << t.rows[1][0] << " " << t.rows[1][1] << " " << t.rows[1][2] << " " << t.rows[1][3] << "\n";
        os << " " << t.rows[2][0] << " " << t.rows[2][1] << " " << t.rows[2][2] << " " << t.rows[2][3] << "\n";
        os << " " << t.rows[3][0] << " " << t.rows[3][1] << " " << t.rows[3][2] << " " << t.rows[3][3] << "}\n";
        return os;
    }

protected:
    __m128 rows[4];
};
