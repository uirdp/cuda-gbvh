#pragma once

#include "ray.h"
// #include "embree/common/math/vec3fa.h"
#include "constants.h"
#include "external/glm/vec3.hpp"
#include "external/glm/glm.hpp"

#include <string>

using glm::vec3;
struct AABB {
    vec3 vmin;
    vec3 vmax;

    __host__ __device__
    AABB() {
        vmin = vec3( std::numeric_limits<float>::infinity());
        vmax = vec3(-std::numeric_limits<float>::infinity());
    }

    __host__ __device__
    AABB(const vec3& min, const vec3& max) : vmin(min), vmax(max) {}

    // ここら辺はcudaでも実装しないとだめ
    // kernel_compute_surface();

    __host__ __device__ float surface_area() const {
#ifdef CPU_PARALLEL
        if(embree::any(embree::lt_mask(vmax, vmin))) return 0.0f;
        embree::Vec3fa d = vmax - vmin;
        return embree::halfArea(d);
#else 
        if (vmax.x < vmin.x || vmax.y < vmin.y || vmax.z < vmin.z) return 0.0f;
        vec3 d = vmax - vmin;
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
#endif
    }

    __host__ __device__ float length() const {
    #ifdef __CUDA_ARCH__
        vec3 d = vmax - vmin;
        return fmaxf(d.x, fmaxf(d.y, d.z));
    #else
        #ifdef CPU_PARALLEL
            return reduce_max(vmax - vmin);
        #else
            vec3 d = vmax - vmin;
            return std::max(d.x, std::max(d.y, d.z));
        #endif
    #endif
    }

    __host__ __device__
    void insert(const vec3& point){
#ifdef __CUDA_ARCH__
        // device
        vmin.x = fminf(vmin.x, point.x);
        vmin.y = fminf(vmin.y, point.y);
        vmin.z = fminf(vmin.z, point.z);

        vmax.x = fmaxf(vmax.x, point.x);
        vmax.y = fmaxf(vmax.y, point.y);
        vmax.z = fmaxf(vmax.z, point.z);
#else
        // host
        vmin = glm::min(vmin, point);
        vmax = glm::max(vmax, point);
#endif
    }

    __host__ __device__
    void insert(const AABB& aabb){
#ifdef __CUDA_ARCH__
        // device
        vmin.x = fminf(vmin.x, aabb.vmin.x);
        vmin.y = fminf(vmin.y, aabb.vmin.y);
        vmin.z = fminf(vmin.z, aabb.vmin.z);

        vmax.x = fmaxf(vmax.x, aabb.vmax.x);
        vmax.y = fmaxf(vmax.y, aabb.vmax.y);
        vmax.z = fmaxf(vmax.z, aabb.vmax.z);
#else
        // host
        vmin = glm::min(vmin, aabb.vmin);
        vmax = glm::max(vmax, aabb.vmax);
#endif
    }

    /* 誤差余裕をもった少し大きな AABB を返す */
    __host__ __device__ AABB dilate() const {
        float d = this->length() * EPSILON_AABB;
        return AABB(vmin - vec3(d), vmax + vec3(d));
    }

    /* 各頂点の座標を返す */
    glm::vec3 vertex(int idx) const {
        return glm::vec3(
            (idx & 1) ? vmax.x : vmin.x,
            (idx & 2) ? vmax.y : vmin.y,
            (idx & 4) ? vmax.z : vmin.z
        );
    }

    std::string to_string() const {
        char buf[RT_BUFLEN];
        snprintf(buf, RT_BUFLEN, "AABB(vmin: (%.6f, %.6f, %.6f), vmax: (%.6f, %.6f, %.6f))",
                 vmin.x, vmin.y, vmin.z, vmax.x, vmax.y, vmax.z);
        return buf;
    }

    __host__ __device__ static AABB merge(const AABB a, const AABB b) {
        AABB m;
        m.vmin.x = fminf(a.vmin.x, b.vmin.x);
        m.vmin.y = fminf(a.vmin.y, b.vmin.y);
        m.vmin.z = fminf(a.vmin.z, b.vmin.z);
        m.vmax.x = fmaxf(a.vmax.x, b.vmax.x);
        m.vmax.y = fmaxf(a.vmax.y, b.vmax.y);
        m.vmax.z = fmaxf(a.vmax.z, b.vmax.z);
        return m;
    }
};