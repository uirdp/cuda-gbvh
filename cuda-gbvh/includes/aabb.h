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

    AABB() {
        vmin = vec3( std::numeric_limits<float>::infinity());
        vmax = vec3(-std::numeric_limits<float>::infinity());
    }

    AABB(const vec3& min, const vec3& max) : vmin(min), vmax(max) {}

    // ここら辺はcudaでも実装しないとだめ
    // kernel_compute_surface();

    float surface_area() const {
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

    float length() const {
    #ifdef CPU_PARALLEL
        return reduce_max(vmax - vmin);
    #else
        vec3 d = vmax - vmin;
        return std::max(d.x, std::max(d.y, d.z));
    #endif
    }

    void insert(const vec3& point){
        vmin = glm::min(vmin, point);
        vmax = glm::max(vmax, point);
    }

    void insert(const AABB & aabb){
        vmin = glm::min(vmin, aabb.vmin);
        vmax = glm::max(vmax, aabb.vmax);
    }

    /* 誤差余裕をもった少し大きな AABB を返す */
    AABB dilate() const {
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
};