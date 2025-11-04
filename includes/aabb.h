#pragma once

#include "ray.h"
#include "embree/common/math/vec3fa.h"
#include "constants.h"

#include <string>


struct AABB {
    embree::Vec3fa vmin;
    embree::Vec3fa vmax;

    AABB() {
        vmin = embree::Vec3fa( std::numeric_limits<float>::infinity());
        vmax = embree::Vec3fa(-std::numeric_limits<float>::infinity());
    }

    AABB(const embree::Vec3fa& min, const embree::Vec3fa& max) : vmin(min), vmax(max) {}
    AABB(const glm::vec3& min, const glm::vec3& max)
        : vmin(embree::Vec3fa(min.x, min.y, min.z)), vmax(embree::Vec3fa(max.x, max.y, max.z)) {}

    // ここら辺はcudaでも実装しないとだめ
    // kernel_compute_surface();

    float surface_area() const {
    #ifdef CPU_PARALLEL
        if(embree::any(embree::lt_mask(vmax, vmin))) return 0.0f;
        embree::Vec3fa d = vmax - vmin;
        return embree::halfArea(d);
    #else 
        if (vmax.x < vmin.x || vmax.y < vmin.y || vmax.z < vmin.z) return 0.0f;
        embree::Vec3fa d = vmax - vmin;
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    #endif
    }

    float length() const {
    #ifdef CPU_PARALLEL
        return reduce_max(vmax - vmin);
    #else
        embree::Vec3fa d = vmax - vmin;
        return std::max({d.x, d.y, d.z});
    #endif
    }

    void insert(const embree::Vec3fa& point){
        vmin = embree::min(vmin, point);
        vmax = embree::max(vmax, point);
    }

    void insert(const glm::vec3& point){
        embree::Vec3fa p(point.x, point.y, point.z);
        vmin = embree::min(vmin, p);
        vmax = embree::max(vmax, p);
    }

    void insert(const AABB & aabb){
        vmin = embree::min(vmin, aabb.vmin);
        vmax = embree::max(vmax, aabb.vmax);
    }

    /* 誤差余裕をもった少し大きな AABB を返す */
    AABB dilate() const {
        float d = this->length() * EPSILON_AABB;
        return AABB(vmin - embree::Vec3fa(d), vmax + embree::Vec3fa(d));
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