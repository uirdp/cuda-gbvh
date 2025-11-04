#pragma once

#include "object.h"
#include "embree/common/math/vec3fa.h"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/geometric.hpp"
#include "utility.h"
#include "embree/common/math/math.h"
#include "constants.h"
#include "aabb.h"
#include "ray.h"

#include <utility>
#include <string>

using glm::vec3;
using glm::vec2;
using embree::Vec3fa;

class Triangle : public Object {
    Vec3fa p[3]; //反時計回りに格納
    vec3 n[3]; //正規化されてる
    vec2 t[3];

    bool has_normal : 1;
    bool has_texture_coord : 1;
    float inv_area; // 1/面積*2
    vec3 face_normal; // 面法線、正規化されてる

public:
    Triangle(const vec3& p0, const vec3& p1, const vec3& p2) : has_normal(false), has_texture_coord(false) {
        this->p[0] = vec3_to_vec3fa(p0);
        this->p[1] = vec3_to_vec3fa(p1);
        this->p[2] = vec3_to_vec3fa(p2);

        vec3 edge1 = p1 - p0;
        vec3 edge2 = p2 - p0;
        inv_area = calc_inv_area();
        face_normal = calc_surface_normal();
    }

    vec3 vertex(int idx) const {
        return vec3fa_to_vec3(p[idx]);
    }

    const Vec3fa& vertex_fa(int idx) const {
        return p[idx];
    }

    void change_vertex(int k, vec3 p){
        this->p[k] = vec3_to_vec3fa(p);
        inv_area = calc_inv_area();
        face_normal = calc_surface_normal();
    }

    vec3& normal(int idx) {
        return n[idx];
    }

    const vec3& normal(int idx) const {
        return n[idx];
    }

    bool has_normal() const { return has_normal; }

    void set_normal(vec3 *n0, vec3 *n1, vec3 *n2) {
        n[0] = *n0;
        n[1] = *n1;
        n[2] = *n2;
        has_normal = true;
    }

    void change_normal(int k, vec3* n){
        this->n[k] = *n;
    }

    vec2 texture_coord(int idx) const {
        return t[idx];
    }   

    const vec2& texture_coord_ref(int idx) const {
        return t[idx];
    }

    bool has_texture_coord() const { return has_texture_coord; }

    void set_texture_coord(vec2* t0, vec2* t1, vec2* t2) {
        t[0] = *t0;
        t[1] = *t1;
        t[2] = *t2;
        has_texture_coord = true;
    }

    void change_texture_coord(int k, vec2* t){
        this->t[k] = *t;
    }

    float calc_inv_area() const {
        return 1.0f / glm::length(glm::cross(vertex(1) - vertex(0), vertex(2) - vertex(0)));
    }

    vec3 calc_surface_normal() const {
        return inv_area * glm::cross(vertex(1) - vertex(0), vertex(2) - vertex(0));
    }

    vec3 get_normal_at_intersection(const Ray& ray, float t, const vec2& uv) const {
        if (!has_normal) return face_normal;
        else return normalize((1 - uv.x - uv.y) * normal(0) + uv.x * normal(1) + uv.y * normal(2));
    }

    vec2 get_texture_coord_at_intersection(const Ray& ray, float t, const vec2& uv) const {
        if (!has_texture_coord) return vec2(0.0f, 0.0f);
        else return (1 - uv.x - uv.y) * texture_coord(0) + uv.x * texture_coord(1) + uv.y * texture_coord(2);
    }

    bool get_surface_normal(vec3& normal) const {
        normal = face_normal;
        return true;
    }

    float get_area() const {
        return 0.5f / inv_area;
    }

    AABB get_aabb() const {
        Vec3fa vmin = embree::min(vertex_fa(0), vertex_fa(1), vertex_fa(2));
        Vec3fa vmax = embree::max(vertex_fa(0), vertex_fa(1), vertex_fa(2));
        return AABB(vmin, vmax);
    }

    std::pair<float, float> get_span(int axis) const {
        float min_val = glm::min(glm::min(vertex(0)[axis], vertex(1)[axis]), vertex(2)[axis]);
        float max_val = glm::max(glm::max(vertex(0)[axis], vertex(1)[axis]), vertex(2)[axis]);
        return std::make_pair(min_val, max_val);
    }

    std::pair<vec3, float> get_bounding_sphere() const {
        vec3 a = vertex(1) - vertex(0);
        vec3 b = vertex(2) - vertex(0);
        vec3 c = vertex(2) - vertex(1);
        
        float aa = glm::dot(a, a);
        float bb = glm::dot(b, b);
        float cc = glm::dot(c, c);

        float aa_tmp = aa;
        float bb_tmp = bb;
        float cc_tmp = cc;

        int k = 2;
        if(bb_tmp < cc_tmp) bb_tmp = cc, cc_tmp = bb, k = 1;
        if(aa_tmp < bb_tmp){ aa_tmp = bb_tmp, bb_tmp = aa; }
        else k = 0;

        if(bb_tmp + cc_tmp <= aa_tmp){
            // 鋭角三角形の場合は円の中心は最長辺の中点
            float radius = std::sqrt(aa) * 0.5f;
            vec3 center = (vertex(k) + vertex((k + 1) % 3)) * 0.5f;
            return std::make_pair(center, radius);
        }
        else {
            // 外心
            float radius = sqrt(aa * bb * cc) * inv_area * 0.5f;
            vec3 center = glm::cross(aa * b - bb * a, glm::cross(a, b)) * (inv_area * inv_area * 0.5f) + vertex(0);
            return std::make_pair(center, radius);
        }

    }

    float get_texture_lod(const Ray& ray, float dest) const {
        /* L. Gritz and J. Hahn, "BMRT: A Global Illumination Implementation
	   of the RenderMan Standard" で述べられている簡易的な方法 */
        // 後でちゃんと実装する
        return 0.0f;
    } 

    vec3 bump_mapping(Texture *bump_map, const vec3& normal, const vec2& uv, float mip) const {
        // 後で実装
        return normal;
    }

    std::string to_string() const {
        std::string std;
        char buf[RT_BUFLEN];
        snprintf(buf, RT_BUFLEN, "Triangle(\n  p0: (%.6f, %.6f, %.6f),\n  p1: (%.6f, %.6f, %.6f),\n  p2: (%.6f, %.6f, %.6f),\n  has_normal: %s,\n  has_texture_coord: %s\n)",
                 vertex(0).x, vertex(0).y, vertex(0).z,
                 vertex(1).x, vertex(1).y, vertex(1).z,
                 vertex(2).x, vertex(2).y, vertex(2).z,
                 has_normal ? "true" : "false",
                 has_texture_coord ? "true" : "false");
        if(has_normal){
            snprintf(buf + strlen(buf), RT_BUFLEN - strlen(buf), "  n0: (%.6f, %.6f, %.6f),\n  n1: (%.6f, %.6f, %.6f),\n  n2: (%.6f, %.6f, %.6f),\n",
                     normal(0).x, normal(0).y, normal(0).z,
                     normal(1).x, normal(1).y, normal(1).z,
                     normal(2).x, normal(2).y, normal(2).z);
        }
        if(has_texture_coord){
            snprintf(buf + strlen(buf), RT_BUFLEN - strlen(buf), "  t0: (%.6f, %.6f),\n  t1: (%.6f, %.6f),\n  t2: (%.6f, %.6f),\n",
                     texture_coord(0).x, texture_coord(0).y,
                     texture_coord(1).x, texture_coord(1).y,
                     texture_coord(2).x, texture_coord(2).y);
        }
        snprintf(buf + strlen(buf), RT_BUFLEN - strlen(buf), ")");
        std = buf;
        return std;
    }

};