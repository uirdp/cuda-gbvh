#pragma once

#include "external/glm/vec3.hpp"
#include "external/glm/vec2.hpp"
#include "triangle.cuh"
#include "object.cuh"
#include "ray.h"

#include <cuda_runtime.h>

using glm::vec3;
using glm::vec2;


// Möller–Trumbore algorithm
__device__ inline bool intersect_triangle(const Ray& ray, const Triangle* tri, Intersection& isect, float t_min, float t_max){
    const vec3 e1 = tri->vertex(1) - tri->vertex(0);
    const vec3 e2 = tri->vertex(2) - tri->vertex(0);
    const vec3 pvec = glm::cross(ray.direction, e2);

    const float det = glm::dot(e1, pvec);
    if (fabs(det) < 1e-8f) return false; // 平面と平行

    const float inv_det = 1.0f / det;
    
    const vec3 tvec = ray.origin - tri->vertex(0);
    const float u = glm::dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;

    const vec3 qvec = glm::cross(tvec, e1);
    const float v = glm::dot(ray.direction, qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;
    const float t = glm::dot(e2, qvec) * inv_det;
    if (t < t_min || t > t_max) return false;

    // 交差情報の更新
    isect.t = t;
    isect.obj = (Object*)tri;
    isect.uv = tri->get_texture_coord_at_intersection(ray, t, vec2(u, v));
    return true;
}

__device__ inline bool intersect_sphere(const Ray& ray, const vec3& center, float radius, float& t_hit, float t_min, float t_max) {
    vec3 oc = ray.origin - center;
    float a = glm::dot(ray.direction, ray.direction);
    float b = 2.0f * glm::dot(oc, ray.direction);
    float c = glm::dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
        return false;
    } else {
        float sqrt_discriminant = sqrtf(discriminant);
        float t1 = (-b - sqrt_discriminant) / (2.0f * a);
        float t2 = (-b + sqrt_discriminant) / (2.0f * a);

        if (t1 >= t_min && t1 <= t_max) {
            t_hit = t1;
            return true;
        }
        if (t2 >= t_min && t2 <= t_max) {
            t_hit = t2;
            return true;
        }
        return false;
    }
}