#pragma once

#include "material.h"
#include "ray.h"
#include "aabb.h"
#include <limits.h>
#include <string>
#include <utility>

using glm::vec2;
using glm::vec3;

typedef int CodeType; // 後で適切な場所で定義してください
class Object {
    Material* material;
public:
    CodeType code; // Grid Code
    int expire; // 消滅は後々扱うかも、1以上だとそのフレームで消滅
    int id = -1;
#if DO_REFIT
#endif

public:
    Object() : material(&Material::default_material), expire(INT_MAX) {}

    Material* get_material() const { return material;}
    Object* set_material(Material *m) {
        material = m;
        return this;
    }

    glm::vec3 get_normal_at_intersection(const Ray& ray, float t, const glm::vec2& uv) const;
    glm::vec2 get_texture_coord_at_intersection(const Ray& ray, float t, const glm::vec2& uv) const;
    bool get_surface_normal(glm::vec3& normal) const;

    AABB get_aabb() const;
    std::pair<float, float> get_span(int axis) const;
    std::pair<glm::vec3, float> get_bounding_sphere() const;

    float get_texture_lod(const Ray& ray, float dest) const;

    glm::vec3 bump_mapping(Texture* bumpMap, const glm::vec3& normal, const glm::vec2& uv, float mipmapLevel) const;

    std::string to_string() const;

};

struct Intersection {
    Object *obj;
    float t;
    glm::vec2 uv;

    __host__ __device__ Intersection(float _t) : obj(nullptr), t(_t), uv(0.0f, 0.0f) {}
    __host__ __device__ Intersection() = default;
};

