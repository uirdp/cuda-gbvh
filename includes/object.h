#pragma once

#include "material.h"
#include "ray.h"
#include "aabb.h"
#include <limits.h>
#include <string>
#include <utility>

typedef int CodeType; // 後で適切な場所で定義してください
class Object {
    Material* material;
public:
    CodeType code; // Grid Code
    int expire; // 消滅は後々扱うかも、1以上だとそのフレームで消滅
#if DO_REFIT
#endif

public:
    Object() : material(&Material::defaultMaterial), expire(INT_MAX) {}

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

#include "Triangle.h"

inline glm::vec3 Object::get_normal_at_intersection(const Ray& ray,
					    float t, const glm::vec2& uv) const {
    return ((Triangle*)this)->get_normal_at_intersection(ray, t, uv);
}
inline glm::vec2 Object::get_texture_coord_at_intersection(const Ray& ray,
					      float t, const vec2& uv) const {
    return ((Triangle*)this)->get_texture_coord_at_intersection(ray, t, uv);
}
inline bool Object::get_surface_normal(glm::vec3& normal) const {
    return ((Triangle*)this)->get_surface_normal(normal);
}
inline AABB Object::get_aabb() const {
    return ((Triangle*)this)->get_aabb();
}
inline std::pair<float, float> Object::get_span(int axis) const {
    return ((Triangle*)this)->get_span(axis);
}
inline std::pair<vec3, float> Object::get_bounding_sphere() const {
    return ((Triangle*)this)->get_bounding_sphere();
}
inline float Object::get_texture_lod(const Ray& ray, float dest) const {
    return ((Triangle*)this)->get_texture_lod(ray, dest);
}
inline vec3 Object::bump_mapping(Texture *bumpMap,
				  const vec3& normal,
				  const vec2& texcoord,
				  float mipmapLevel) const {
    return ((Triangle*)this)->bump_mapping(bumpMap, normal,
					    texcoord, mipmapLevel);
}
inline std::string Object::to_string() const {
    return ((Triangle*)this)->to_string();
}

struct Intersection {
    Object *obj;
    float t;
    glm::vec2 uv;

    Intersection(float _t) : obj(nullptr), t(_t), uv(0.0f, 0.0f) {}
};



