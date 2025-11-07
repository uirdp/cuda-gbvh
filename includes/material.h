#pragma once

#include "texture.h"
#include <string>

#include "glm/vec3.hpp"

struct Material {
    std::string name;
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float exponent;
    float reflectance;
    float transparency;
    float refraction_index;
    Texture *ambient_map;
    Texture *diffuse_map;
    Texture *specular_map;
    Texture *bump_map;
    float bump_scale;

    static Material default_material;

    // static Material defaultMaterial() {
    //     Material mat;
    //     mat.name = "default";
    //     mat.ambient = glm::vec3(0.1f, 0.1f, 0.1f);
    //     mat.diffuse = glm::vec3(0.8f, 0.8f, 0.8f);
    //     mat.specular = glm::vec3(1.0f, 1.0f, 1.0f);
    //     mat.exponent = 32.0f;
    //     mat.reflectance = 0.0f;
    //     mat.transparency = 0.0f;
    //     mat.refractionIndex = 1.0f;
    //     mat.ambientMap = nullptr;
    //     mat.diffuseMap = nullptr;
    //     mat.specularMap = nullptr;
    //     mat.bumpMap = nullptr;
    //     mat.bumpScale = 1.0f;
    //     return mat;
    // }

    Material(const std::string& _name) : name(_name), ambient(0.1f, 0.1f, 0.1f), diffuse(0.8f, 0.8f, 0.8f),
        specular(1.0f, 1.0f, 1.0f), exponent(32.0f), reflectance(0.0f), transparency(0.0f),
        refraction_index(1.0f), ambient_map(nullptr), diffuse_map(nullptr), specular_map(nullptr),
        bump_map(nullptr), bump_scale(1.0f) {}

    bool possible_caustics() const {
        return transparency != 0 || reflectance != 0;
    }
};