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
    float refractionIndex;
    Texture *ambientMap;
    Texture *diffuseMap;
    Texture *specularMap;
    Texture *bumpMap;
    float bumpScale;

    static Material defaultMaterial;

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
        refractionIndex(1.0f), ambientMap(nullptr), diffuseMap(nullptr), specularMap(nullptr),
        bumpMap(nullptr), bumpScale(1.0f) {}

    bool possible_caustics() const {
        return transparency != 0 || reflectance != 0;
    }
};