#pragma once
#include <string>

#include "../includes/glm/vec3.hpp"

using glm::vec3;

struct Light {
    
    enum LightType {
        DIRECTIONAL,
        POINT
    } type;

    vec3 position;
    vec3 direction;
    vec3 color;
    vec3 attenuation;

    Light(const vec3& _dir, const vec3& _color)
        : type(DIRECTIONAL), direction(_dir), color(_color),
          position(0.0f), attenuation(1.0f, 0.0f, 0.0f) {}

    Light(const vec3& _pos, const vec3& _color, const vec3& _attenuation)
        : type(POINT), position(_pos), color(_color), attenuation(_attenuation) {}

    std::string to_string() const {
        std::string result = "Light Type: ";
        result += (type == DIRECTIONAL) ? "Directional\n" : "Point\n";
        result += "Position: (" + std::to_string(position.x) + ", " + std::to_string(position.y) + ", " + std::to_string(position.z) + ")\n";
        result += "Direction: (" + std::to_string(direction.x) + ", " + std::to_string(direction.y) + ", " + std::to_string(direction.z) + ")\n";
        result += "Color: (" + std::to_string(color.x) + ", " + std::to_string(color.y) + ", " + std::to_string(color.z) + ")\n";
        result += "Attenuation: (" + std::to_string(attenuation.x) + ", " + std::to_string(attenuation.y) + ", " + std::to_string(attenuation.z) + ")\n";
        return result;
    }
};