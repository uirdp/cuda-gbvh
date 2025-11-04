#pragma once

#include "glm/vec3.hpp"
#include <string>
#include "constants.h"

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;

    float tlength;
    float spread;

    Ray(const glm::vec3& o, const glm::vec3& d, float t, float s) : origin(o), direction(d), tlength(t), spread(s) {}

    std::string to_string() const {
        char buf[RT_BUFLEN];
        snprintf(buf, RT_BUFLEN, "Ray(origin: (%.6f, %.6f, %.6f), direction: (%.6f, %.6f, %.6f), tlength: %.6f, spread: %.6f)",
                 origin.x, origin.y, origin.z, direction.x, direction.y, direction.z, tlength, spread);
        return buf;
    }

};