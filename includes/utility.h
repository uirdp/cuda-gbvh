#pragma once

#include "glm/vec3.hpp"
#include "embree/common/math/vec3fa.h"

inline glm::vec3 vec3fa_to_vec3(const embree::Vec3fa& v) {
    return glm::vec3(v.x, v.y, v.z);
}

inline embree::Vec3fa vec3_to_vec3fa(const glm::vec3& v) {
    return embree::Vec3fa(v.x, v.y, v.z);
}

static void no_memory() { fprintf(stderr, "Not enough memory\n") ; exit(1); }
