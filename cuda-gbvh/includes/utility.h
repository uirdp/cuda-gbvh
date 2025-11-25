#pragma once

#include <iostream>
#include "external/glm/vec3.hpp"


static inline void no_memory() { fprintf(stderr, "Not enough memory\n") ; exit(1); }
