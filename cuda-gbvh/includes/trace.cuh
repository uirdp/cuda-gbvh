#pragma once

#include "ray.h"
#include "object.cuh"
#include "triangle.cuh"
#include "intersection.cuh"
#include "external/glm/vec3.hpp"
#include "scene.cuh"

#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>

using glm::vec3;

__device__ vec3 raycast(const Ray& ray, const DeviceScene* d_scene, int num_tris, int frame);