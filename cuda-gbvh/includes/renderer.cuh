#pragma once
#include "triangle.cuh"
#include "object.cuh"
#include "ray.h"
#include "trace.cuh"
#include "scene.cuh"
#include "external/glm/vec3.hpp"
#include "intersection.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>


using glm::vec3;
using std::vector;
using std::string;

struct CameraParameter{
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 origin;
};

__global__ void render_image(vec3* framebuffer, int image_width, int image_height, CameraParameter cam_params, DeviceScene* d_scene, int frame);

void export_to_ppm(string filename, vec3* framebuffers, int image_width, int image_height, int num_frames);
__device__ vec3 raycast(const Ray& ray, const DeviceScene* d_scene, int frame);

void print_frame_buffer(vec3* framebuffer, int image_width, int image_height, int num_frames);
void print_one_frame_buffer(vec3* framebuffer, int image_width, int image_height, int frame);

struct DeviceScene;

__device__ bool find_intersection_bvh(
    const DeviceScene* d_scene,
    const Ray& ray,
    Intersection& itsc,
    float t_min,
    float t_max
);