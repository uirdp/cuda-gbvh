#include <string>
#include <vector>
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "light.h"
#pragma once


struct InputParameter {
    std::string obj_file;
    std::string file_name; // パラメタファイルのファイル名
    glm::ivec2 image_size;
    glm::vec3 view_point;
    glm::vec3 look_at;
    glm::vec3 view_up;
    float camera_fov;
    float shadowIntensity;
    glm::vec3 background_color;
    float ambient_intensity;
    std::vector<Light> lights;
    int start_frame;
    int end_frame;
    int build_type;
    int render_repeat;
    float bump_strength;

    InputParameter() :  image_size(512, 512),
                        view_point(0.0f, 0.0f, 5.0f),
                        look_at(0.0f, 0.0f, 0.0f),
                        view_up(0.0f, 1.0f, 0.0f),
                        camera_fov(45.0f),
                        shadowIntensity(0.5f),
                        background_color(0.0f, 0.0f, 0.0f),
                        ambient_intensity(0.1f),
                        start_frame(0),
                        end_frame(0),
                        build_type(0),
                        render_repeat(1),
                        bump_strength(1.0f) {}

    std::string get_obj_file_name() const {
        return obj_file;
    }
};



