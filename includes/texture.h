#pragma once
#include <string>
#include <vector>
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/base.hpp>

class Texture{
    std::string name;
    std::vector<cv::Mat> images; /* 画像の配列 
                                - CV_8UC3 フォーマット(色はBGRの順) 又は
				  CV_8UC1 フォーマット(バンプマップの場合)
				- 0番要素がオリジナル画像  */
    float max_mip;

private:
    glm::vec2 get_img_coord(const cv::Mat& img, const glm::vec2& uv) const {return glm::vec2(0.0, 0.0);} // 後で実装
public:
    Texture(const std::string& name, const cv::Mat& img) : name(name), max_mip(0.0f) {
        assert(img.type() == CV_8UC3 || img.type() == CV_8UC1);
        images.push_back(img);
    }

    void generate_mipmaps() {} // 後で実装
    
    glm::vec3 get_color(const glm::vec2& uv, float mip_level) const {
        // 後で実装
        return glm::vec3(1.0f, 1.0f, 1.0f);
    }

};