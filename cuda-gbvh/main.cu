#include <iostream>
#include <string.h>
#include <cuda_runtime.h>

#include "../includes/glm/vec3.hpp"

#define CHECK_CUDA(val)
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line){
    if(result){
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";

        cudaDeviceReset();
        exit(99);
    }
}

int main(){
    std::string obj_path = "../assets/FairyForest/f000.obj";
    std::string mtl_path = "../assets/FairyForest/";

    int image_width = 800;
    int image_height = 800;

    int block_width = 8;
    int block_height = 8;

    std::cerr << "Rendering " << obj_path << " at " << image_width << "x" << image_height << " resolution." << "\n";
    std::cerr << "in " << block_width << "x" << block_height << " blocks." << "\n";
    
    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(glm::vec3);

    glm::vec3* framebuffer;
    CHECK_CUDA(cudaMallocManaged((void**)&framebuffer, fb_size));
    
    
}