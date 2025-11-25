#include "includes/check_cuda.h"
#include <iostream>
#include <stdio.h>

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line){
    if(result != cudaSuccess){
        std::cerr << "CUDA error = " << cudaGetErrorString(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";

        cudaDeviceReset();
        exit(99);
    }
}