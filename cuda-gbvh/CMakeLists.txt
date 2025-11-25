cmake_minimum_required(VERSION 3.16)
project(cuda-gbvh LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_CUDA_ARCHTARGETS "60;61;70;75;80;86;89;90")

find_package(OpenCV REQUIRED)

add_executable(cuda_gbvh
    trace.cu
    renderer.cu
    fileio.cpp
    scene.cu
    check_cuda.cpp
    main.cu
)           

target_include_directories(cuda_gbvh PRIVATE
    ${CMAKE_SOURCE_DIR}/includes
    /usr/local/cuda/include
)

target_include_directories(cuda_gbvh
    SYSTEM PRIVATE
        ${CMAKE_SOURCE_DIR}/includes/external
)

target_link_libraries(cuda_gbvh PRIVATE
    ${OpenCV_LIBS}
)

target_compile_options(cuda_gbvh
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
