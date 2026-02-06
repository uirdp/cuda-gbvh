#include <iostream>
#include <string.h>
#include <cuda_runtime.h>

#include "includes/external/glm/vec3.hpp"
#include "includes/external/glm/vec2.hpp"
#include "includes/scene.cuh"
#include "includes/paramerters.h"
#include "includes/fileio.h"
#include "includes/statistics.h"
#include "includes/renderer.cuh"
#include "includes/utility.h"

#define DUMMY_REPEATS  2

Statistics statistics;


int main(int argc, char** argv){
    InputParameter param;
    Scene scene;
    int nthreads = 0;
    bool interactive = false;
    std::string outfile;

    /* 引数の解析 */
    while( argc >= 2 ) {
        if( strncmp(argv[1], "-n", 2) == 0 ) {
            nthreads = atoi(&argv[1][2]);
        } else if( strcmp(argv[1], "-i") == 0 ) {
            interactive = true;
        } else if( strncmp(argv[1], "-o", 2) == 0 ) {
            outfile = argv[1][2] ? &argv[1][2] : "out";
        } else if( strcmp(argv[1], "-gbvh") == 0 ) {
            param.build_type = BUILD_TREE_GBVH;
        } else if( strcmp(argv[1], "-bin") == 0 ) {
            param.build_type = BUILD_TREE_BIN;
        } else if( strcmp(argv[1], "-lbvh") == 0 ) {
            param.build_type = BUILD_TREE_LBVH;
        } else if( strcmp(argv[1], "-hlbvh") == 0 ) {
            param.build_type = BUILD_TREE_HLBVH;
        } else if( strcmp(argv[1], "-agc") == 0 ) {
            param.build_type = BUILD_TREE_AGC;
        } else if( strcmp(argv[1], "-rebuild") == 0 ) {
            param.build_type = BUILD_TREE_GBVH_REBUILD;
	    } else break;
	    --argc, ++argv;
    }

    param.build_type = BUILD_TREE_GBVH;

    if( argc < 2 || argv[1][0] == '-' ) {
        fprintf(stderr, "usage: raytr [-n<nthreads>] [-o[imgfile]] [-i] [-gbvh/-bin/-lbvh/-hlbvh/-agc] parameter-file\n");
        exit(1);
    }

    // パラメータファイルの読み込み
    std::string param_file = argv[1];
    printf("Input parameter file: %s\n", param_file.c_str());
    int dpos = param_file.find_last_of('./\\');
    if(dpos < 0 || param_file[dpos] != '.') param_file += ".param";

#ifdef _OPENMP
    if( nthreads >= 1 ) omp_set_num_threads(nthreads);
    printf("Number of threads = %d\n", omp_get_max_threads());
#else
    if( nthreads ) fprintf(stderr, "Warning: Single-thread only. The -n option will be ignored.\n");
#endif

     /* ファイルの読み込み */
    printf("Parameter file = \"%s\"\n", param_file.c_str());
    if( RT_ReadParamFile(param_file, param, scene) ) exit(1);
    printf("Obj file = \"%s\"\n", param.get_obj_file_name().c_str());
    if( RT_ReadObjFile(param, scene) ) exit(1);
    long long nobjs_f0 = 0, nins = 0, ndel = 0;
    for( int frame = 0; frame < scene.scenario.size(); frame++ ) {
        for( int i = 0; i < scene.scenario[frame].size(); i++ ) {
            if( scene.scenario[frame][i].obj_id >= 0 ) {
            frame == 0 ? nobjs_f0++ : nins++;
            } else ndel++;
        }
    }
    printf("Number of objects (frame 0): %s\n",
	   Statistics::format_int(nobjs_f0).c_str());
    printf("Number of object ins/dels: %lld %lld (%f%%)\n", nins, ndel,
	   100.0 * ndel / (nobjs_f0 * (param.end_frame - param.start_frame)));
    printf("Number of frames: %d\n", param.end_frame - param.start_frame + 1);

    {
        static const char* tree_type_names[] = {"GBVH", "GBVH-rebuild", "BIN", "LBVH", "HLBVH", "AGC"};
        printf("Tree type: %s%s ns=%d gl=%d KT=%f KI=%f\n",
           tree_type_names[param.build_type],
           DO_REFIT ? " REFIT" : "",
           NDIV_SHIFT, MAX_GRID_LEVEL, (float)BVTREE_SAH_KT, (float)BVTREE_SAH_KI);
    }

    int repeat_init = param.render_repeat == 1 ? 0 : -DUMMY_REPEATS;
    for( int repeat = repeat_init; repeat < param.render_repeat; repeat++ ) {
	    fprintf(stderr, "Rendering (%d/%d)...\r", repeat, param.render_repeat);

        if( repeat == 0 ) {
            statistics.clear_timer(ST_RAY_TRACE);
            statistics.clear_timer(ST_TREE_CONSTRUCT);
            statistics.clear_timer(ST_GRID_CONSTRUCT);
            statistics.clear_timer(ST_BV_CONSTRUCT);
        }
    }

    const int num_frames = scene.scenario.size();

    printf("Starting rendering %d frames...\n", num_frames);
    glm::vec3* framebuffers;
    const int image_width = param.image_size.x;
    const int image_height = param.image_size.y;
    const size_t framebuffer_size = num_frames * image_width * image_height * sizeof(vec3);

    CHECK_CUDA(cudaMallocManaged(&framebuffers, framebuffer_size));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    glm::vec2 resolution(image_width, image_height);

    vector<LeafNode*> dirty_leaves;
    build_initial_tree(scene, param, 0, dirty_leaves);
    printf("Initial tree built.\n");
    
    DeviceScene* d_scene;
    copy_scene_to_device_scene(scene, d_scene);
    printf("Scene copied to device.\n");
    
    glm::vec2 thread_size(8,8);
    dim3 blocks(image_width / thread_size.x + 1, image_height / thread_size.y + 1);
    dim3 threads(thread_size.x, thread_size.y);

    printf("Launching kernel with blocks (%d, %d), threads (%d, %d)\n", blocks.x, blocks.y, threads.x, threads.y);

    // 将来的にはparam fileから読めるようにしたい
    CameraParameter camera_param;
    camera_param.lower_left_corner = vec3(-1.0, -1.0, -1.0);
    camera_param.horizontal = vec3(2.0, 0.0, 0.0);
    camera_param.vertical = vec3(0.0, 2.0, 0.0);
    camera_param.origin = vec3(1.5, 2.0, 3.5);


    for(int frame = 0; frame < num_frames; frame++){
        if(scene.scenario.size() > 2){
            printf("Rendering frame %d / %d\n", frame, num_frames);
            
            if(frame > 0){
                // dirty_leaves.clear();
                printf("Modifying scene for frame %d...\n", frame);
                modify_scene(scene, param, frame, dirty_leaves);
                printf("Updating device BVH for frame %d...\n", frame);
                update_device_bvh(scene, d_scene);
            }
            render_image<<<blocks, threads>>>(framebuffers, image_width, image_height, camera_param, d_scene, frame);

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }
    
    export_to_ppm("/home/m5291093/cuda-gbvh/cuda-gbvh/build/results/f", framebuffers, image_width, image_height, num_frames);

    CHECK_CUDA(cudaFree(framebuffers));
    free_device_scene(d_scene);
    CHECK_CUDA(cudaDeviceReset());


    // print_frame_buffer(framebuffers, image_width, image_height, num_frames);
    return 0;
}