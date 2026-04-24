#include <iostream>
#include <string.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <chrono>

#include "includes/external/glm/vec3.hpp"
#include "includes/external/glm/vec2.hpp"
#include "includes/scene.cuh"
#include "includes/paramerters.h"
#include "includes/fileio.h"
#include "includes/statistics.h"
#include "includes/renderer.cuh"
#include "includes/utility.h"
#include "includes/bvtree.cuh"

#define DUMMY_REPEATS  2

Statistics statistics; 

enum BVHBuildDeviceType {
    BUILD_TREE_CPU = 0,
    BUILD_TREE_GPU
};

static void debug_print_dirty_leaf_distribution(const std::vector<LeafNode*>& dirty_leaves)
{
    printf("========== dirty_leaves distribution ==========\n");
    printf("num dirty leaves = %zu\n", dirty_leaves.size());

    // bitsごとの個数
    std::map<int, int> bits_count;

    // (bits, code) ごとの個数
    struct Key {
        uint64_t code;
        uint8_t bits;

        bool operator==(const Key& other) const {
            return code == other.code && bits == other.bits;
        }
    };

    struct KeyHash {
        size_t operator()(const Key& k) const {
            // 雑でも十分
            return std::hash<uint64_t>()(k.code) ^ (std::hash<int>()((int)k.bits) << 1);
        }
    };

    std::unordered_map<Key, int, KeyHash> key_count;

    // bitsごとの code 分布
    std::map<int, std::unordered_map<uint64_t, int>> code_count_per_bits;

    int null_count = 0;

    for (size_t i = 0; i < dirty_leaves.size(); ++i) {
        LeafNode* leaf = dirty_leaves[i];
        if (!leaf) {
            null_count++;
            continue;
        }

        int bits = (int)leaf->grid_bits;
        uint64_t code = leaf->grid_code;

        bits_count[bits]++;

        Key k{code, (uint8_t)bits};
        key_count[k]++;

        code_count_per_bits[bits][code]++;
    }

    if (null_count > 0) {
        printf("WARNING: null dirty leaves = %d\n", null_count);
    }

    printf("\n-- bits histogram --\n");
    for (auto& kv : bits_count) {
        printf("bits=%d : count=%d\n", kv.first, kv.second);
    }

    printf("\n-- per bits unique code counts --\n");
    for (auto& kv : code_count_per_bits) {
        int bits = kv.first;
        auto& mp = kv.second;
        printf("bits=%d : unique_codes=%zu\n", bits, mp.size());
    }

    printf("\n-- duplicated (bits, code) groups --\n");
    int num_dup_groups = 0;
    int max_group_size = 0;
    Key max_key{0, 0};

    for (auto& kv : key_count) {
        const Key& k = kv.first;
        int cnt = kv.second;
        if (cnt > 1) {
            printf("bits=%u code=%llu count=%d\n",
                   (unsigned)k.bits,
                   (unsigned long long)k.code,
                   cnt);
            num_dup_groups++;

            if (cnt > max_group_size) {
                max_group_size = cnt;
                max_key = k;
            }
        }
    }

    if (num_dup_groups == 0) {
        printf("No duplicated (bits, code) groups found.\n");
    } else {
        printf("num duplicated groups = %d\n", num_dup_groups);
        printf("max duplicated group = (bits=%u, code=%llu), count=%d\n",
               (unsigned)max_key.bits,
               (unsigned long long)max_key.code,
               max_group_size);
    }

    printf("\n-- sample dirty leaves (first 32) --\n");
    for (size_t i = 0; i < dirty_leaves.size() && i < 32; ++i) {
        LeafNode* leaf = dirty_leaves[i];
        if (!leaf) {
            printf("[%zu] null\n", i);
            continue;
        }

        printf("[%zu] bits=%u code=%llu nobjs=%d is_dirty=%d\n",
               i,
               (unsigned)leaf->grid_bits,
               (unsigned long long)leaf->grid_code,
               leaf->nobjs,
               (int)leaf->is_dirty);
    }

    printf("==============================================\n");
}

static void debug_print_code_range_per_bits(const std::vector<LeafNode*>& dirty_leaves)
{
    std::map<int, uint64_t> min_code;
    std::map<int, uint64_t> max_code;
    std::map<int, bool> initialized;

    for (LeafNode* leaf : dirty_leaves) {
        if (!leaf) continue;

        int bits = (int)leaf->grid_bits;
        uint64_t code = leaf->grid_code;

        if (!initialized[bits]) {
            min_code[bits] = code;
            max_code[bits] = code;
            initialized[bits] = true;
        } else {
            min_code[bits] = std::min(min_code[bits], code);
            max_code[bits] = std::max(max_code[bits], code);
        }
    }

    printf("\n-- code range per bits --\n");
    for (auto& kv : initialized) {
        int bits = kv.first;
        if (!kv.second) continue;

        printf("bits=%d : min_code=%llu max_code=%llu\n",
               bits,
               (unsigned long long)min_code[bits],
               (unsigned long long)max_code[bits]);
    }
}

static void print_dirty_leaf_bits_distribution(const std::vector<LeafNode*>& dirty_leaves)
{
    std::map<int, int> bits_count;
    int null_count = 0;

    for (LeafNode* leaf : dirty_leaves)
    {
        if (!leaf) {
            null_count++;
            continue;
        }

        bits_count[(int)leaf->grid_bits]++;
    }

    printf("=== dirty_leaves target_bits distribution ===\n");
    for (const auto& kv : bits_count)
    {
        printf("bits %d : %d clusters\n", kv.first, kv.second);
    }

    if (null_count > 0) {
        printf("null leaves : %d\n", null_count);
    }
    printf("total dirty leaves : %zu\n", dirty_leaves.size());
    printf("============================================\n");
}

struct LeafKey {
    uint64_t code;
    uint8_t bits;

    bool operator==(const LeafKey& other) const {
        return code == other.code && bits == other.bits;
    }
};

struct LeafKeyHash {
    size_t operator()(const LeafKey& k) const {
        return std::hash<uint64_t>()(k.code) ^ (std::hash<int>()((int)k.bits) << 1);
    }
};

static void print_same_key_pair_stats(const std::vector<LeafNode*>& dirty_leaves)
{
    std::unordered_map<LeafKey, int, LeafKeyHash> counts;
    counts.reserve(dirty_leaves.size());

    int null_count = 0;

    for (LeafNode* leaf : dirty_leaves)
    {
        if (!leaf) {
            null_count++;
            continue;
        }

        LeafKey k{leaf->grid_code, leaf->grid_bits};
        counts[k]++;
    }

    long long total_clusters = 0;
    long long total_pairs_possible = 0;
    int num_groups = 0;
    int num_groups_with_pairs = 0;
    int max_group_size = 0;

    printf("=== same-key stats (CPU) ===\n");

    for (const auto& kv : counts)
    {
        const LeafKey& k = kv.first;
        int cnt = kv.second;
        int pairs = cnt / 2;

        total_clusters += cnt;
        total_pairs_possible += pairs;
        num_groups++;
        if (pairs > 0) num_groups_with_pairs++;
        max_group_size = std::max(max_group_size, cnt);

        if (cnt >= 2) {
            printf("bits=%d code=%llu : count=%d pairs=%d\n",
                   (int)k.bits,
                   (unsigned long long)k.code,
                   cnt,
                   pairs);
        }
    }

    printf("---------------------------------\n");
    printf("total dirty leaves      : %lld\n", total_clusters);
    printf("num key groups          : %d\n", num_groups);
    printf("groups with pairs       : %d\n", num_groups_with_pairs);
    printf("max group size          : %d\n", max_group_size);
    printf("total possible pairs    : %lld\n", total_pairs_possible);
    printf("null leaves             : %d\n", null_count);
    printf("=================================\n");
}

int main(int argc, char** argv){
    InputParameter param;
    Scene scene;
    int nthreads = 0;
    bool interactive = false;
    std::string outfile;
    BVHBuildDeviceType build_device = BUILD_TREE_GPU;

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
    DirtyKeySet dirty_keys;
    if(build_device == BUILD_TREE_CPU){
        build_initial_tree(scene, param, 0, dirty_leaves, dirty_keys);
        printf("Initial tree built.\n");
    } else {
        build_initial_grid(scene, param, 0, dirty_leaves, dirty_keys);
    }

    vector<ulonglong2> h_dirty_keys = build_sorted_dirty_keys_from_set(dirty_keys);


    // ここ修正案件、どうせclearするならbuild_initial_gridに渡さなくてよい
    dirty_leaves.clear();
    // dirty leavesの収集
    collect_dirty_leaves(scene.grid_root, dirty_leaves);
    scene.dirty_leaves = dirty_leaves;

    print_same_key_pair_stats(dirty_leaves);
    
    DeviceScene* d_scene;
    DeviceScene h_device_scene{};
    // sceneをGPU用の構造体にコピーしてGPUに転送
    copy_scene_to_device_scene(scene, d_scene, h_device_scene, h_dirty_keys);
    printf("Scene copied to device.\n");

    if(build_device == BUILD_TREE_GPU){
        build_initial_bvh_gpu(scene, d_scene, h_device_scene, 0, h_dirty_keys);
        printf("Initial BVH built on GPU.\n");
    }
    
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
            auto start_time = std::chrono::high_resolution_clock::now();
            printf("Rendering frame %d / %d\n", frame, num_frames);
            
            // if(frame > 0){
            //     // dirty_leaves.clear();
            //     modify_scene(scene, param, frame, dirty_leaves);
            //     dirty_leaves.clear();
            //     collect_dirty_leaves(scene.grid_root, dirty_leaves);
            //     scene.dirty_leaves = dirty_leaves;
            //     update_device_bvh(scene, d_scene);
            //     // sort_dirty_leaves_by_grid_code(d_scene->dirty_leaves, d_scene->num_dirty_leaves);
            // }

            if(frame > 0){
                dirty_keys.clear();
                // CPUでグリッド木を更新
                auto start_update = std::chrono::high_resolution_clock::now();
                update_grid_tree(scene, param, frame, dirty_leaves, dirty_keys);
                auto end_update = std::chrono::high_resolution_clock::now();
                double update_ms = std::chrono::duration<double, std::milli>(end_update
    - start_update).count();
                printf("Grid tree updated in %.2f ms, num_dirty_leaves=%zu\n", update_ms, dirty_leaves.size());

                h_dirty_keys = build_sorted_dirty_keys_from_set(dirty_keys);
                dirty_leaves.clear();
                // dirty leavesの収集、いったんすべてのleafをdirtyとして扱う
                collect_dirty_leaves(scene.grid_root, dirty_leaves);
                scene.dirty_leaves = dirty_leaves;
                auto start_update_gpu = std::chrono::high_resolution_clock::now();
                update_bvh_gpu(d_scene, h_device_scene, scene, h_dirty_keys, 0);
                auto end_update_gpu = std::chrono::high_resolution_clock::now();
                double update_gpu_ms = std::chrono::duration<double, std::milli>(end_update_gpu - start_update_gpu).count();
                printf("BVH updated on GPU in %.2f ms\n", update_gpu_ms);
            }
            
            auto start_render = std::chrono::high_resolution_clock::now();
            render_image<<<blocks, threads>>>(framebuffers, image_width, image_height, camera_param, d_scene, frame);
            promote_curr_to_prev(h_device_scene);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            auto end_render = std::chrono::high_resolution_clock::now();
            double render_ms = std::chrono::duration<double, std::milli>(end_render - start_render).count();
            printf("Frame %d rendered in %.2f ms\n", frame, render_ms);

            auto end_time = std::chrono::high_resolution_clock::now();

            double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            printf("Frame %d rendered in %.2f ms\n", frame, ms);
        }
    }
    
    export_to_ppm("/home/m5291093/cuda-gbvh/cuda-gbvh/build/results/f", framebuffers, image_width, image_height, num_frames);

    CHECK_CUDA(cudaFree(framebuffers));
    free_device_scene(d_scene);
    CHECK_CUDA(cudaDeviceReset());


    // print_frame_buffer(framebuffers, image_width, image_height, num_frames);
    return 0;
}



