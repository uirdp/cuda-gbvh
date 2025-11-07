#include <iostream>
#include <string.h>
#include <cuda_runtime.h>

#include "../includes/glm/vec3.hpp"
#include "../includes/scene.h"
#include "../includes/paramerters.h"
#include "../includes/fileio.h"
#include "../includes/statistics.h"

#define DUMMY_REPEATS  2

#define CHECK_CUDA(val)
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line){
    if(result){
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";

        cudaDeviceReset();
        exit(99);
    }
}

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

    if( argc < 2 || argv[1][0] == '-' ) {
        fprintf(stderr, "usage: raytr [-n<nthreads>] [-o[imgfile]] [-i] [-gbvh/-bin/-lbvh/-hlbvh/-agc] parameter-file\n");
        exit(1);
    }

    // パラメータファイルの読み込み
    std::string param_file = argv[1];
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


    
    return 0;
}