#include "includes/scene.cuh"

void copy_scene_to_device_scene(Scene& scene, DeviceScene*& d_scene){
        
    int num_objects = scene.objects.size();
    int num_frames = scene.scenario.size();

    DeviceScene h_device_scene;
    h_device_scene.bvtree_root = nullptr;
    h_device_scene.grid_root = nullptr;
    h_device_scene.tree_buffer = nullptr;

    std::vector<Triangle> h_tris(num_objects);
    for (int i = 0; i < num_objects; ++i) {
        Triangle* tri = static_cast<Triangle*>(scene.objects[i]); // object = triangle 前提
        h_tris[i] = *tri;  // ここで Triangle 全体をコピー
    }

    Triangle* d_tris = nullptr;
    CHECK_CUDA(cudaMalloc(&d_tris, sizeof(Triangle) * num_objects));
    CHECK_CUDA(cudaMemcpy(d_tris, h_tris.data(),
                          sizeof(Triangle) * num_objects,
                          cudaMemcpyHostToDevice));

    h_device_scene.triangles = d_tris;
    h_device_scene.num_triangles = num_objects;

    vector<int> h_num_actions(num_frames);
    int max_num_action = 0;

    for(int frame = 0; frame < num_frames; frame++){
        h_num_actions[frame] = scene.scenario[frame].size();
        max_num_action = std::max(max_num_action, h_num_actions[frame]);
    }

    int* d_num_actions;
    CHECK_CUDA(cudaMalloc(&d_num_actions, sizeof(int) * num_frames));
    CHECK_CUDA(cudaMemcpy(d_num_actions, h_num_actions.data(), sizeof(int) * num_frames, cudaMemcpyHostToDevice));

    h_device_scene.num_actions_at_frame = d_num_actions;
    h_device_scene.max_num_actions = max_num_action;

    // scenarioをフレームごとにmax_num_actionのサイズに揃えて1次元配列に変換
    vector<Action> h_scenario(num_frames * max_num_action);
    for(int f = 0; f < num_frames; f++){
        for(int i = 0; i < max_num_action; i++){
            int idx = f * max_num_action + i;
            if(i < h_num_actions[f]){
                h_scenario[idx] = scene.scenario[f][i];
            } else {
                int not_effective = -1;                 // GBVHだと-1になることもあるので別の方法を考える
                h_scenario[idx].obj_id = not_effective; 
            }
        }
    }

    Action* d_scenario;
    CHECK_CUDA(cudaMalloc(&d_scenario, sizeof(Action) * num_frames * max_num_action));
    CHECK_CUDA(cudaMemcpy(d_scenario, h_scenario.data(), sizeof(Action) * num_frames * max_num_action, cudaMemcpyHostToDevice));

    h_device_scene.scenario = d_scenario;

    CHECK_CUDA(cudaMalloc(&d_scene, sizeof(DeviceScene)));
    CHECK_CUDA(cudaMemcpy(d_scene, &h_device_scene, sizeof(DeviceScene), cudaMemcpyHostToDevice));
       
}

