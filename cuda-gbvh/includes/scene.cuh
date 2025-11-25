#pragma once

#include "external/glm/vec3.hpp"
#include "object.cuh"
#include "aabb.h"
#include "bvtree.h"
#include "utility.h"
#include "check_cuda.h"
#include <map>
#include <vector>
#include <string>
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>



struct Action {
    int obj_id; // 正のときは insert、負(1's complement)のときは delete
    __host__ __device__ Action() {}
    __host__ __device__ Action(int id) : obj_id(id) {}
};

struct ViewAction {
    enum ActionType{ VIEWPOINT, LOOKAT, VIEWUP } type;
    glm::vec3 value;

    ViewAction() {}
    ViewAction(ActionType t, const glm::vec3& v) : type(t), value(v) {}
};

using std::vector;
using std::map;
struct Scene {
    vector<Object*> objects;               //　全フレーム分のオブジェクト
    map<std::string, Material*> materials;
    map<std::string, Texture*> textures;
    TreeNode *bvtree_root;
    TreeNode *grid_root;
    void *tree_buffer;

    std::vector<std::vector<Action>> scenario; //　ここら辺もポインタに変更しないといけない
    std::vector<std::vector<ViewAction>> view_scenario;

    Scene() : bvtree_root(nullptr), grid_root(nullptr), tree_buffer(nullptr) {}
    ~Scene() {
        for(int i = 0; i < objects.size(); i++){
            // delete objects[i]; 
        }
        for(std::map<std::string, Material*>::iterator m = materials.begin();
            m != materials.end(); m++){
            delete m->second;
        }
        // if(bvtree_root) delete bvtree_root; // tree_bufferも解放される
    }

    std::vector<Triangle*> get_triangle_at_frame(int frame) const {
        vector<Triangle*> objs;
        vector<Action> actions = scenario[frame];
        for(int i = 0; i < actions.size(); i++){
            // -gbvh以外で起動した場合、actions[i]にはそのフレームでアクティブなオブジェクトのidが入っている
            objs.push_back((Triangle*)objects[actions[i].obj_id >= 0 ? actions[i].obj_id : ~actions[i].obj_id]);
        }
        return objs;
    }

    // これらはGBVH以外での使用を想定、GBVHで使う場合はそれ用に変更すること
    Triangle* get_triangle_at_frame(int frame, int idx) const {
        vector<Action> actions = scenario[frame];
        return (Triangle*)objects[actions[idx].obj_id >= 0 ? actions[idx].obj_id : ~actions[idx].obj_id];
    }

    int get_num_triangles_at_frame(int frame) const {
        return scenario[frame].size();
    }
};

// __device__側のコードではvectorやmapが使えないので、Sceneをdeviceに送る際は一度SceneをDeviceSceneに変更する
struct DeviceScene{
    Triangle* triangles;
    TreeNode *bvtree_root;
    TreeNode *grid_root;
    void *tree_buffer;

    int num_triangles;
    int* num_actions_at_frame;     // 各フレームのアクションのサイズ
    int max_num_actions; // Scenarioの中で最も大きいAction配列のサイズ max(scenario[frame].size())


    Action* scenario; 

    DeviceScene() : bvtree_root(nullptr), grid_root(nullptr), tree_buffer(nullptr) {}
    ~DeviceScene() {
        for(int i = 0; i < num_triangles; i++){
            // delete objects[i]; 
        }
        // if(bvtree_root) delete bvtree_root; // tree_bufferも解放される
    }

    // これらはGBVH以外での使用を想定、GBVHで使う場合はそれ用に変更すること
    __device__ Triangle* get_triangle_at_frame(int frame, int idx) const {
        int action_index = frame * max_num_actions + idx;
        Action action = scenario[action_index];

        // 無効スロットはすぐ nullptr を返す
        if (action.obj_id == -1) return nullptr;

        int tri_id = (action.obj_id >= 0) ? action.obj_id : ~action.obj_id;

        if (tri_id < 0 || tri_id >= num_triangles) return nullptr;

        return &triangles[tri_id];
    }

    __device__ int get_num_triangles_at_frame(int frame, int *num_actions) const {
        return num_actions[frame];
    }
};

__device__ inline int get_num_triangles_at_frame(const DeviceScene* scene, int frame) {
    return scene->num_actions_at_frame[frame];
}

void copy_scene_to_device_scene(Scene& scene, DeviceScene*& d_scene);

