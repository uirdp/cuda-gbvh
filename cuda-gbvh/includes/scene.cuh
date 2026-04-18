#pragma once

#include "external/glm/vec3.hpp"
#include "object.cuh"
#include "aabb.h"
#include "bvtree.cuh"
#include "utility.h"
#include "check_cuda.h"
#include "paramerters.h"
#include "keys.h"
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
using glm::vec3;

struct TreeNode;
struct LeafNode;

struct Scene {
    vector<Object*> objects;               //　全フレーム分のオブジェクト
    map<std::string, Material*> materials;
    map<std::string, Texture*> textures;
    AABB aabb; // シーン全体のAABB
    AABB cent_aabb; // シーン全体の重心AABB
    TreeNode *bvtree_root;
    TreeNode *grid_root;
    void *tree_buffer;
    vector<LeafNode*> dirty_leaves;

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

struct GPU_BVH_Node;
struct GPU_LeafNode;
struct GPU_Cluster;
// __device__側のコードではvectorやmapが使えないので、Sceneをdeviceに送る際は一度SceneをDeviceSceneに変更する
struct DeviceScene{
    Triangle* triangles;
    int num_triangles;

    Action* scenario;
    int* num_actions_at_frame;
    int max_num_actions;

    // 今フレーム dirty leaves
    GPU_LeafNode* dirty_leaves;
    int num_dirty_leaves;

    // prev completed BVH
    GPU_LeafNode* prev_complete_leaves;
    int num_prev_complete_leaves;

    GPU_BVH_Node* prev_bvh_nodes;
    int* num_prev_bvh_nodes;
    int prev_bvh_root_node_idx;

    // curr build nodes
    GPU_BVH_Node* curr_bvh_nodes;
    int* num_curr_bvh_nodes;

    // AGC input clusters
    GPU_Cluster* clusters;
    int num_clusters;

    // 今フレーム完成 root（実質 curr 側）
    int curr_bvh_root_node_idx;

    ulonglong2* dirty_keys;
    int num_dirty_keys;

    GPU_LeafNode* frame_leaves;
    int num_frame_leaves;

    DeviceScene() : prev_bvh_root_node_idx(-1), curr_bvh_root_node_idx(-1) {}
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



void copy_scene_to_device_scene(
    Scene& scene,
    DeviceScene*& d_scene,
    DeviceScene& h_device_scene,
    const std::vector<ulonglong2>& h_dirty_keys   // CPU側で作った dirty_keys を受け取る
);

void free_device_scene(DeviceScene* d_scene);

static void calc_scene_aabb(vector<Object*>& objects, AABB& scene_aabb, AABB &cent_aabb);
void build_initial_tree(Scene &scene, InputParameter &param, int frame, vector<LeafNode *> &dirty_leaves, DirtyKeySet& dirty_keys);
void build_initial_grid(Scene &scene, InputParameter &param, int frame, vector<LeafNode *> &dirty_leaves, DirtyKeySet& dirty_keys);
void modify_scene(Scene &scene, InputParameter& param, int frame, vector<LeafNode*>& dirty_leaves);
void update_device_bvh(const Scene& scene, DeviceScene* d_scene);
void update_grid_tree(Scene &scene, InputParameter &param, int frame, vector<LeafNode *> &dirty_leaves, DirtyKeySet& dirty_keys);
void update_bvh_gpu(
    DeviceScene* d_scene,
    DeviceScene& h_scene,
    Scene& scene,
    const std::vector<ulonglong2>& h_dirty_keys,
    cudaStream_t stream
);