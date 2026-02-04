#include "includes/scene.cuh"

int flatten_node(TreeNode* node, FlattenContext& ctx);
void destroy_tree(TreeNode *root);
void process_actions(TreeNode* &root, const std::vector<Object*>& objects,
		     const std::vector<struct Action>& actions,
		     const AABB& cent_aabb, int frame);


void copy_scene_to_device_scene(Scene& scene, DeviceScene*& d_scene){
        
    int num_objects = scene.objects.size();
    int num_frames = scene.scenario.size();

    DeviceScene h_device_scene {};

    std::vector<Triangle> h_tris(num_objects);
    for (int i = 0; i < num_objects; ++i) {
        Triangle* tri = static_cast<Triangle*>(scene.objects[i]); // object = triangle 前提
        h_tris[i] = *tri;  // ここで Triangle 全体をコピー
    }

    // Triangle* d_tris = nullptr;
    // CHECK_CUDA(cudaMalloc(&d_tris, sizeof(Triangle) * num_objects));
    // CHECK_CUDA(cudaMemcpy(d_tris, h_tris.data(),
    //                       sizeof(Triangle) * num_objects,
    //                       cudaMemcpyHostToDevice));

    // h_device_scene.triangles = d_tris;
    // h_device_scene.num_triangles = num_objects;

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

    FlattenContext ctx;

    // flatten（root index を取得）
    int root_idx = flatten_node(scene.bvtree_root, ctx);

    // flatten 後
    Triangle* d_bvh_tris = nullptr;
    int num_bvh_tris = (int)ctx.triangles.size();
    CHECK_CUDA(cudaMalloc(&d_bvh_tris, sizeof(Triangle) * num_bvh_tris));
    CHECK_CUDA(cudaMemcpy(d_bvh_tris,
                        ctx.triangles.data(),
                        sizeof(Triangle) * num_bvh_tris,
                        cudaMemcpyHostToDevice));

    printf("Number of triangles in BVH: %d\n", num_bvh_tris);

    // こっちを traversal 用に使う
    h_device_scene.triangles = d_bvh_tris;
    h_device_scene.num_triangles = num_bvh_tris; // これも合わせる;


    // --- GPU メモリ確保 ---
    GPU_BVH_Node* d_bvh_nodes = nullptr;
    GPU_LeafNode* d_bvh_leaves = nullptr;

    int num_nodes  = ctx.nodes.size();
    int num_leaves = ctx.leaves.size();

    CHECK_CUDA(cudaMalloc(&d_bvh_nodes,
        sizeof(GPU_BVH_Node) * num_nodes));
    CHECK_CUDA(cudaMalloc(&d_bvh_leaves,
        sizeof(GPU_LeafNode) * num_leaves));

    // --- 転送 ---
    CHECK_CUDA(cudaMemcpy(d_bvh_nodes,
        ctx.nodes.data(),
        sizeof(GPU_BVH_Node) * num_nodes,
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_bvh_leaves,
        ctx.leaves.data(),
        sizeof(GPU_LeafNode) * num_leaves,
        cudaMemcpyHostToDevice));

    // --- DeviceScene に設定 ---
    h_device_scene.bvh_nodes      = d_bvh_nodes;
    h_device_scene.bvh_leaves     = d_bvh_leaves;
    h_device_scene.bvh_root       = root_idx;
    h_device_scene.num_bvh_nodes  = num_nodes;
    h_device_scene.num_bvh_leaves = num_leaves;

    CHECK_CUDA(cudaMalloc(&d_scene, sizeof(DeviceScene)));
    CHECK_CUDA(cudaMemcpy(d_scene, &h_device_scene, sizeof(DeviceScene), cudaMemcpyHostToDevice));
       
}



// bvh_nodes / bvh_leaves に加えて、flatten ctx の triangles も毎フレーム更新する版
// 前提：DeviceScene に以下のメンバがあること
//   GPU_BVH_Node* bvh_nodes;
//   GPU_LeafNode* bvh_leaves;
//   int bvh_root;
//   int num_bvh_nodes;
//   int num_bvh_leaves;
//   Triangle* triangles;       // BVH 用（ctx.triangles）
//   int num_triangles;

void update_device_bvh(const Scene& scene, DeviceScene* d_scene)
{
    // ----------------------------
    // 1) CPU側で flatten
    // ----------------------------
    FlattenContext ctx{};
    int root_idx = flatten_node(scene.bvtree_root, ctx);

    const int num_nodes  = (int)ctx.nodes.size();
    const int num_leaves = (int)ctx.leaves.size();
    const int num_tris   = (int)ctx.triangles.size();

    // flatten が失敗しているなら安全に抜ける（必要なら abort）
    if (root_idx < 0 || num_nodes <= 0) {
        // ここで device 側BVHを無効化するなら root=-1 を書き戻すなど
        DeviceScene h{};
        CHECK_CUDA(cudaMemcpy(&h, d_scene, sizeof(DeviceScene), cudaMemcpyDeviceToHost));
        h.bvh_root = -1;
        h.num_bvh_nodes = 0;
        h.num_bvh_leaves = 0;
        h.num_triangles = 0;
        // 既存の領域は解放してもいいが、デバッグしやすいように残す運用もあり
        CHECK_CUDA(cudaMemcpy(d_scene, &h, sizeof(DeviceScene), cudaMemcpyHostToDevice));
        return;
    }

    // ----------------------------
    // 2) 既存 DeviceScene を取得
    // ----------------------------
    DeviceScene h_device_scene{};
    CHECK_CUDA(cudaMemcpy(&h_device_scene, d_scene, sizeof(DeviceScene), cudaMemcpyDeviceToHost));

    // ----------------------------
    // 3) 古いGPUバッファを解放（nodes/leaves/triangles）
    // ----------------------------
    if (h_device_scene.bvh_nodes) {
        CHECK_CUDA(cudaFree(h_device_scene.bvh_nodes));
        h_device_scene.bvh_nodes = nullptr;
    }
    if (h_device_scene.bvh_leaves) {
        CHECK_CUDA(cudaFree(h_device_scene.bvh_leaves));
        h_device_scene.bvh_leaves = nullptr;
    }
    if (h_device_scene.triangles) {
        CHECK_CUDA(cudaFree(h_device_scene.triangles));
        h_device_scene.triangles = nullptr;
    }

    // ----------------------------
    // 4) 新規GPUバッファ確保
    // ----------------------------
    GPU_BVH_Node* d_bvh_nodes   = nullptr;
    GPU_LeafNode* d_bvh_leaves  = nullptr;
    Triangle*     d_bvh_tris    = nullptr;

    CHECK_CUDA(cudaMalloc(&d_bvh_nodes,  sizeof(GPU_BVH_Node) * num_nodes));
    CHECK_CUDA(cudaMalloc(&d_bvh_leaves, sizeof(GPU_LeafNode) * num_leaves));

    // triangles が 0 のケースもあるのでガード
    if (num_tris > 0) {
        CHECK_CUDA(cudaMalloc(&d_bvh_tris, sizeof(Triangle) * num_tris));
    } else {
        d_bvh_tris = nullptr;
    }

    // ----------------------------
    // 5) 転送（Host -> Device）
    // ----------------------------
    CHECK_CUDA(cudaMemcpy(d_bvh_nodes,
                          ctx.nodes.data(),
                          sizeof(GPU_BVH_Node) * num_nodes,
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_bvh_leaves,
                          ctx.leaves.data(),
                          sizeof(GPU_LeafNode) * num_leaves,
                          cudaMemcpyHostToDevice));

    if (num_tris > 0) {
        CHECK_CUDA(cudaMemcpy(d_bvh_tris,
                              ctx.triangles.data(),
                              sizeof(Triangle) * num_tris,
                              cudaMemcpyHostToDevice));
    }

    // ----------------------------
    // 6) DeviceScene を更新して書き戻す
    // ----------------------------
    h_device_scene.bvh_nodes      = d_bvh_nodes;
    h_device_scene.bvh_leaves     = d_bvh_leaves;
    h_device_scene.bvh_root       = root_idx;
    h_device_scene.num_bvh_nodes  = num_nodes;
    h_device_scene.num_bvh_leaves = num_leaves;

    h_device_scene.triangles      = d_bvh_tris;
    h_device_scene.num_triangles  = num_tris;

    CHECK_CUDA(cudaMemcpy(d_scene, &h_device_scene, sizeof(DeviceScene), cudaMemcpyHostToDevice));
}


void free_device_scene(DeviceScene* d_scene){
    CHECK_CUDA(cudaFree(d_scene->triangles));
    CHECK_CUDA(cudaFree(d_scene->scenario));
    CHECK_CUDA(cudaFree(d_scene->num_actions_at_frame));
    CHECK_CUDA(cudaFree(d_scene->bvh_nodes));
    CHECK_CUDA(cudaFree(d_scene->bvh_leaves));
    CHECK_CUDA(cudaFree(d_scene));
}

// ここからは別のファイルに移すべきかも
static void calc_scene_aabb(vector<Object*>& objects, AABB& scene_aabb, AABB &cent_aabb){
    for(int i = 0; i < objects.size(); i++){
        AABB oaabb = objects[i]->get_aabb();
        vec3 centroid = 0.5f * (oaabb.vmin + oaabb.vmax);
        scene_aabb.insert(oaabb);
        cent_aabb.insert(centroid);
    }
}

// 初期木の生成
void build_initial_tree(Scene& scene, InputParameter& param, int frame, vector<LeafNode*>& dirty_leaves){
    calc_scene_aabb(scene.objects, scene.aabb, scene.cent_aabb);
    destroy_tree(scene.grid_root);

    scene.grid_root = nullptr;

    process_actions(scene.grid_root, scene.objects, scene.scenario[frame], scene.cent_aabb, frame, dirty_leaves);

    scene.bvtree_root = build_bvh(scene.grid_root, scene.cent_aabb, 0).node;
}

void modify_scene(Scene &scene, InputParameter& param, int frame, vector<LeafNode*>& dirty_leaves){

    if(param.build_type == BUILD_TREE_GBVH_REBUILD){
        destroy_tree(scene.grid_root);
        scene.grid_root = nullptr;
    }

    process_actions(scene.grid_root, scene.objects, scene.scenario[frame], scene.cent_aabb, frame, dirty_leaves);
    scene.bvtree_root = build_bvh(scene.grid_root, scene.cent_aabb, 0).node;
}
