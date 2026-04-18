#include "includes/scene.cuh"

int flatten_node(TreeNode *node, FlattenContext &ctx);
void destroy_tree(TreeNode *root);
// void process_actions(TreeNode* &root, const std::vector<Object*>& objects,
// 		     const std::vector<struct Action>& actions,
// 		     const AABB& cent_aabb, int frame, vector<LeafNode> dirty_leaves);

// void copy_scene_to_device_scene_cpu(Scene& scene, DeviceScene*& d_scene){
        
//     int num_objects = scene.objects.size();
//     int num_frames = scene.scenario.size();

//     DeviceScene h_device_scene {};

//     std::vector<Triangle> h_tris(num_objects);
//     for (int i = 0; i < num_objects; ++i) {
//         Triangle* tri = static_cast<Triangle*>(scene.objects[i]); // object = triangle 前提
//         h_tris[i] = *tri;  // ここで Triangle 全体をコピー
//     }

//     // Triangle* d_tris = nullptr;
//     // CHECK_CUDA(cudaMalloc(&d_tris, sizeof(Triangle) * num_objects));
//     // CHECK_CUDA(cudaMemcpy(d_tris, h_tris.data(),
//     //                       sizeof(Triangle) * num_objects,
//     //                       cudaMemcpyHostToDevice));

//     // h_device_scene.triangles = d_tris;
//     // h_device_scene.num_triangles = num_objects;

//     vector<int> h_num_actions(num_frames);
//     int max_num_action = 0;

//     for(int frame = 0; frame < num_frames; frame++){
//         h_num_actions[frame] = scene.scenario[frame].size();
//         max_num_action = std::max(max_num_action, h_num_actions[frame]);
//     }

//     int* d_num_actions;
//     CHECK_CUDA(cudaMalloc(&d_num_actions, sizeof(int) * num_frames));
//     CHECK_CUDA(cudaMemcpy(d_num_actions, h_num_actions.data(), sizeof(int) * num_frames, cudaMemcpyHostToDevice));

//     h_device_scene.num_actions_at_frame = d_num_actions;
//     h_device_scene.max_num_actions = max_num_action;

//     // scenarioをフレームごとにmax_num_actionのサイズに揃えて1次元配列に変換
//     vector<Action> h_scenario(num_frames * max_num_action);
//     for(int f = 0; f < num_frames; f++){
//         for(int i = 0; i < max_num_action; i++){
//             int idx = f * max_num_action + i;
//             if(i < h_num_actions[f]){
//                 h_scenario[idx] = scene.scenario[f][i];
//             } else {
//                 int not_effective = -1;                 // GBVHだと-1になることもあるので別の方法を考える
//                 h_scenario[idx].obj_id = not_effective; 
//             }
//         }
//     }

//     Action* d_scenario;
//     CHECK_CUDA(cudaMalloc(&d_scenario, sizeof(Action) * num_frames * max_num_action));
//     CHECK_CUDA(cudaMemcpy(d_scenario, h_scenario.data(), sizeof(Action) * num_frames * max_num_action, cudaMemcpyHostToDevice));

//     h_device_scene.scenario = d_scenario;

//     FlattenContext ctx;

//     // flatten（root index を取得）
//     int root_idx = flatten_node(scene.bvtree_root, ctx);

//     // flatten 後
//     Triangle* d_bvh_tris = nullptr;
//     int num_bvh_tris = (int)ctx.triangles.size();
//     CHECK_CUDA(cudaMalloc(&d_bvh_tris, sizeof(Triangle) * num_bvh_tris));
//     CHECK_CUDA(cudaMemcpy(d_bvh_tris,
//                         ctx.triangles.data(),
//                         sizeof(Triangle) * num_bvh_tris,
//                         cudaMemcpyHostToDevice));

//     printf("Number of triangles in BVH: %d\n", num_bvh_tris);

//     // こっちを traversal 用に使う
//     h_device_scene.triangles = d_bvh_tris;
//     h_device_scene.num_triangles = num_bvh_tris; // これも合わせる;


//     // --- GPU メモリ確保 ---
//     GPU_BVH_Node* d_bvh_nodes = nullptr;
//     GPU_LeafNode* d_bvh_leaves = nullptr;

//     int num_nodes  = ctx.nodes.size();
//     int num_leaves = ctx.leaves.size();

//     CHECK_CUDA(cudaMalloc(&d_bvh_nodes,
//         sizeof(GPU_BVH_Node) * num_nodes));
//     CHECK_CUDA(cudaMalloc(&d_bvh_leaves,
//         sizeof(GPU_LeafNode) * num_leaves));

//     // --- 転送 ---
//     CHECK_CUDA(cudaMemcpy(d_bvh_nodes,
//         ctx.nodes.data(),
//         sizeof(GPU_BVH_Node) * num_nodes,
//         cudaMemcpyHostToDevice));

//     CHECK_CUDA(cudaMemcpy(d_bvh_leaves,
//         ctx.leaves.data(),
//         sizeof(GPU_LeafNode) * num_leaves,
//         cudaMemcpyHostToDevice));

//     // --- DeviceScene に設定 ---
//     h_device_scene.bvh_nodes      = d_bvh_nodes;
//     h_device_scene.bvh_leaves     = d_bvh_leaves;
//     h_device_scene.bvh_root       = root_idx;
//     // h_device_scene.num_bvh_nodes  = num_nodes;
//     h_device_scene.num_bvh_leaves = num_leaves;

//     CHECK_CUDA(cudaMalloc(&d_scene, sizeof(DeviceScene)));
//     CHECK_CUDA(cudaMemcpy(d_scene, &h_device_scene, sizeof(DeviceScene), cudaMemcpyHostToDevice));
       
// }

void copy_scene_to_device_scene(
    Scene& scene,
    DeviceScene*& d_scene,
    DeviceScene& h_device_scene,
    const std::vector<ulonglong2>& h_dirty_keys   // CPU側で作った dirty_keys を受け取る
)
{
    int num_objects = (int)scene.objects.size();
    int num_frames  = (int)scene.scenario.size();

    // ------------------------------------------------------------
    // 0. h_device_scene を明示初期化
    // ------------------------------------------------------------
    h_device_scene.triangles = nullptr;
    h_device_scene.num_triangles = 0;

    h_device_scene.scenario = nullptr;
    h_device_scene.num_actions_at_frame = nullptr;
    h_device_scene.max_num_actions = 0;

    h_device_scene.dirty_leaves = nullptr;
    h_device_scene.num_dirty_leaves = 0;

    h_device_scene.prev_complete_leaves = nullptr;
    h_device_scene.num_prev_complete_leaves = 0;

    h_device_scene.prev_bvh_nodes = nullptr;
    h_device_scene.num_prev_bvh_nodes = nullptr;
    h_device_scene.prev_bvh_root_node_idx = -1;

    h_device_scene.curr_bvh_nodes = nullptr;
    h_device_scene.num_curr_bvh_nodes = nullptr;

    h_device_scene.clusters = nullptr;
    h_device_scene.num_clusters = 0;

    h_device_scene.curr_bvh_root_node_idx = -1;

    h_device_scene.dirty_keys = nullptr;
    h_device_scene.num_dirty_keys = 0;

    // ------------------------------------------------------------
    // 1. 全TriangleをGPUへ
    // ------------------------------------------------------------
    std::vector<Triangle> h_tris(num_objects);
    for (int i = 0; i < num_objects; ++i)
    {
        Triangle* tri = static_cast<Triangle*>(scene.objects[i]); // object = triangle 前提
        h_tris[i] = *tri;
    }

    Triangle* d_tris = nullptr;
    if (num_objects > 0) {
        CHECK_CUDA(cudaMalloc(&d_tris, sizeof(Triangle) * num_objects));
        CHECK_CUDA(cudaMemcpy(d_tris,
                              h_tris.data(),
                              sizeof(Triangle) * num_objects,
                              cudaMemcpyHostToDevice));
    }

    h_device_scene.triangles = d_tris;
    h_device_scene.num_triangles = num_objects;

    // ------------------------------------------------------------
    // 2. scenario をGPUへ
    // ------------------------------------------------------------
    std::vector<int> h_num_actions(num_frames);
    int max_num_action = 0;

    for (int frame = 0; frame < num_frames; frame++)
    {
        h_num_actions[frame] = (int)scene.scenario[frame].size();
        max_num_action = std::max(max_num_action, h_num_actions[frame]);
    }

    int* d_num_actions = nullptr;
    if (num_frames > 0) {
        CHECK_CUDA(cudaMalloc(&d_num_actions, sizeof(int) * num_frames));
        CHECK_CUDA(cudaMemcpy(d_num_actions,
                              h_num_actions.data(),
                              sizeof(int) * num_frames,
                              cudaMemcpyHostToDevice));
    }

    h_device_scene.num_actions_at_frame = d_num_actions;
    h_device_scene.max_num_actions = max_num_action;

    std::vector<Action> h_scenario(num_frames * max_num_action);
    for (int f = 0; f < num_frames; f++)
    {
        for (int i = 0; i < max_num_action; i++)
        {
            int idx = f * max_num_action + i;
            if (i < h_num_actions[f]) {
                h_scenario[idx] = scene.scenario[f][i];
            } else {
                h_scenario[idx].obj_id = -1;
            }
        }
    }

    Action* d_scenario = nullptr;
    if (num_frames > 0 && max_num_action > 0) {
        CHECK_CUDA(cudaMalloc(&d_scenario, sizeof(Action) * num_frames * max_num_action));
        CHECK_CUDA(cudaMemcpy(d_scenario,
                              h_scenario.data(),
                              sizeof(Action) * num_frames * max_num_action,
                              cudaMemcpyHostToDevice));
    }

    h_device_scene.scenario = d_scenario;

    // ------------------------------------------------------------
    // 3. dirty leaves をGPUへ
    // ------------------------------------------------------------
    int num_dirty_leaves = (int)scene.dirty_leaves.size();

    GPU_LeafNode* d_dirty_leaves = nullptr;
    if (num_dirty_leaves > 0) {
        CHECK_CUDA(cudaMalloc(&d_dirty_leaves, sizeof(GPU_LeafNode) * num_dirty_leaves));
    }

    std::vector<GPU_LeafNode> h_dirty_leaves(num_dirty_leaves);

    for (int i = 0; i < num_dirty_leaves; i++)
    {
        GPU_LeafNode h_dirty_leaf{};

        if (scene.dirty_leaves[i] == nullptr)
        {
            printf("WARNING: dirty_leaves[%d] is null\n", i);
            h_dirty_leaves[i] = h_dirty_leaf;
            continue;
        }

        h_dirty_leaf.aabb = scene.dirty_leaves[i]->aabb;

        int num_tris = std::min(scene.dirty_leaves[i]->nobjs, MAX_LEAF_SIZE);
        for (int j = 0; j < num_tris; ++j)
        {
            h_dirty_leaf.triangles[j] = scene.dirty_leaves[i]->triangles[j];
        }

        for (int j = num_tris; j < MAX_LEAF_SIZE; ++j)
        {
            h_dirty_leaf.triangles[j] = Triangle();
        }

        h_dirty_leaf.tri_offset = 0;
        h_dirty_leaf.tri_count  = num_tris;
        h_dirty_leaf.grid_code  = scene.dirty_leaves[i]->grid_code;
        h_dirty_leaf.grid_bits  = scene.dirty_leaves[i]->grid_bits;

        h_dirty_leaves[i] = h_dirty_leaf;
    }

    if (num_dirty_leaves > 0) {
        CHECK_CUDA(cudaMemcpy(d_dirty_leaves,
                              h_dirty_leaves.data(),
                              sizeof(GPU_LeafNode) * num_dirty_leaves,
                              cudaMemcpyHostToDevice));
    }

    h_device_scene.dirty_leaves = d_dirty_leaves;
    h_device_scene.num_dirty_leaves = num_dirty_leaves;

    // dirty leaf aabb 初期化
    if (num_dirty_leaves > 0) {
        constexpr int BLOCK_SIZE = 256;
        int grid_size = (num_dirty_leaves + BLOCK_SIZE - 1) / BLOCK_SIZE;

        init_dirty_leaf_aabbs_kernel<<<grid_size, BLOCK_SIZE>>>(
            d_dirty_leaves,
            num_dirty_leaves
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ------------------------------------------------------------
    // 4. dirty keys をGPUへ
    // ------------------------------------------------------------
    ulonglong2* d_dirty_keys = nullptr;
    int num_dirty_keys = (int)h_dirty_keys.size();

    if (num_dirty_keys > 0) {
        CHECK_CUDA(cudaMalloc(&d_dirty_keys, sizeof(ulonglong2) * num_dirty_keys));
        CHECK_CUDA(cudaMemcpy(d_dirty_keys,
                              h_dirty_keys.data(),
                              sizeof(ulonglong2) * num_dirty_keys,
                              cudaMemcpyHostToDevice));
    }

    h_device_scene.dirty_keys = d_dirty_keys;
    h_device_scene.num_dirty_keys = num_dirty_keys;

    // ------------------------------------------------------------
    // 5. dirty leaves -> initial GPU_Cluster[]
    // ------------------------------------------------------------
    GPU_Cluster* d_clusters = nullptr;
    if (num_dirty_leaves > 0) {
        CHECK_CUDA(cudaMalloc(&d_clusters, sizeof(GPU_Cluster) * num_dirty_leaves));

        int block = 256;
        int grid  = (num_dirty_leaves + block - 1) / block;
        kernel_build_initial_clusters_from_leaves<<<grid, block>>>(
            d_dirty_leaves,
            num_dirty_leaves,
            d_clusters
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    h_device_scene.clusters = d_clusters;
    h_device_scene.num_clusters = num_dirty_leaves;

    // ------------------------------------------------------------
    // 6. prev completed BVH は初回は空
    // ------------------------------------------------------------
    h_device_scene.prev_complete_leaves = nullptr;
    h_device_scene.num_prev_complete_leaves = 0;

    h_device_scene.prev_bvh_nodes = nullptr;

    int* d_num_prev_bvh_nodes = nullptr;
    CHECK_CUDA(cudaMalloc(&d_num_prev_bvh_nodes, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_num_prev_bvh_nodes, 0, sizeof(int)));

    h_device_scene.num_prev_bvh_nodes = d_num_prev_bvh_nodes;
    h_device_scene.prev_bvh_root_node_idx = -1;

    // ------------------------------------------------------------
    // 7. curr build nodes を確保
    //    初回は dirty clusters だけから build するので上限は N-1
    // ------------------------------------------------------------
    GPU_BVH_Node* d_curr_bvh_nodes = nullptr;
    int* d_num_curr_bvh_nodes = nullptr;

    if (num_dirty_leaves >= 2) {
        CHECK_CUDA(cudaMalloc(&d_curr_bvh_nodes,
                              sizeof(GPU_BVH_Node) * (num_dirty_leaves - 1)));
    }

    CHECK_CUDA(cudaMalloc(&d_num_curr_bvh_nodes, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_num_curr_bvh_nodes, 0, sizeof(int)));

    h_device_scene.curr_bvh_nodes = d_curr_bvh_nodes;
    h_device_scene.num_curr_bvh_nodes = d_num_curr_bvh_nodes;
    h_device_scene.curr_bvh_root_node_idx = -1;

    // ------------------------------------------------------------
    // 8. DeviceScene 自体をGPUへ
    // ------------------------------------------------------------
    CHECK_CUDA(cudaMalloc(&d_scene, sizeof(DeviceScene)));
    CHECK_CUDA(cudaMemcpy(d_scene,
                          &h_device_scene,
                          sizeof(DeviceScene),
                          cudaMemcpyHostToDevice));
}

// void update_device_bvh_cpu(const Scene &scene, DeviceScene *d_scene)
// {
//     // ----------------------------
//     // 1) CPU側で flatten
//     // ----------------------------
//     FlattenContext ctx{};
//     int root_idx = flatten_node(scene.bvtree_root, ctx);

//     const int num_nodes = (int)ctx.nodes.size();
//     const int num_leaves = (int)ctx.leaves.size();
//     const int num_tris = (int)ctx.triangles.size();
//     const int num_dirty_leaves = (int)scene.dirty_leaves.size();

//     // flatten が失敗しているなら安全に抜ける（必要なら abort）
//     if (root_idx < 0 || num_nodes <= 0)
//     {
//         // ここで device 側BVHを無効化するなら root=-1 を書き戻すなど
//         DeviceScene h{};
//         CHECK_CUDA(cudaMemcpy(&h, d_scene, sizeof(DeviceScene), cudaMemcpyDeviceToHost));
//         h.bvh_root = -1;
//         h.num_bvh_nodes = 0;
//         h.num_bvh_leaves = 0;
//         h.num_triangles = 0;
//         // 既存の領域は解放してもいいが、デバッグしやすいように残す運用もあり
//         CHECK_CUDA(cudaMemcpy(d_scene, &h, sizeof(DeviceScene), cudaMemcpyHostToDevice));
//         return;
//     }

//     // ----------------------------
//     // 2) 既存 DeviceScene を取得
//     // ----------------------------
//     DeviceScene h_device_scene{};
//     CHECK_CUDA(cudaMemcpy(&h_device_scene, d_scene, sizeof(DeviceScene), cudaMemcpyDeviceToHost));

//     // ----------------------------
//     // 3) 古いGPUバッファを解放（nodes/leaves/triangles）
//     // ----------------------------
//     if (h_device_scene.bvh_nodes)
//     {
//         CHECK_CUDA(cudaFree(h_device_scene.bvh_nodes));
//         h_device_scene.bvh_nodes = nullptr;
//     }
//     if (h_device_scene.bvh_leaves)
//     {
//         CHECK_CUDA(cudaFree(h_device_scene.bvh_leaves));
//         h_device_scene.bvh_leaves = nullptr;
//     }
//     if (h_device_scene.triangles)
//     {
//         CHECK_CUDA(cudaFree(h_device_scene.triangles));
//         h_device_scene.triangles = nullptr;
//     }
//     if (h_device_scene.dirty_leaves)
//     {
//         CHECK_CUDA(cudaFree(h_device_scene.dirty_leaves));
//         h_device_scene.dirty_leaves = nullptr;
//     }

//     // ----------------------------
//     // 4) 新規GPUバッファ確保
//     // ----------------------------
//     GPU_BVH_Node *d_bvh_nodes = nullptr;
//     GPU_LeafNode *d_bvh_leaves = nullptr;
//     Triangle *d_bvh_tris = nullptr;
//     GPU_LeafNode *d_dirty_leaves = nullptr;

//     CHECK_CUDA(cudaMalloc(&d_bvh_nodes, sizeof(GPU_BVH_Node) * num_nodes));
//     CHECK_CUDA(cudaMalloc(&d_bvh_leaves, sizeof(GPU_LeafNode) * num_leaves));

//     // triangles が 0 のケースもあるのでガード
//     if (num_tris > 0)
//     {
//         CHECK_CUDA(cudaMalloc(&d_bvh_tris, sizeof(Triangle) * num_tris));
//     }
//     else
//     {
//         d_bvh_tris = nullptr;
//     }

//     if (num_dirty_leaves > 0)
//     {
//         CHECK_CUDA(cudaMalloc(&d_dirty_leaves, sizeof(GPU_LeafNode) * num_dirty_leaves));
//     }
//     else
//     {
//         d_dirty_leaves = nullptr;
//     }

//     // ----------------------------
//     // 5) 転送（Host -> Device）
//     // ----------------------------
//     CHECK_CUDA(cudaMemcpy(d_bvh_nodes,
//                           ctx.nodes.data(),
//                           sizeof(GPU_BVH_Node) * num_nodes,
//                           cudaMemcpyHostToDevice));

//     CHECK_CUDA(cudaMemcpy(d_bvh_leaves,
//                           ctx.leaves.data(),
//                           sizeof(GPU_LeafNode) * num_leaves,
//                           cudaMemcpyHostToDevice));

//     if (num_tris > 0)
//     {
//         CHECK_CUDA(cudaMemcpy(d_bvh_tris,
//                               ctx.triangles.data(),
//                               sizeof(Triangle) * num_tris,
//                               cudaMemcpyHostToDevice));
//     }

//     if (num_dirty_leaves > 0)
//     {
//         vector<GPU_LeafNode> h_dirty_leaves(num_dirty_leaves);
//         for (int i = 0; i < num_dirty_leaves; i++)
//         {
//             GPU_LeafNode h_dirty_leaf{};

//             // Check for null pointer before dereferencing
//             if (scene.dirty_leaves[i] == nullptr)
//             {
//                 printf("WARNING: dirty_leaves[%d] is null\n", i);
//                 continue;
//             }

//             h_dirty_leaf.aabb = scene.dirty_leaves[i]->aabb;
//             int num_tris = std::min(scene.dirty_leaves[i]->nobjs, MAX_LEAF_SIZE);
//             for (int j = 0; j < num_tris; ++j)
//             {
//                 h_dirty_leaf.triangles[j] = scene.dirty_leaves[i]->triangles[j];
//             }

//             for (int j = num_tris; j < MAX_LEAF_SIZE; ++j)
//             {
//                 h_dirty_leaf.triangles[j] = Triangle(); // ダミーで初期化
//             }

//             h_dirty_leaf.tri_offset = 0; // 未使用
//             h_dirty_leaf.tri_count = scene.dirty_leaves[i]->nobjs;
//             h_dirty_leaf.grid_code = scene.dirty_leaves[i]->grid_code;
//             h_dirty_leaf.grid_bits = scene.dirty_leaves[i]->grid_bits;
//             h_dirty_leaves[i] = h_dirty_leaf;
//         }

//         CHECK_CUDA(cudaMemcpy(d_dirty_leaves,
//                               h_dirty_leaves.data(),
//                               sizeof(GPU_LeafNode) * num_dirty_leaves,
//                               cudaMemcpyHostToDevice));
//     }

//     // ----------------------------
//     // 6) DeviceScene を更新して書き戻す
//     // ----------------------------
//     h_device_scene.bvh_nodes = d_bvh_nodes;
//     h_device_scene.bvh_leaves = d_bvh_leaves;
//     h_device_scene.dirty_leaves = d_dirty_leaves;
//     h_device_scene.bvh_root = root_idx;
//     // h_device_scene.num_bvh_nodes = num_nodes;
//     h_device_scene.num_bvh_leaves = num_leaves;
//     h_device_scene.num_dirty_leaves = num_dirty_leaves;

//     h_device_scene.triangles = d_bvh_tris;
//     h_device_scene.num_triangles = num_tris;
//     CHECK_CUDA(cudaMemcpy(d_scene, &h_device_scene, sizeof(DeviceScene), cudaMemcpyHostToDevice));
// }

void free_device_scene(DeviceScene* d_scene)
{
    if (!d_scene) return;

    // ------------------------------------------------------------
    // 1. DeviceScene を host にコピー
    // ------------------------------------------------------------
    DeviceScene h_scene{};
    CHECK_CUDA(cudaMemcpy(&h_scene, d_scene, sizeof(DeviceScene), cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------
    // 2. 基本データ
    // ------------------------------------------------------------
    if (h_scene.triangles)
        CHECK_CUDA(cudaFree(h_scene.triangles));

    if (h_scene.scenario)
        CHECK_CUDA(cudaFree(h_scene.scenario));

    if (h_scene.num_actions_at_frame)
        CHECK_CUDA(cudaFree(h_scene.num_actions_at_frame));

    // ------------------------------------------------------------
    // 3. dirty
    // ------------------------------------------------------------
    if (h_scene.dirty_leaves)
        CHECK_CUDA(cudaFree(h_scene.dirty_leaves));

    if (h_scene.dirty_keys)
        CHECK_CUDA(cudaFree(h_scene.dirty_keys));

    // ------------------------------------------------------------
    // 4. prev BVH
    // ------------------------------------------------------------
    if (h_scene.prev_bvh_nodes)
        CHECK_CUDA(cudaFree(h_scene.prev_bvh_nodes));

    if (h_scene.num_prev_bvh_nodes)
        CHECK_CUDA(cudaFree(h_scene.num_prev_bvh_nodes));

    if (h_scene.prev_complete_leaves)
        CHECK_CUDA(cudaFree(h_scene.prev_complete_leaves));

    // ------------------------------------------------------------
    // 5. curr BVH
    // ------------------------------------------------------------
    if (h_scene.curr_bvh_nodes)
        CHECK_CUDA(cudaFree(h_scene.curr_bvh_nodes));

    if (h_scene.num_curr_bvh_nodes)
        CHECK_CUDA(cudaFree(h_scene.num_curr_bvh_nodes));

    // ------------------------------------------------------------
    // 6. clusters
    // ------------------------------------------------------------
    if (h_scene.clusters)
        CHECK_CUDA(cudaFree(h_scene.clusters));

    // ------------------------------------------------------------
    // 7. frame leaves
    // ------------------------------------------------------------
    if (h_scene.frame_leaves)
        CHECK_CUDA(cudaFree(h_scene.frame_leaves));

    // ------------------------------------------------------------
    // 8. 最後に DeviceScene 本体
    // ------------------------------------------------------------
    CHECK_CUDA(cudaFree(d_scene));
}

// ここからは別のファイルに移すべきかも
static void calc_scene_aabb(vector<Object *> &objects, AABB &scene_aabb, AABB &cent_aabb)
{
    for (int i = 0; i < objects.size(); i++)
    {
        AABB oaabb = objects[i]->get_aabb();
        vec3 centroid = 0.5f * (oaabb.vmin + oaabb.vmax);
        scene_aabb.insert(oaabb);
        cent_aabb.insert(centroid);
    }
}

// 初期木の生成
void build_initial_tree(Scene &scene, InputParameter &param, int frame, vector<LeafNode *> &dirty_leaves, DirtyKeySet& dirty_keys)
{
    calc_scene_aabb(scene.objects, scene.aabb, scene.cent_aabb);
    destroy_tree(scene.grid_root);

    scene.grid_root = nullptr;

    process_actions(scene.grid_root, scene.objects, scene.scenario[frame], scene.cent_aabb, frame, dirty_leaves, dirty_keys);

    scene.bvtree_root = build_bvh(scene.grid_root, scene.cent_aabb, 0).node;
}

void build_initial_grid(Scene &scene, InputParameter &param, int frame, vector<LeafNode *> &dirty_leaves, DirtyKeySet& dirty_keys)
{
    calc_scene_aabb(scene.objects, scene.aabb, scene.cent_aabb);
    destroy_tree(scene.grid_root);

    scene.grid_root = nullptr;

    process_actions(scene.grid_root, scene.objects, scene.scenario[frame], scene.cent_aabb, frame, dirty_leaves, dirty_keys);
}

void build_initial_bvh_gpu(Scene scene, DeviceScene* d_scene, DeviceScene& h_scene, cudaStream_t stream, const vector<ulonglong2>& h_dirty_keys)
{
    build_bvh_on_gpu(d_scene, 
                     h_scene,
                     scene.dirty_leaves, // CPU側で作った dirty_leaves を渡
                     h_dirty_keys,        // CPU側で作った dirty_keys を渡す
                     stream);
}

void update_bvh_gpu(
    DeviceScene* d_scene,
    DeviceScene& h_scene,
    Scene& scene,
    const std::vector<ulonglong2>& h_dirty_keys,
    cudaStream_t stream
)
{
    // ------------------------------------------------------------
    // 1. dirty leaves を host 側で GPU_LeafNode に詰める
    // ------------------------------------------------------------
    int num_dirty_leaves = (int)scene.dirty_leaves.size();

    std::vector<GPU_LeafNode> h_dirty_leaves(num_dirty_leaves);

    for (int i = 0; i < num_dirty_leaves; ++i)
    {
        GPU_LeafNode h_dirty_leaf{};

        if (scene.dirty_leaves[i] == nullptr)
        {
            printf("WARNING: scene.dirty_leaves[%d] is null\n", i);
            h_dirty_leaves[i] = h_dirty_leaf;
            continue;
        }

        // CPU 側 aabb は古い可能性があるので一旦コピーし、
        // あとで GPU 側で triangles から再計算
        h_dirty_leaf.aabb = scene.dirty_leaves[i]->aabb;

        int num_tris = std::min(scene.dirty_leaves[i]->nobjs, MAX_LEAF_SIZE);
        for (int j = 0; j < num_tris; ++j)
        {
            h_dirty_leaf.triangles[j] = scene.dirty_leaves[i]->triangles[j];
        }

        for (int j = num_tris; j < MAX_LEAF_SIZE; ++j)
        {
            h_dirty_leaf.triangles[j] = Triangle();
        }

        h_dirty_leaf.tri_offset = 0;
        h_dirty_leaf.tri_count  = num_tris;
        h_dirty_leaf.grid_code  = scene.dirty_leaves[i]->grid_code;
        h_dirty_leaf.grid_bits  = scene.dirty_leaves[i]->grid_bits;

        h_dirty_leaves[i] = h_dirty_leaf;
    }

    // ------------------------------------------------------------
    // 2. 今フレームで作り直す GPU リソースだけ解放
    //    prev_* は前フレーム完成BVHなので残す
    // ------------------------------------------------------------
    if (h_scene.dirty_leaves) {
        CHECK_CUDA(cudaFree(h_scene.dirty_leaves));
        h_scene.dirty_leaves = nullptr;
    }

    if (h_scene.dirty_keys) {
        CHECK_CUDA(cudaFree(h_scene.dirty_keys));
        h_scene.dirty_keys = nullptr;
    }

    if (h_scene.clusters) {
        CHECK_CUDA(cudaFree(h_scene.clusters));
        h_scene.clusters = nullptr;
    }

    if (h_scene.curr_bvh_nodes) {
        CHECK_CUDA(cudaFree(h_scene.curr_bvh_nodes));
        h_scene.curr_bvh_nodes = nullptr;
    }

    if (h_scene.num_curr_bvh_nodes) {
        CHECK_CUDA(cudaFree(h_scene.num_curr_bvh_nodes));
        h_scene.num_curr_bvh_nodes = nullptr;
    }

    h_scene.num_dirty_leaves = 0;
    h_scene.num_dirty_keys = 0;
    h_scene.num_clusters = 0;
    h_scene.curr_bvh_root_node_idx = -1;

    // ------------------------------------------------------------
    // 3. 新しい dirty leaves を GPU に転送
    // ------------------------------------------------------------
    GPU_LeafNode* d_dirty_leaves = nullptr;
    if (num_dirty_leaves > 0) {
        CHECK_CUDA(cudaMalloc(&d_dirty_leaves, sizeof(GPU_LeafNode) * num_dirty_leaves));
        CHECK_CUDA(cudaMemcpyAsync(d_dirty_leaves,
                                   h_dirty_leaves.data(),
                                   sizeof(GPU_LeafNode) * num_dirty_leaves,
                                   cudaMemcpyHostToDevice,
                                   stream));
    }

    h_scene.dirty_leaves = d_dirty_leaves;
    h_scene.num_dirty_leaves = num_dirty_leaves;

    // ------------------------------------------------------------
    // 4. dirty_keys を GPU に転送
    // ------------------------------------------------------------
    ulonglong2* d_dirty_keys = nullptr;
    int num_dirty_keys = (int)h_dirty_keys.size();

    if (num_dirty_keys > 0) {
        CHECK_CUDA(cudaMalloc(&d_dirty_keys, sizeof(ulonglong2) * num_dirty_keys));
        CHECK_CUDA(cudaMemcpyAsync(d_dirty_keys,
                                   h_dirty_keys.data(),
                                   sizeof(ulonglong2) * num_dirty_keys,
                                   cudaMemcpyHostToDevice,
                                   stream));
    }

    h_scene.dirty_keys = d_dirty_keys;
    h_scene.num_dirty_keys = num_dirty_keys;

    // ------------------------------------------------------------
    // 5. GPU で dirty leaf の AABB を triangles から再計算
    // ------------------------------------------------------------
    if (num_dirty_leaves > 0) {
        constexpr int BLOCK_SIZE = 256;
        int grid_size = (num_dirty_leaves + BLOCK_SIZE - 1) / BLOCK_SIZE;

        init_dirty_leaf_aabbs_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            d_dirty_leaves,
            num_dirty_leaves
        );
        CHECK_CUDA(cudaGetLastError());
    }

    // ------------------------------------------------------------
    // 6. dirty leaves -> initial GPU_Cluster[]
    // ------------------------------------------------------------
    GPU_Cluster* d_clusters = nullptr;
    if (num_dirty_leaves > 0) {
        CHECK_CUDA(cudaMalloc(&d_clusters, sizeof(GPU_Cluster) * num_dirty_leaves));

        int block = 256;
        int grid  = (num_dirty_leaves + block - 1) / block;
        kernel_build_initial_clusters_from_leaves<<<grid, block, 0, stream>>>(
            d_dirty_leaves,
            num_dirty_leaves,
            d_clusters
        );
        CHECK_CUDA(cudaGetLastError());
    }

    h_scene.clusters = d_clusters;
    h_scene.num_clusters = num_dirty_leaves;

    // ------------------------------------------------------------
    // 7. curr build nodes を確保
    //    初期状態では dirty_clusters 数を上限に確保
    //    affected_clusters を後で連結するなら build_bvh_on_gpu 内で
    //    必要に応じて再確保する設計でもよい
    // ------------------------------------------------------------
    GPU_BVH_Node* d_curr_bvh_nodes = nullptr;
    int* d_num_curr_bvh_nodes = nullptr;

    if (num_dirty_leaves >= 2) {
        CHECK_CUDA(cudaMalloc(&d_curr_bvh_nodes,
                              sizeof(GPU_BVH_Node) * (num_dirty_leaves - 1)));
    }

    CHECK_CUDA(cudaMalloc(&d_num_curr_bvh_nodes, sizeof(int)));
    CHECK_CUDA(cudaMemsetAsync(d_num_curr_bvh_nodes, 0, sizeof(int), stream));

    h_scene.curr_bvh_nodes = d_curr_bvh_nodes;
    h_scene.num_curr_bvh_nodes = d_num_curr_bvh_nodes;
    h_scene.curr_bvh_root_node_idx = -1;

    // ------------------------------------------------------------
    // 8. prev_* はそのまま残す
    //    （前フレームの完成BVHなのでここでは触らない）
    // ------------------------------------------------------------

    // ------------------------------------------------------------
    // 9. 更新済み DeviceScene を GPU に書き戻す
    // ------------------------------------------------------------
    CHECK_CUDA(cudaMemcpyAsync(d_scene,
                               &h_scene,
                               sizeof(DeviceScene),
                               cudaMemcpyHostToDevice,
                               stream));

    // ------------------------------------------------------------
    // 10. BVH 構築
    //     affected_clusters を使う場合は build_bvh_on_gpu 側で
    //     prev から集めて dirty clusters と連結してから reduce
    // ------------------------------------------------------------
    if (num_dirty_leaves >= 2) {
        build_bvh_on_gpu(
            d_scene,
            h_scene,              // ← 今の設計では h_scene ごと渡す方が自然
            scene.dirty_leaves,   // CPU 側 dirty leaf 一覧
            h_dirty_keys,         // CPU 側 dirty keys
            stream
        );
    }
    else if (num_dirty_leaves == 1) {
        printf("Only one dirty leaf, skipping AGC and using it as the BVH root candidate.\n");
    }

    printf("update_bvh_gpu: num_dirty_leaves=%d, num_dirty_keys=%d\n",
           num_dirty_leaves, num_dirty_keys);

    CHECK_CUDA(cudaGetLastError());
}
void update_grid_tree(Scene &scene, InputParameter &param, int frame, vector<LeafNode *> &dirty_leaves, DirtyKeySet& dirty_keys)
{
    if (param.build_type == BUILD_TREE_GBVH_REBUILD)
    {
        destroy_tree(scene.grid_root);
        scene.grid_root = nullptr;
    }

    process_actions(scene.grid_root, scene.objects, scene.scenario[frame], scene.cent_aabb, frame, dirty_leaves, dirty_keys);
}


void modify_scene(Scene &scene, InputParameter &param, int frame, vector<LeafNode *> &dirty_leaves, DirtyKeySet& dirty_keys)
{

    if (param.build_type == BUILD_TREE_GBVH_REBUILD)
    {
        destroy_tree(scene.grid_root);
        scene.grid_root = nullptr;
    }

    process_actions(scene.grid_root, scene.objects, scene.scenario[frame], scene.cent_aabb, frame, dirty_leaves, dirty_keys);
    scene.bvtree_root = build_bvh(scene.grid_root, scene.cent_aabb, 0).node;
}
