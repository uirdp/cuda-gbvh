#include "includes/scene.cuh"

int flatten_node(TreeNode *node, FlattenContext &ctx);
void destroy_tree(TreeNode *root);
// void process_actions(TreeNode* &root, const std::vector<Object*>& objects,
// 		     const std::vector<struct Action>& actions,
// 		     const AABB& cent_aabb, int frame, vector<LeafNode> dirty_leaves);

void copy_scene_to_device_scene_cpu(Scene& scene, DeviceScene*& d_scene){
        
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
    // h_device_scene.num_bvh_nodes  = num_nodes;
    h_device_scene.num_bvh_leaves = num_leaves;

    CHECK_CUDA(cudaMalloc(&d_scene, sizeof(DeviceScene)));
    CHECK_CUDA(cudaMemcpy(d_scene, &h_device_scene, sizeof(DeviceScene), cudaMemcpyHostToDevice));
       
}

void copy_scene_to_device_scene(Scene& scene, DeviceScene*& d_scene, DeviceScene& h_device_scene)
{
    int num_objects = (int)scene.objects.size();
    int num_frames  = (int)scene.scenario.size();

    // ------------------------------------------------------------
    // 1. 全TriangleをGPUへ（必要なら）
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
            if (i < h_num_actions[f])
            {
                h_scenario[idx] = scene.scenario[f][i];
            }
            else
            {
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
            h_dirty_leaf.triangles[j] = Triangle(); // ダミー初期化
        }

        h_dirty_leaf.tri_offset = 0; // 今は未使用
        h_dirty_leaf.tri_count  = num_tris;
        h_dirty_leaf.grid_code  = scene.dirty_leaves[i]->grid_code;
        h_dirty_leaf.grid_bits  = scene.dirty_leaves[i]->grid_bits;

        h_dirty_leaves[i] = h_dirty_leaf;
    }

    // for (int i = 0; i < num_dirty_leaves; ++i) {
    // LeafNode* leaf = scene.dirty_leaves[i];
    // if (!leaf) continue;

    // const AABB& a = leaf->aabb;
    // printf("dirty[%d] nobjs=%d bits=%u code=%llu "
    //        "vmin=(%f,%f,%f) vmax=(%f,%f,%f)\n",
    //        i,
    //        leaf->nobjs,
    //        (unsigned)leaf->grid_bits,
    //        (unsigned long long)leaf->grid_code,
    //        a.vmin.x, a.vmin.y, a.vmin.z,
    //        a.vmax.x, a.vmax.y, a.vmax.z);
    // }

    // auto finite3 = [](const vec3& v){
    //     return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
    // };

//     for (int i = 0; i < num_dirty_leaves; ++i) {
//     LeafNode* leaf = scene.dirty_leaves[i];
//     if (!leaf) continue;

//     // const AABB& a = leaf->aabb;
//     // if (!finite3(a.vmin) || !finite3(a.vmax)) {
//     //     printf("BAD AABB at dirty[%d]\n", i);
//     // }
// }

    if (num_dirty_leaves > 0) {
        CHECK_CUDA(cudaMemcpy(d_dirty_leaves,
                              h_dirty_leaves.data(),
                              sizeof(GPU_LeafNode) * num_dirty_leaves,
                              cudaMemcpyHostToDevice));
    }

    h_device_scene.dirty_leaves     = d_dirty_leaves;
    h_device_scene.num_dirty_leaves = num_dirty_leaves;

    // init dirty leaf aabb
    if(num_dirty_leaves) {
        constexpr int BLOCK_SIZE = 256;
        int grid_size = (num_dirty_leaves + BLOCK_SIZE - 1) / BLOCK_SIZE;

        init_dirty_leaf_aabbs_kernel<<<grid_size, BLOCK_SIZE>>>(d_dirty_leaves, num_dirty_leaves);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ------------------------------------------------------------
    // 4. dirty leaves -> initial GPU_Cluster[]
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

    h_device_scene.clusters     = d_clusters;
    h_device_scene.num_clusters = num_dirty_leaves;

    // ------------------------------------------------------------
    // 5. AGC構築用 BVH node 配列 + node数カウンタ
    //    dirty leaf が N 個なら internal node 最大数は N-1
    // ------------------------------------------------------------
    GPU_BVH_Node* d_bvh_nodes = nullptr;
    int* d_num_bvh_nodes = nullptr;

    if (num_dirty_leaves >= 2) {
        CHECK_CUDA(cudaMalloc(&d_bvh_nodes, sizeof(GPU_BVH_Node) * (num_dirty_leaves - 1)));
    }

    CHECK_CUDA(cudaMalloc(&d_num_bvh_nodes, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_num_bvh_nodes, 0, sizeof(int)));

    h_device_scene.bvh_nodes = d_bvh_nodes;
    h_device_scene.num_bvh_nodes = d_num_bvh_nodes;

    // root はまだ未構築
    h_device_scene.bvh_root_node_idx = -1;

    // ------------------------------------------------------------
    // 6. DeviceScene 自体をGPUへ
    // ------------------------------------------------------------
    CHECK_CUDA(cudaMalloc(&d_scene, sizeof(DeviceScene)));
    CHECK_CUDA(cudaMemcpy(d_scene,
                          &h_device_scene,
                          sizeof(DeviceScene),
                          cudaMemcpyHostToDevice));
}

void update_device_bvh_cpu(const Scene &scene, DeviceScene *d_scene)
{
    // ----------------------------
    // 1) CPU側で flatten
    // ----------------------------
    FlattenContext ctx{};
    int root_idx = flatten_node(scene.bvtree_root, ctx);

    const int num_nodes = (int)ctx.nodes.size();
    const int num_leaves = (int)ctx.leaves.size();
    const int num_tris = (int)ctx.triangles.size();
    const int num_dirty_leaves = (int)scene.dirty_leaves.size();

    // flatten が失敗しているなら安全に抜ける（必要なら abort）
    if (root_idx < 0 || num_nodes <= 0)
    {
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
    if (h_device_scene.bvh_nodes)
    {
        CHECK_CUDA(cudaFree(h_device_scene.bvh_nodes));
        h_device_scene.bvh_nodes = nullptr;
    }
    if (h_device_scene.bvh_leaves)
    {
        CHECK_CUDA(cudaFree(h_device_scene.bvh_leaves));
        h_device_scene.bvh_leaves = nullptr;
    }
    if (h_device_scene.triangles)
    {
        CHECK_CUDA(cudaFree(h_device_scene.triangles));
        h_device_scene.triangles = nullptr;
    }
    if (h_device_scene.dirty_leaves)
    {
        CHECK_CUDA(cudaFree(h_device_scene.dirty_leaves));
        h_device_scene.dirty_leaves = nullptr;
    }

    // ----------------------------
    // 4) 新規GPUバッファ確保
    // ----------------------------
    GPU_BVH_Node *d_bvh_nodes = nullptr;
    GPU_LeafNode *d_bvh_leaves = nullptr;
    Triangle *d_bvh_tris = nullptr;
    GPU_LeafNode *d_dirty_leaves = nullptr;

    CHECK_CUDA(cudaMalloc(&d_bvh_nodes, sizeof(GPU_BVH_Node) * num_nodes));
    CHECK_CUDA(cudaMalloc(&d_bvh_leaves, sizeof(GPU_LeafNode) * num_leaves));

    // triangles が 0 のケースもあるのでガード
    if (num_tris > 0)
    {
        CHECK_CUDA(cudaMalloc(&d_bvh_tris, sizeof(Triangle) * num_tris));
    }
    else
    {
        d_bvh_tris = nullptr;
    }

    if (num_dirty_leaves > 0)
    {
        CHECK_CUDA(cudaMalloc(&d_dirty_leaves, sizeof(GPU_LeafNode) * num_dirty_leaves));
    }
    else
    {
        d_dirty_leaves = nullptr;
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

    if (num_tris > 0)
    {
        CHECK_CUDA(cudaMemcpy(d_bvh_tris,
                              ctx.triangles.data(),
                              sizeof(Triangle) * num_tris,
                              cudaMemcpyHostToDevice));
    }

    if (num_dirty_leaves > 0)
    {
        vector<GPU_LeafNode> h_dirty_leaves(num_dirty_leaves);
        for (int i = 0; i < num_dirty_leaves; i++)
        {
            GPU_LeafNode h_dirty_leaf{};

            // Check for null pointer before dereferencing
            if (scene.dirty_leaves[i] == nullptr)
            {
                printf("WARNING: dirty_leaves[%d] is null\n", i);
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
                h_dirty_leaf.triangles[j] = Triangle(); // ダミーで初期化
            }

            h_dirty_leaf.tri_offset = 0; // 未使用
            h_dirty_leaf.tri_count = scene.dirty_leaves[i]->nobjs;
            h_dirty_leaf.grid_code = scene.dirty_leaves[i]->grid_code;
            h_dirty_leaf.grid_bits = scene.dirty_leaves[i]->grid_bits;
            h_dirty_leaves[i] = h_dirty_leaf;
        }

        CHECK_CUDA(cudaMemcpy(d_dirty_leaves,
                              h_dirty_leaves.data(),
                              sizeof(GPU_LeafNode) * num_dirty_leaves,
                              cudaMemcpyHostToDevice));
    }

    // ----------------------------
    // 6) DeviceScene を更新して書き戻す
    // ----------------------------
    h_device_scene.bvh_nodes = d_bvh_nodes;
    h_device_scene.bvh_leaves = d_bvh_leaves;
    h_device_scene.dirty_leaves = d_dirty_leaves;
    h_device_scene.bvh_root = root_idx;
    // h_device_scene.num_bvh_nodes = num_nodes;
    h_device_scene.num_bvh_leaves = num_leaves;
    h_device_scene.num_dirty_leaves = num_dirty_leaves;

    h_device_scene.triangles = d_bvh_tris;
    h_device_scene.num_triangles = num_tris;
    CHECK_CUDA(cudaMemcpy(d_scene, &h_device_scene, sizeof(DeviceScene), cudaMemcpyHostToDevice));
}

void free_device_scene(DeviceScene *d_scene)
{
    CHECK_CUDA(cudaFree(d_scene->triangles));
    CHECK_CUDA(cudaFree(d_scene->scenario));
    CHECK_CUDA(cudaFree(d_scene->num_actions_at_frame));
    CHECK_CUDA(cudaFree(d_scene->bvh_nodes));
    CHECK_CUDA(cudaFree(d_scene->bvh_leaves));
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
void build_initial_tree(Scene &scene, InputParameter &param, int frame, vector<LeafNode *> &dirty_leaves)
{
    calc_scene_aabb(scene.objects, scene.aabb, scene.cent_aabb);
    destroy_tree(scene.grid_root);

    scene.grid_root = nullptr;

    process_actions(scene.grid_root, scene.objects, scene.scenario[frame], scene.cent_aabb, frame, dirty_leaves);

    scene.bvtree_root = build_bvh(scene.grid_root, scene.cent_aabb, 0).node;
}

void build_initial_grid(Scene &scene, InputParameter &param, int frame, vector<LeafNode *> &dirty_leaves)
{
    calc_scene_aabb(scene.objects, scene.aabb, scene.cent_aabb);
    destroy_tree(scene.grid_root);

    scene.grid_root = nullptr;

    process_actions(scene.grid_root, scene.objects, scene.scenario[frame], scene.cent_aabb, frame, dirty_leaves);
}

void build_initial_bvh_gpu(DeviceScene* d_scene, DeviceScene& h_scene, cudaStream_t stream)
{
    build_bvh_on_gpu(d_scene, 
                     h_scene.clusters,
                     h_scene.num_clusters,
                     h_scene.bvh_nodes,
                     h_scene.num_bvh_nodes,
                     stream);
}

void update_bvh_gpu(DeviceScene* d_scene, DeviceScene& h_scene, Scene& scene, cudaStream_t stream)
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

        // CPU 側の aabb は古い可能性があるが、ひとまずコピーしておく
        // 直後に GPU 側で triangles から再計算する
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

        h_dirty_leaf.tri_offset = 0; // 未使用
        h_dirty_leaf.tri_count  = num_tris;
        h_dirty_leaf.grid_code  = scene.dirty_leaves[i]->grid_code;
        h_dirty_leaf.grid_bits  = scene.dirty_leaves[i]->grid_bits;

        h_dirty_leaves[i] = h_dirty_leaf;
    }

    // ------------------------------------------------------------
    // 2. 前フレームの GPU リソースを解放
    // ------------------------------------------------------------
    if (h_scene.dirty_leaves) {
        CHECK_CUDA(cudaFree(h_scene.dirty_leaves));
        h_scene.dirty_leaves = nullptr;
    }

    if (h_scene.clusters) {
        CHECK_CUDA(cudaFree(h_scene.clusters));
        h_scene.clusters = nullptr;
    }

    if (h_scene.bvh_nodes) {
        CHECK_CUDA(cudaFree(h_scene.bvh_nodes));
        h_scene.bvh_nodes = nullptr;
    }

    if (h_scene.num_bvh_nodes) {
        CHECK_CUDA(cudaFree(h_scene.num_bvh_nodes));
        h_scene.num_bvh_nodes = nullptr;
    }

    h_scene.num_dirty_leaves = 0;
    h_scene.num_clusters = 0;
    h_scene.bvh_root_node_idx = -1;

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
    // 4. GPU で dirty leaf の AABB を triangles から再計算
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
    // 5. dirty leaves -> initial GPU_Cluster[]
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
    // 6. AGC構築用 BVH node 配列 + node数カウンタを確保
    // ------------------------------------------------------------
    GPU_BVH_Node* d_bvh_nodes = nullptr;
    int* d_num_bvh_nodes = nullptr;

    if (num_dirty_leaves >= 2) {
        CHECK_CUDA(cudaMalloc(&d_bvh_nodes, sizeof(GPU_BVH_Node) * (num_dirty_leaves - 1)));
    }

    CHECK_CUDA(cudaMalloc(&d_num_bvh_nodes, sizeof(int)));
    CHECK_CUDA(cudaMemsetAsync(d_num_bvh_nodes, 0, sizeof(int), stream));

    h_scene.bvh_nodes = d_bvh_nodes;
    h_scene.num_bvh_nodes = d_num_bvh_nodes;
    h_scene.bvh_root_node_idx = -1; // root はまだ未構築

    // ------------------------------------------------------------
    // 7. 更新済み DeviceScene を GPU に書き戻す
    // ------------------------------------------------------------
    CHECK_CUDA(cudaMemcpyAsync(d_scene,
                               &h_scene,
                               sizeof(DeviceScene),
                               cudaMemcpyHostToDevice,
                               stream));

    // ------------------------------------------------------------
    // 8. AGC で BVH を構築
    // ------------------------------------------------------------
    if (num_dirty_leaves >= 2) {
        build_bvh_on_gpu(d_scene,
                         h_scene.clusters,
                         h_scene.num_clusters,
                         h_scene.bvh_nodes,
                         h_scene.num_bvh_nodes,
                         stream);
    }
    else if (num_dirty_leaves == 1) {
        printf("Only one dirty leaf, skipping AGC and using it as the BVH root.\n");
    }

    printf("update_bvh_gpu: num_dirty_leaves=%d\n", num_dirty_leaves);

    CHECK_CUDA(cudaGetLastError());
}
void update_grid_tree(Scene &scene, InputParameter &param, int frame, vector<LeafNode *> &dirty_leaves)
{
    if (param.build_type == BUILD_TREE_GBVH_REBUILD)
    {
        destroy_tree(scene.grid_root);
        scene.grid_root = nullptr;
    }

    process_actions(scene.grid_root, scene.objects, scene.scenario[frame], scene.cent_aabb, frame, dirty_leaves);
}


void modify_scene(Scene &scene, InputParameter &param, int frame, vector<LeafNode *> &dirty_leaves)
{

    if (param.build_type == BUILD_TREE_GBVH_REBUILD)
    {
        destroy_tree(scene.grid_root);
        scene.grid_root = nullptr;
    }

    process_actions(scene.grid_root, scene.objects, scene.scenario[frame], scene.cent_aabb, frame, dirty_leaves);
    scene.bvtree_root = build_bvh(scene.grid_root, scene.cent_aabb, 0).node;
}
