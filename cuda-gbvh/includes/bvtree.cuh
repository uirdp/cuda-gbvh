#pragma once

#include "object.cuh"
#include "keys.h"
// #include "embree/kernels/bvh/bvh.h"
// #include "embree/common/math/vec3fa.h"
// #include "embree/kernels/geometry/triangle.h"
#include "aabb.h"
#include "external/glm/vec3.hpp"
#include "triangle.cuh"
#include "utility.h"
#include "intersection.cuh"
#include "scene.cuh"

#include <cuda_runtime.h>
#include <limits>
#include <unordered_set>
#include <vector>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#define BVTREE_MAX_LEVEL 30 // Tree の最大深さ
#define BVTREE_SAH_KB 1.0   // SAHにおけるAABB交差判定コストの定数
#define BVTREE_SAH_KT 1     // SAHにおけるトラバーサルコストの定数
#define BVTREE_SAH_KI 0.2   // SAHにおける交差計算コストの定数

// GBVHでNDIV_SHIFTを3にするときには、MAX_GRID_LEVELを3にするか、CodeTypeを拡張する。
// HLBVHで分割数を増やすときは、NDIV_SHIFTを増やし、NBITS_INT_COORDは10に固定。
#define NDIV_SHIFT 2
// #define NDIV_SHIFT   3
#define NDIV (1 << NDIV_SHIFT)
#define NDIV_MASK ((1 << NDIV_SHIFT) - 1)
#define IDX_MASK ((1 << (NDIV_SHIFT * 3)) - 1)

#define MAX_GRID_LEVEL 5 /* 生成されるBVHの高さは最大で (MAX_GRID_LEVEL *(NDIV - 1) * 3) になる */
#define NBITS_INT_COORD (MAX_GRID_LEVEL * NDIV_SHIFT)
// #define NBITS_INT_COORD   10
#define CODE_LENGTH (NBITS_INT_COORD * 3)

#define EXT_GRID_LEVEL 3
#define EXT_NBITS_INT_COORD (EXT_GRID_LEVEL * NDIV_SHIFT)
#define EXT_CODE_LENGTH (EXT_NBITS_INT_COORD * 3)

#define MAX_LEAF_SIZE 8 // must be a multiple of 4
#define LEAF_BUF_SIZE MAX_LEAF_SIZE
#define NT_LEAF 0
#define NT_BRANCH 1
#define NT_GRID 2

#define USE_EXPIRE 0
#if DO_REFIT
#undef USE_EXPIRE
#define USE_EXPIRE 0
#endif

using std::vector;

struct TreeNode
{
    int type;
};

// 中間ノード
struct BVH_Node : public TreeNode
{
    int axis; // 分割軸
    AABB aabbs[2];
    TreeNode *left, *right;
    BVH_Node(int axis, TreeNode *left, TreeNode *right)
        : axis(axis), left(left), right(right)
    {
        type = NT_BRANCH;
    }
};

enum GPUChildType
{
    GPU_CHILD_LEAF = 0,
    GPU_CHILD_NODE = 1
};

enum GPUNodeSource
{
    GPU_NODE_SRC_NONE = 0,
    GPU_NODE_SRC_PREV = 1,
    GPU_NODE_SRC_CURR = 2
};

struct GPU_BVH_Node
{
    AABB aabb;

    int left_idx;
    int right_idx;

    int left_type;  // GPU_CHILD_LEAF or GPU_CHILD_NODE
    int right_type; // GPU_CHILD_LEAF or GPU_CHILD_NODE

    int left_source;  // GPU_NODE_SRC_NONE, GPU_NODE_SRC_PREV, GPU_NODE_SRC_CURR
    int right_source; // GPU_NODE_SRC_NONE, GPU_NODE_SRC_PREV, GPU_NODE

    int left;
    int right;
    int leaf;

    uint64_t grid_code;
    uint8_t grid_bits;

    __host__ __device__
    GPU_BVH_Node()
        : left_idx(-1), right_idx(-1),
          left_type(-1), right_type(-1),
          left_source(GPU_NODE_SRC_NONE), right_source(GPU_NODE_SRC_NONE),
          left(-1), right(-1), leaf(-1),
          grid_code(0), grid_bits(0) {}
};

struct GPU_LeafNode
{
    AABB aabb;
    int tri_offset; // 未使用
    uint64_t grid_code;
    uint8_t grid_bits;
    uint64_t leaf_code;
    int tri_count; // 未使用
    Triangle triangles[MAX_LEAF_SIZE];
};

struct GPU_Cluster
{
    AABB aabb;
    uint64_t grid_code;
    uint8_t grid_bits;
    int leaf_idx;    // -1 if not a leaf, otherwise the index of the leaf in the GPU_LeafNode array
    int node_idx;    // -1 if not a node, otherwise the index of the node in the GPU_BVH_Node array
    int node_source; // GPU_NODE_SRC_NONE, GPU_NODE_SRC_PREV, GPU_NODE_SRC_CURR
};

struct GPU_BVH
{
    GPU_BVH_Node *nodes;
    GPU_LeafNode *leaves;
    Triangle *triangles;
    int root;
};

struct FlattenContext
{
    vector<GPU_BVH_Node> nodes;
    vector<GPU_LeafNode> leaves;
    vector<Triangle> triangles;
    vector<GPU_LeafNode> dirty_leaves;
};

struct GridNodeBase : public TreeNode
{
    int nobjs;
    int expire;
    GridNodeBase() : nobjs(0), expire(INT_MAX) {}
};

static constexpr int GRID_BITS_PER_LEVEL = NDIV_SHIFT * 3;
// グリッドノード
struct GridNode : public GridNodeBase
{
    TreeNode *cells[NDIV * NDIV * NDIV];
    AABB aabb;
    BVH_Node *bvh_node;
    BVH_Node *node_alloc_buf;
    /* bvh_node と bvh_alloc_buf を１つにすることも考えたが、子が１つしかない
       グリッドノードのときに、２重freeの問題があったのでやめた */
    float cost;
    bool is_dirty;

    uint64_t grid_code; // GBVHで使用
    uint8_t grid_bits;

public:
    GridNode() : node_alloc_buf(nullptr), is_dirty(true)
    {
        type = NT_GRID;
        memset((void *)cells, 0, sizeof(TreeNode *) * NDIV * NDIV * NDIV);
    }

    static int get_index(int x, int y, int z) { return (z * NDIV + y) * NDIV + x; }
    static int get_index(const glm::ivec3 &idx) { return (idx.z * NDIV + idx.y) * NDIV + idx.x; }

    inline void set_code(uint64_t code, uint8_t bits)
    {
        this->grid_code = code;
        this->grid_bits = bits;
    }
};

struct LeafLessByGridCode
{
    __host__ __device__ bool operator()(const GPU_LeafNode &a, const GPU_LeafNode &b) const
    {
        if (a.grid_code == b.grid_code)
        {
            return a.grid_bits > b.grid_bits; // bitsが多い方が先
        }
        return a.grid_code < b.grid_code; // codeが小さい方が先
    }
};

// 葉ノード
struct LeafNode : public GridNodeBase
{
    Triangle *triangles;
    int *obj_ids;
    Triangle triangles_buf[LEAF_BUF_SIZE];
    AABB aabb;          // GBVHのみで使用
    BVH_Node *bvh_node; // GBVHで葉をBVH木に分解したときにその根が入る
    bool is_dirty;

    // allocated は「heap 側 triangles の容量（Triangle個数）」。
    // 0 なら triangles_buf を使っている
    int allocated;

    uint64_t grid_code = 0;
    uint8_t grid_bits = 0;
    uint64_t leaf_code = 0; // 追加しました


    static constexpr int SMALL_CAP = MAX_LEAF_SIZE;
    Triangle *tri_buf[SMALL_CAP];
    int id_buf[SMALL_CAP];

    int capacity = 0;

    LeafNode() : is_dirty(true), allocated(0), bvh_node(nullptr)
    {
        type = NT_LEAF;
        triangles = triangles_buf;
        obj_ids = id_buf;
        capacity = SMALL_CAP;

        leaf_code = alloc_leaf_code();
    }

    ~LeafNode()
    {
        if (allocated)
        {
            if (triangles != triangles_buf)
                free(triangles);
            if (obj_ids != id_buf)
                free(obj_ids);
        }
    }

 
    static uint64_t g_next_leaf_code;
    static inline uint64_t alloc_leaf_code()
    {
        return g_next_leaf_code++;
    }

    __host__ __device__ Object *get_object(int i)
    {
        return (Object *)&triangles[i];
    }

    void set_object(int i, Object *obj)
    {
        triangles[i] = *(Triangle *)obj;
    }

    inline void set_grid_code(uint64_t code, uint8_t bits)
    {
        grid_code = code;
        grid_bits = bits;
    }

    // nobjs 個の triangle に十分な領域を割り当てる（必要なら）
    void allocate()
    {
        if (nobjs <= LEAF_BUF_SIZE)
            return;

        // heap に nobjs 個ぶん確保
        triangles = (Triangle *)malloc(sizeof(Triangle) * nobjs);
        if (!triangles)
            no_memory();

        // いま buf に入っている分をコピー（nobjs は buf容量を超えているが、
        // 実際に入っているのは最大 LEAF_BUF_SIZE のはず）
        const int copy_n = (LEAF_BUF_SIZE < nobjs) ? LEAF_BUF_SIZE : nobjs;
        memcpy(triangles, triangles_buf, sizeof(Triangle) * copy_n);

        allocated = nobjs; // capacity = nobjs
    }

    // 1つ triangle を追加できるように容量を確保（push の直前に呼ぶ想定）
    void expand()
    {
        // buf にまだ入る
        if (allocated == 0)
        {
            if (nobjs < LEAF_BUF_SIZE)
                return;

            // buf から heap に移行（まずは倍）
            int new_cap = LEAF_BUF_SIZE * 2;
            triangles = (Triangle *)malloc(sizeof(Triangle) * new_cap);
            if (!triangles)
                no_memory();

            memcpy(triangles, triangles_buf, sizeof(Triangle) * LEAF_BUF_SIZE);
            allocated = new_cap;
            return;
        }

        // heap 使用中：容量が足りる
        if (nobjs < allocated)
            return;

        // heap 容量拡張（倍々）
        int new_cap = allocated * 2;
        Triangle *new_ptr = (Triangle *)realloc(triangles, sizeof(Triangle) * new_cap);
        if (!new_ptr)
            no_memory();

        triangles = new_ptr;
        allocated = new_cap;
    }

    inline void ensure_capacity(int need)
    {
        if (need <= capacity)
            return;

        int new_cap = capacity;
        while (new_cap < need)
        {
            new_cap *= 2;
        }
    }

    __host__ __device__ inline int get_obj_id(int i) const
    {
        return obj_ids[i];
    }
};

struct ReturnRecord
{
    AABB aabb;
    TreeNode *node;
    float cost;
    ReturnRecord() : node(nullptr), cost(0) {}
    ReturnRecord(TreeNode *node, float cost, const AABB &aabb) : node(node), cost(cost), aabb(aabb) {}
};

struct ListElement : public ReturnRecord
{
    int idx;
    ListElement() {}
    ListElement(const ReturnRecord &re, int idx) : ReturnRecord(re), idx(idx) {}
};

TreeNode *build_tree_bin(const std::vector<Object *> &objects,
                         const std::vector<struct Action> &actions,
                         const AABB &aabb, const AABB &cent_aabb);
TreeNode *build_tree_lbvh(const std::vector<Object *> &objects,
                          const std::vector<struct Action> &actions,
                          const AABB &aabb, const AABB &cent_aabb);
TreeNode *build_tree_hlbvh(const std::vector<Object *> &objects,
                           const std::vector<struct Action> &actions,
                           const AABB &aabb, const AABB &cent_aabb);
TreeNode *build_tree_agc(const std::vector<Object *> &objects,
                         const std::vector<struct Action> &actions,
                         const AABB &aabb, const AABB &cent_aabb);

void reset_grid_tree(TreeNode *node);
void process_actions(TreeNode *&root, const std::vector<Object *> &objects,
                     const std::vector<struct Action> &actions,
                     const AABB &cent_aabb, int frame, vector<LeafNode *> &dirty_leaves, DirtyKeySet &dirty_keys);
void refit_tree(TreeNode *root, const std::vector<Object *> &objects,
                const std::vector<struct Action> &actions,
                const AABB &aabb, const AABB &cent_aabb);

ReturnRecord build_bvh(TreeNode *node, const AABB &cent_aabb, int level);
__device__ bool find_intersection(TreeNode *root, const Ray &ray, Intersection &itsc);
bool find_any_intersection(TreeNode *root, const Ray &ray, Intersection &itsc);

void destroy_tree(TreeNode *root);
void count_nodes(TreeNode *node,
                 int &ngrids, int &nbranches, int &nleaves,
                 int &ndirtynodes, int &nlfbytes);
float calc_sah_cost(TreeNode *node);
void print_tree(TreeNode *root, int level = 0);
int flatten_node(TreeNode *node, FlattenContext &ctx);
int collect_dirty_leaves(TreeNode *node, vector<LeafNode *> &dirty_leaves);

inline void sort_dirty_leaves_by_grid_code(
    GPU_LeafNode *d_dirty_leaves,
    int num_dirty_leaves,
    cudaStream_t stream = 0);


static std::vector<DirtyKey>
build_sorted_dirty_keys_from_set(const DirtyKeySet& dirty_keys)
{
    std::vector<DirtyKey> out;
    out.reserve(dirty_keys.size());

    for (const auto& k : dirty_keys) {
        out.push_back(DirtyKey{
            k.code,
            k.leaf_code,
            k.bits,
            k.type
        });
    }

    std::sort(out.begin(), out.end(),
        [](const DirtyKey& a, const DirtyKey& b) {
            if (a.type != b.type) return a.type < b.type;
            if (a.bits != b.bits) return a.bits < b.bits;
            if (a.code != b.code) return a.code < b.code;
            return a.leaf_code < b.leaf_code;
        });

    return out;
}

__global__ void kernel_build_initial_clusters_from_leaves(
    const GPU_LeafNode *leaves,
    int num_leaves,
    GPU_Cluster *clusters);

struct DeviceScene;
struct Scene;

__global__ void kernel_set_bvh_root_from_final_cluster(
    DeviceScene *d_scene,
    const GPU_Cluster *d_clusters,
    int n_clusters);

void build_bvh_on_gpu(
    DeviceScene *d_scene,
    DeviceScene &h_scene,
    const std::vector<LeafNode *> &dirty_leaves_cpu,
    const std::vector<DirtyKey> &h_dirty_keys,
    cudaStream_t stream);

void build_initial_bvh_gpu(Scene scene, DeviceScene *d_scene, DeviceScene &h_scene, cudaStream_t stream, const vector<DirtyKey> &h_dirty_keys);

__global__ void init_dirty_leaf_aabbs_kernel(GPU_LeafNode *dirty_leaves, int num_dirty_leaves);

void materialize_curr_bvh_to_self_contained(
    DeviceScene *d_scene,
    DeviceScene &h_scene);
void promote_curr_to_prev(DeviceScene *d_scene, DeviceScene &h_scene);
