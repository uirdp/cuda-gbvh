#include "includes/bvtree.cuh"
#include "includes/object.cuh"
#include "includes/scene.cuh"
#include "includes/external/glm/vec3.hpp"

#include <vector>
#include <alloca.h>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstdint>
#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>

using glm::vec3;
using std::unordered_map;
using std::vector;

void register_triangle_to_leaf(LeafNode *leaf, int i, Object *obj)
{
    Triangle *tri = (Triangle *)obj;
    leaf->triangles[i] = *tri;
}

inline static void add_object_to_leaf(LeafNode *leaf, Object *obj)
{
    leaf->expand();
    register_triangle_to_leaf(leaf, leaf->nobjs, obj);
    leaf->nobjs++;
    leaf->expire = std::min(leaf->expire, obj->expire);
}

void set_aabbs(BVH_Node *node, const AABB &aabb0, const AABB &aabb1)
{
    node->aabbs[0] = aabb0;
    node->aabbs[1] = aabb1;
}

inline void sort_dirty_leaves_by_grid_code(
    GPU_LeafNode *d_dirty_leaves,
    int num_dirty_leaves,
    cudaStream_t stream)
{
    if (d_dirty_leaves == nullptr || num_dirty_leaves <= 1)
        return;

    thrust::device_ptr<GPU_LeafNode> begin(d_dirty_leaves);
    thrust::device_ptr<GPU_LeafNode> end = begin + num_dirty_leaves;

    auto policy = thrust::cuda::par.on(stream);
    thrust::sort(policy, begin, end, LeafLessByGridCode{});
}

int collect_dirty_leaves(TreeNode *node, vector<LeafNode *> &dirty_leaves)
{
    if (!node)
        return 0;

    int count = 0;
    if (node->type == NT_LEAF)
    {
        LeafNode *leaf = (LeafNode *)node;
        if (leaf->is_dirty)
        {
            dirty_leaves.push_back(leaf);
            count = 1;
        }
        // いったんすべてのleafをdirtyとして扱う
        // dirty_leaves.push_back(leaf);
        // count = 1;
    }
    else if (node->type == NT_BRANCH)
    {
        BVH_Node *bvh = (BVH_Node *)node;
        count += collect_dirty_leaves(bvh->left, dirty_leaves);
        count += collect_dirty_leaves(bvh->right, dirty_leaves);
    }
    else if (node->type == NT_GRID)
    {
        GridNode *grid = (GridNode *)node;
        for (int i = 0; i < NDIV * NDIV * NDIV; ++i)
        {
            count += collect_dirty_leaves(grid->cells[i], dirty_leaves);
        }
    }

    return count;
}

inline static void delete_object_from_leaf(LeafNode *leaf, Object *obj)
{
    for (int i = 0; i < leaf->nobjs; i++)
    {
        Object *current_obj = leaf->get_object(i);
        if (current_obj->id == obj->id)
        {
            leaf->nobjs--;
            register_triangle_to_leaf(leaf, i, leaf->get_object(leaf->nobjs));
            return;
        }
    }

    printf("Error: Object to delete not found in leaf node.\n");
}

inline static void delete_expired_objects(LeafNode *Leaf, int frame)
{
    // 後で実装
    printf("delete_expired_objects is not implemented yet.\n");
}

inline static void collect_objects(GridNode *grid, LeafNode *dest)
{
    for (int i = 0; i < NDIV * NDIV * NDIV; i++)
    {
        if (!grid->cells[i])
            continue;

        assert(grid->cells[i]->type == NT_LEAF);
        LeafNode *src = (LeafNode *)grid->cells[i];

        for (int j = 0; j < src->nobjs; j++)
        {
            add_object_to_leaf(dest, src->get_object(j));
        }
    }
}

inline static CodeType get_hgrid_code(CodeType x, CodeType y, CodeType z)
{
    /* this function is originally embree::bitInterleave */
#if NDIV_SHIFT == 2
    if (sizeof(CodeType) == 4)
    {
        x = (x | (x << 16)) & 0x030000FF;
        x = (x | (x << 8)) & 0x0300F00F;
        x = (x | (x << 4)) & 0x030C30C3;

        y = (y | (y << 16)) & 0x030000FF;
        y = (y | (y << 8)) & 0x0300F00F;
        y = (y | (y << 4)) & 0x030C30C3;

        z = (z | (z << 16)) & 0x030000FF;
        z = (z | (z << 8)) & 0x0300F00F;
        z = (z | (z << 4)) & 0x030C30C3;
    }
    else
    {
        x = (x | ((long long)x << 32)) & 0x0FFF00000000FFFFLL;
        x = (x | (x << 16)) & 0x00FF0000FF0000FFLL;
        x = (x | (x << 8)) & 0x000F00F00F00F00FLL;
        x = (x | (x << 4)) & 0x00C30C30C30C30C3LL;

        y = (y | ((long long)y << 32)) & 0x0FFF00000000FFFFLL;
        y = (y | (y << 16)) & 0x00FF0000FF0000FFLL;
        y = (y | (y << 8)) & 0x000F00F00F00F00FLL;
        y = (y | (y << 4)) & 0x00C30C30C30C30C3LL;

        z = (z | ((long long)z << 32)) & 0x0FFF00000000FFFFLL;
        z = (z | (z << 16)) & 0x00FF0000FF0000FFLL;
        z = (z | (z << 8)) & 0x000F00F00F00F00FLL;
        z = (z | (z << 4)) & 0x00C30C30C30C30C3LL;
    }
    return x | (y << 2) | (z << 4);

#elif NDIV_SHIFT == 3
    assert(sizeof(CodeType) == 8);
    x = (x | (x << 24)) & 0x0000FFF000000FFFLL;
    x = (x | (x << 12)) & 0x0FC003F000FC003FLL;
    x = (x | (x << 6)) & 0x01C0E070381C0E07LL;

    y = (y | (y << 24)) & 0x0000FFF000000FFFLL;
    y = (y | (y << 12)) & 0x0FC003F000FC003FLL;
    y = (y | (y << 6)) & 0x01C0E070381C0E07LL;

    z = (z | (z << 24)) & 0x0000FFF000000FFFLL;
    z = (z | (z << 12)) & 0x0FC003F000FC003FLL;
    z = (z | (z << 6)) & 0x01C0E070381C0E07LL;

    return x | (y << 3) | (z << 6);

#else
    fprintf(stderr, "Can't calculate hierarchical grid code");
    exit(1);
#endif
}

static inline void add_dirty_key(DirtyKeySet &dirty_keys, uint64_t code, uint8_t bits)
{
    dirty_keys.insert({code, bits});
}

static inline void add_dirty_grid_key(DirtyKeySet &dirty_keys, GridNode *grid)
{
    if (!grid)
        return;
    add_dirty_key(dirty_keys, grid->grid_code, grid->grid_bits);
}

static inline void add_dirty_leaf_key(DirtyKeySet &dirty_keys, LeafNode *leaf)
{
    if (!leaf)
        return;
    add_dirty_key(dirty_keys, leaf->grid_code, leaf->grid_bits);
}

// -- dirty keys

void insert_object(TreeNode *&node, Object *obj, int glevel, vector<LeafNode *> &dirty_leaves, DirtyKeySet &dirty_keys)
{
    TreeNode **np = &node;

    uint64_t path_code = 0;
    uint8_t path_bits = 0;

    uint64_t parent_code = 0;
    uint8_t parent_bits = 0;

    while (*np && (*np)->type == NT_GRID)
    {
        GridNode *grid = (GridNode *)(*np);
        grid->nobjs++;
        grid->is_dirty = true;
        add_dirty_grid_key(dirty_keys, grid);
        grid->expire = std::min(grid->expire, obj->expire);

        int idx = (obj->code >> ((MAX_GRID_LEVEL - 1 - glevel) *
                                 NDIV_SHIFT * 3)) &
                  IDX_MASK;

        parent_code = path_code;
        parent_bits = path_bits;

        path_code = (path_code << GRID_BITS_PER_LEVEL) | (uint64_t)(idx);
        path_bits += GRID_BITS_PER_LEVEL;

        np = &grid->cells[idx];
        glevel++;
    }

    if (*np == nullptr)
    {
        // デフォルトでdirtyになっているはず
        LeafNode *leaf = new LeafNode();
        add_object_to_leaf(leaf, obj);
        // AIによると修正がいるらしいが、正しいと思う
        // leawf->set_grid_code(parent_code, parent_bits);
        leaf->set_grid_code(path_code, path_bits);
        *np = leaf;
        dirty_leaves.push_back(leaf);
        add_dirty_leaf_key(dirty_keys, leaf);
    }
    else
    {
        LeafNode *leaf = (LeafNode *)*np;
        leaf->is_dirty = true;
        add_dirty_leaf_key(dirty_keys, leaf);
        // leaf->set_grid_code(parent_code, parent_bits);
        if (leaf->nobjs + 1 > MAX_LEAF_SIZE && glevel < MAX_GRID_LEVEL)
        {
            GridNode *grid = new GridNode();
            grid->set_code(path_code, path_bits);
            for (int i = 0; i < leaf->nobjs; i++)
            {
                Object *o = leaf->get_object(i);
                insert_object((TreeNode *&)grid, o, glevel, dirty_leaves, dirty_keys);
            }
            insert_object((TreeNode *&)grid, obj, glevel, dirty_leaves, dirty_keys);
            *np = (TreeNode *)grid;
            if (leaf->bvh_node)
                destroy_tree(leaf->bvh_node);
            delete leaf;
        }
        else
        {
            add_object_to_leaf(leaf, obj);
            dirty_leaves.push_back(leaf);
        }
    }
}

static void delete_object(TreeNode *&node, Object *obj, int glevel, vector<LeafNode *> &dirty_leaves, DirtyKeySet &dirty_keys)
{

    if (node->type == NT_GRID)
    {
        GridNode *grid = (GridNode *)node;
        grid->is_dirty = true;
        add_dirty_grid_key(dirty_keys, grid);
        grid->nobjs--;
        int idx = (obj->code >> ((MAX_GRID_LEVEL - 1 - glevel) *
                                 NDIV_SHIFT * 3)) &
                  IDX_MASK;
        delete_object(grid->cells[idx], obj, glevel + 1, dirty_leaves, dirty_keys);
        if (grid->nobjs == 0)
        {
            if (grid->node_alloc_buf)
                free((void *)grid->node_alloc_buf);
            delete grid;
            node = nullptr;
        }
        else if (grid->nobjs <= MAX_LEAF_SIZE)
        {
            LeafNode *leaf = new LeafNode();
            auto code = (grid->grid_code >> GRID_BITS_PER_LEVEL);
            auto bits = grid->grid_bits - GRID_BITS_PER_LEVEL;
            leaf->set_grid_code(code, bits);

            collect_objects(grid, leaf);
            destroy_tree(grid);
            node = leaf;
            dirty_leaves.push_back(leaf);
            add_dirty_leaf_key(dirty_keys, leaf);
        }
    }
    else
    {
        assert(node->type == NT_LEAF);
        LeafNode *leaf = (LeafNode *)node;
        delete_object_from_leaf(leaf, obj);

        leaf->is_dirty = true;
        add_dirty_leaf_key(dirty_keys, leaf);
        if (leaf->nobjs == 0)
        {
            if (leaf->bvh_node)
                destroy_tree(leaf->bvh_node);

            delete leaf;
            node = nullptr;
        }

        else
        {
            dirty_leaves.push_back(leaf);
        }
    }
}

void reset_grid_tree(TreeNode *node)
{
    if (node == nullptr)
        return;

    if (node->type == NT_GRID)
    {
        GridNode *grid = (GridNode *)node;

        grid->is_dirty = false;

        for (int i = 0; i < NDIV * NDIV * NDIV; i++)
        {
            reset_grid_tree(grid->cells[i]);
        }
    }
    else if (node->type == NT_LEAF)
    {
        LeafNode *leaf = (LeafNode *)node;
        leaf->is_dirty = false;
    }
}

class GridBuilder
{
public:
    struct ObjectInfo
    {
        int obj_id;
        CodeType code;
        operator unsigned CodeType() const { return code; }
    };
    vector<ObjectInfo> obj_infos;
    int del_start;
    int frame;
    const vector<Object *> *objects;

    GridBuilder(const vector<Object *> &_objects, const vector<Action> &actions, const AABB &cent_aabb, int _frame)
        : frame(_frame)
    {
        objects = &_objects;

        obj_infos.resize(actions.size());
        vec3 grid_dim = cent_aabb.vmax - cent_aabb.vmin;
        const int resolution = 1 << NBITS_INT_COORD;
        int ip = 0, dp = actions.size();
        for (int i = 0; i < actions.size(); i++)
        {
            int obj_id = actions[i].obj_id;
#if !USE_EXPIRE
            obj_id ^= (obj_id >> 31); // delete のときに負の値になるので1's complementに変換
#endif
            AABB oaabb = (*objects)[obj_id]->get_aabb();
            vec3 centroid = (oaabb.vmin + oaabb.vmax) * 0.5f;
            vec3 idx = floor((centroid - cent_aabb.vmin) / grid_dim * (float)resolution); // cent_aabb座標系（？）に移して、正規化[0, 1]して解像度で割ることでインデックスを求めている
            idx = glm::clamp(idx, vec3(0.0f), vec3((float)(resolution - 1)));
            CodeType code = get_hgrid_code((CodeType)idx.x, (CodeType)idx.y, (CodeType)idx.z);

            // info bufferに登録する、　insertだったら前から、deleteだったら後ろから
            if (actions[i].obj_id >= 0)
            { // insert
                obj_infos[ip].code = code;
                obj_infos[ip].obj_id = obj_id;
                ip++;
            }
            else
            {
                dp--;
                obj_infos[dp].code = code;
                obj_infos[dp].obj_id = obj_id;
            }
            // printf("Action %d: obj_id=%d, code=0x%llx\n", i, obj_id, (unsigned long long)code);
            (*objects)[obj_id]->code = code;
        }
        del_start = ip;
    }

    void build_serial(TreeNode *&node,
                      int start,
                      int d_start,
                      int end,
                      int glevel,
                      vector<LeafNode *> &dirty_leaves,
                      DirtyKeySet &dirty_keys)
    {

        if (node == nullptr)
        {
            GridNode *root = new GridNode();
            // root->set_code(0, 0);
            node = (TreeNode *)root;
            // node = new GridNode();
        }

        GridNode *grid = (GridNode *)node;
        ObjectInfo *infos = &obj_infos[0];
        // d_start から end までdeleteする
        for (int i = d_start; i < end; i++)
        {
            delete_object(node, (*objects)[infos[i].obj_id], glevel, dirty_leaves, dirty_keys);
        }
        // start から d_start までinsertする

        for (int i = start; i < d_start; i++)
        {
            insert_object(node, (*objects)[infos[i].obj_id], glevel, dirty_leaves, dirty_keys);
        }
    }
};

void process_actions(TreeNode *&node,
                     const vector<Object *> &objects,
                     const vector<Action> &action,
                     const AABB &cent_aabb,
                     int frame,
                     vector<LeafNode *> &dirty_leaves,
                     DirtyKeySet &dirty_keys)
{
    GridBuilder builder(objects, action, cent_aabb, frame);
    builder.build_serial(node, 0, builder.del_start, action.size(), 0, dirty_leaves, dirty_keys); // これだと（多分）nodeがnullなため動かない
}

// agglomerative clustering
ReturnRecord build_fragment_agc(const ListElement *subnodes, int num_subnodes, BVH_Node *&node_alloc_ptr)
{
    if (num_subnodes == 0)
        return ReturnRecord();

    ListElement *buf = (ListElement *)alloca(sizeof(ListElement) * num_subnodes);
    memcpy(buf, subnodes, sizeof(ListElement) * num_subnodes);
#ifdef USE_DIST_CACHE
    float *dist_table = (float *)alloca(sizeof(float) * num_subnodes * num_subnodes);
    memset(dist_table, 0, sizeof(float) * num_subnodes * num_subnodes);
#endif
    int n = num_subnodes;
    while (n > 1)
    {
        float min_dist = FLT_MAX;
        int min_i = -1, min_j = -1;
        for (int i = 0; i < n - 1; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
#ifdef USE_DIST_CACHE
                int tidx = i * num_subnodes + j;
                float dist = dist_table[tidx];
#else
                float dist = 0;
#endif
                if (dist == 0)
                {
                    AABB aabb(buf[i].aabb);
                    aabb.insert(buf[j].aabb);
                    dist = aabb.surface_area();
#ifdef USE_DIST_CACHE
                    dist_table[tidx] = dist;
#endif
                }

                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_i = i;
                    min_j = j;
                }
            }
        }
        assert(min_i != -1 && min_j != -1);

        // merge
        AABB aabb(buf[min_i].aabb);
        aabb.insert(buf[min_j].aabb);
        float cost;
        float s = aabb.surface_area();

        if (s == 0)
        {
            cost = buf[min_i].cost + buf[min_j].cost + BVTREE_SAH_KT;
        }
        else
        {
            cost = (buf[min_i].cost * buf[min_i].aabb.surface_area() +
                    buf[min_j].cost * buf[min_j].aabb.surface_area()) /
                       s +
                   BVTREE_SAH_KT;
        }

        BVH_Node *node;

        int axis = 0;
        if (node_alloc_ptr)
        {
            node = new (node_alloc_ptr) BVH_Node(axis, buf[min_i].node, buf[min_j].node);
            node_alloc_ptr++;
        }
        else
        {
            node = new BVH_Node(axis, buf[min_i].node, buf[min_j].node);
        }

        set_aabbs(node, buf[min_i].aabb, buf[min_j].aabb);

        buf[min_i].node = node;
        buf[min_i].aabb = aabb;
        buf[min_i].cost = cost;
        buf[min_j] = buf[n - 1];
        n--;

#ifdef USE_DIST_CACHE
        for (int i = 0; i < n; i++)
        {
            dist_table[i * num_subnodes + min_i] = 0;
            dist_table[min_i * num_subnodes + i] = 0;
            dist_table[min_j * num_subnodes + i] = 0;
            dist_table[i * num_subnodes + min_j] = dist_table[i * num_subnodes + n];
        }
#endif
    }

    return ReturnRecord(buf[0].node, buf[0].cost, buf[0].aabb);
}

ReturnRecord build_bvh(TreeNode *node, const AABB &cent_aabb, int glevel)
{
    const bool BUILD_TREE_FOR_LEAVES = false;
    const int LEAF_LBVH_THRESHOLD = MAX_LEAF_SIZE * 4;

    if (node == nullptr)
    {
        printf("Node is null in build_bvh\n");
    }

    if (node->type == NT_LEAF)
    {
        LeafNode *leaf = (LeafNode *)node;
        if (leaf->is_dirty)
        {
            // leafのaabbを更新する
            AABB aabb;
            for (int i = 0; i < leaf->nobjs; i++)
            {
                Object *obj = leaf->get_object(i);
                aabb.insert(obj->get_aabb());
            }
            leaf->aabb = aabb.dilate();
            leaf->is_dirty = false;
            return ReturnRecord(leaf, BVTREE_SAH_KI * leaf->nobjs, leaf->aabb);
        }

        return ReturnRecord(leaf, BVTREE_SAH_KI * leaf->nobjs, leaf->aabb);
    }

    assert(node->type == NT_GRID);
    GridNode *grid = (GridNode *)node;

    if (!grid->is_dirty)
    {
        return ReturnRecord(grid->bvh_node, grid->cost, grid->aabb);
    }

    ListElement subnodes[NDIV * NDIV * NDIV];
    int num_subnodes = 0;
    int pos[NDIV * NDIV * NDIV];

    for (int i = 0; i < NDIV * NDIV * NDIV; i++)
    {
        if (grid->cells[i])
            pos[i] = num_subnodes++;
    }

    for (int i = 0; i < NDIV * NDIV * NDIV; i++)
    {
        if (grid->cells[i])
        {
            ReturnRecord rec = build_bvh(grid->cells[i], cent_aabb, glevel + 1);
            subnodes[pos[i]] = ListElement(rec, i);
        }
    }

    grid->is_dirty = false;
    if (grid->node_alloc_buf)
        free((void *)grid->node_alloc_buf);

    BVH_Node *node_alloc_ptr = nullptr;
    if (num_subnodes >= 2)
    {
        node_alloc_ptr = (BVH_Node *)malloc(sizeof(BVH_Node) * (num_subnodes - 1));
        if (!node_alloc_ptr)
            no_memory();
    }

    grid->node_alloc_buf = node_alloc_ptr;
    ReturnRecord rec = build_fragment_agc(subnodes, num_subnodes, node_alloc_ptr);
    ;

    grid->bvh_node = (BVH_Node *)rec.node; // これ
    grid->cost = rec.cost;
    grid->aabb = rec.aabb;

    return rec;
}

__host__ __device__ inline ulonglong2 make_grid_key(uint64_t grid_code, uint8_t grid_bits)
{
    return make_ulonglong2(grid_code, grid_bits);
}

struct KeyFromCluster
{
    __host__ __device__ ulonglong2 operator()(const GPU_Cluster &c) const
    {
        return make_grid_key(c.grid_code, c.grid_bits);
    }
};

struct U128Less
{
    __host__ __device__ bool operator()(const ulonglong2 &a, const ulonglong2 &b) const
    {
        if (a.y != b.y)
            return a.y < b.y; // bits first
        return a.x < b.x;     // then code
    }
};

struct U128Equal
{
    __host__ __device__ bool operator()(const ulonglong2 &a, const ulonglong2 &b) const
    {
        return a.x == b.x && a.y == b.y;
    }
};

// binary search for upper_bound
__device__ __forceinline__ int upper_bound_int(const int *arr, int n, int x)
{
    int l = 0, r = n;
    while (l < r)
    {
        int m = (l + r) >> 1;
        if (arr[m] <= x)
            l = m + 1;
        else
            r = m;
    }
    return l;
}

__device__ inline bool key_less_u128(const ulonglong2 &a, const ulonglong2 &b)
{
    if (a.y != b.y)
        return a.y < b.y; // bits first
    return a.x < b.x;     // then code
}

__device__ inline bool key_equal_u128(const ulonglong2 &a, const ulonglong2 &b)
{
    return a.x == b.x && a.y == b.y;
}

__device__ int lower_bound_u128(const ulonglong2 *arr, int n, const ulonglong2 &key)
{
    int l = 0, r = n;
    while (l < r)
    {
        int m = (l + r) >> 1;
        if (key_less_u128(arr[m], key))
            l = m + 1;
        else
            r = m;
    }
    return l;
}

__device__ bool contains_key_u128(const ulonglong2 *arr, int n, const ulonglong2 &key)
{
    int idx = lower_bound_u128(arr, n, key);
    return idx < n && key_equal_u128(arr[idx], key);
}

__global__ void init_dirty_leaf_aabbs_kernel(GPU_LeafNode *dirty_leaves, int num_dirty_leaves)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dirty_leaves)
        return;

    GPU_LeafNode &leaf = dirty_leaves[idx];

    // CPU側で nullptr leaf を GPU_LeafNode{} として詰めている可能性があるので、
    // tri_count <= 0 は何もしない
    if (leaf.tri_count <= 0)
    {
        return;
    }

    AABB aabb = leaf.triangles[0].get_aabb();

    for (int i = 1; i < leaf.tri_count; ++i)
    {
        aabb.insert(leaf.triangles[i].get_aabb());
    }

    leaf.aabb = aabb.dilate();
}

__global__ void kernel_build_initial_clusters_from_leaves(
    const GPU_LeafNode *leaves,
    int num_leaves,
    GPU_Cluster *clusters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_leaves)
        return;

    const GPU_LeafNode &leaf = leaves[i];

    if (leaf.tri_count <= 0)
    {
        GPU_Cluster c{};
        c.leaf_idx = -1;
        c.node_idx = -1;
        clusters[i] = c;
        return;
    }

    GPU_Cluster c{};
    c.aabb = leaf.aabb;
    c.grid_code = leaf.grid_code;
    c.grid_bits = leaf.grid_bits;
    c.leaf_idx = i;
    c.node_idx = -1;

    clusters[i] = c;
}

__device__ inline GPU_Cluster make_cluster_from_prev_leaf(
    const GPU_LeafNode *prev_leaves,
    int leaf_idx)
{
    GPU_Cluster c{};
    const GPU_LeafNode &leaf = prev_leaves[leaf_idx];
    c.aabb = leaf.aabb;
    c.grid_code = leaf.grid_code;
    c.grid_bits = leaf.grid_bits;
    c.leaf_idx = leaf_idx;
    c.node_idx = -1;
    c.node_source = GPU_NODE_SRC_NONE;
    return c;
}

__device__ inline GPU_Cluster make_cluster_from_prev_node(
    const GPU_BVH_Node *prev_nodes,
    int node_idx)
{
    GPU_Cluster c{};
    const GPU_BVH_Node &node = prev_nodes[node_idx];
    c.aabb = node.aabb;
    c.grid_code = node.grid_code;
    c.grid_bits = node.grid_bits;
    c.leaf_idx = -1;
    c.node_idx = node_idx;
    c.node_source = GPU_NODE_SRC_PREV;
    return c;
}

// affected cluster = dirty leafs + それらの兄弟とその子
__global__ void kernel_expand_frontier_to_affected_clusters(
    const GPU_BVH_Node *prev_nodes,
    const GPU_LeafNode *prev_leaves,

    const int *frontier_cur, // frontier_curには探索中のGPU_Clusterのインデックスが入っている
    int frontier_size,

    const ulonglong2 *dirty_keys,
    int num_dirty_keys,

    int *visited_prev_nodes,
    int *visited_prev_leaves,

    int *frontier_next,
    int *d_frontier_next_size,

    GPU_Cluster *out_clusters,
    int *d_num_out_clusters)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size)
        return;

    int node_idx = frontier_cur[tid];
    const GPU_BVH_Node &node = prev_nodes[node_idx];

    auto process_leaf_child = [&](int leaf_idx)
    {
        if (leaf_idx < 0)
            return;

        ulonglong2 k = make_ulonglong2(
            (unsigned long long)prev_leaves[leaf_idx].grid_code,
            (unsigned long long)prev_leaves[leaf_idx].grid_bits);

        if (contains_key_u128(dirty_keys, num_dirty_keys, k))
            return; // dirtyなleafはすでにaffected clusterに入っているはずなのでスキップ

        if (atomicCAS(&visited_prev_leaves[leaf_idx], 0, 1) == 0)
        { // 最初に訪れたスレッドだけがaffected clusterに追加する
            int out_idx = atomicAdd(d_num_out_clusters, 1);
            out_clusters[out_idx] = make_cluster_from_prev_leaf(prev_leaves, leaf_idx);
        }
    };

    auto process_node_child = [&](int child_node_idx)
    {
        if (child_node_idx < 0)
            return;

        ulonglong2 k = make_ulonglong2(
            (unsigned long long)prev_nodes[child_node_idx].grid_code,
            (unsigned long long)prev_nodes[child_node_idx].grid_bits);

        if (contains_key_u128(dirty_keys, num_dirty_keys, k))
            return; // dirtyなnodeはすでにaffected clusterに入っているはずなのでスキップ

        if (atomicCAS(&visited_prev_nodes[child_node_idx], 0, 1) == 0)
        { // 最初に訪れたスレッドだけがaffected clusterに追加する
            int out_idx = atomicAdd(d_num_out_clusters, 1);
            out_clusters[out_idx] = make_cluster_from_prev_node(prev_nodes, child_node_idx);

            int next_frontier_idx = atomicAdd(d_frontier_next_size, 1);
            frontier_next[next_frontier_idx] = child_node_idx;
        }
    };

    if (node.left_type == GPU_CHILD_LEAF)
        process_leaf_child(node.left_idx);
    else
        process_node_child(node.left_idx);

    if (node.right_type == GPU_CHILD_LEAF)
        process_leaf_child(node.right_idx);
    else
        process_node_child(node.right_idx);
}

// GPU側のPrevNodeをCPUわがに移す（MAPを使うため）
static void download_prev_nodes(
    const DeviceScene &h_scene,
    std::vector<GPU_BVH_Node> &h_prev_nodes)
{
    int num_prev_nodes = 0;
    if (h_scene.num_prev_bvh_nodes)
    {
        CHECK_CUDA(cudaMemcpy(&num_prev_nodes, h_scene.num_prev_bvh_nodes, sizeof(int), cudaMemcpyDeviceToHost));
    }

    h_prev_nodes.resize(num_prev_nodes);
    if (num_prev_nodes > 0)
    {
        CHECK_CUDA(cudaMemcpy(h_prev_nodes.data(), h_scene.prev_bvh_nodes, sizeof(GPU_BVH_Node) * num_prev_nodes, cudaMemcpyDeviceToHost));
    }
}

static void debug_prev_node_key_distribution(
    const std::vector<GPU_BVH_Node> &nodes)
{
    std::map<int, int> bits_count;

    for (const auto &n : nodes)
    {
        bits_count[(int)n.grid_bits]++;
    }

    printf("=== prev node key distribution ===\n");
    for (auto &kv : bits_count)
    {
        printf("bits=%d : nodes=%d\n", kv.first, kv.second);
    }
}

// grid codeとbitsの組み合わせをキーとして、prev_bvh_nodesのインデックスを値とするMAPを作る
static unordered_map<NodeKey, int, NodeKeyHash>
build_prev_node_map(const vector<GPU_BVH_Node> &h_prev_nodes)
{
    unordered_map<NodeKey, int, NodeKeyHash> prev_node_map;
    prev_node_map.reserve(h_prev_nodes.size() * 2);

    for (int i = 0; i < (int)h_prev_nodes.size(); ++i)
    {
        prev_node_map[{h_prev_nodes[i].grid_code, h_prev_nodes[i].grid_bits}] = i;
    }
    return prev_node_map;
}

static std::vector<int> build_unique_prev_parent_roots(
    const std::vector<LeafNode *> &dirty_leaves,
    const std::unordered_map<NodeKey, int, NodeKeyHash> &prev_node_map)
{
    std::vector<int> roots;
    roots.reserve(dirty_leaves.size());

    for (LeafNode *leaf : dirty_leaves)
    {
        if (!leaf)
            continue;

        uint64_t code = leaf->grid_code;
        uint8_t bits = leaf->grid_bits;

        while (true)
        {
            NodeKey key{code, bits};

            auto it = prev_node_map.find(key);
            if (it != prev_node_map.end())
            {
                roots.push_back(it->second);
                break;
            }

            if (bits < GRID_BITS_PER_LEVEL)
                break;

            code >>= GRID_BITS_PER_LEVEL;
            bits -= GRID_BITS_PER_LEVEL;
        }
    }

    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
    return roots;
}

static void upload_root_indices_and_dirty_keys(
    const vector<int> &h_root_indices,
    const vector<ulonglong2> &h_dirty_keys,
    int *&d_root_indices,
    ulonglong2 *&d_dirty_keys)
{
    d_root_indices = nullptr;
    d_dirty_keys = nullptr;

    if (!h_root_indices.empty())
    {
        CHECK_CUDA(cudaMalloc(&d_root_indices, sizeof(int) * h_root_indices.size()));
        CHECK_CUDA(cudaMemcpy(d_root_indices, h_root_indices.data(), sizeof(int) * h_root_indices.size(), cudaMemcpyHostToDevice));
    }

    if (!h_dirty_keys.empty())
    {
        CHECK_CUDA(cudaMalloc(&d_dirty_keys, sizeof(ulonglong2) * h_dirty_keys.size()));
        CHECK_CUDA(cudaMemcpy(d_dirty_keys, h_dirty_keys.data(), sizeof(ulonglong2) * h_dirty_keys.size(), cudaMemcpyHostToDevice));
    }
}

void collect_affected_clusters_from_prev(
    const DeviceScene &h_scene,
    const vector<LeafNode *> &dirty_leaves_cpu,
    const vector<ulonglong2> &h_dirty_keys,
    GPU_Cluster *&d_affected_clusters,
    int &num_affected_clusters,
    cudaStream_t stream = 0)
{

    d_affected_clusters = nullptr;
    num_affected_clusters = 0;

    int num_prev_nodes = 0;
    int num_prev_leaves = h_scene.num_prev_complete_leaves;

    if (h_scene.num_prev_bvh_nodes)
    {
        CHECK_CUDA(cudaMemcpy(&num_prev_nodes, h_scene.num_prev_bvh_nodes, sizeof(int), cudaMemcpyDeviceToHost));
    }

    if (num_prev_nodes <= 0)
        return;

    // CPU側にPrevNodeを移す（MAPを使うため）
    vector<GPU_BVH_Node> h_prev_nodes;
    download_prev_nodes(h_scene, h_prev_nodes);

    debug_prev_node_key_distribution(h_prev_nodes);

    // CPU側で、grid codeとbitsの組み合わせをキーとして、prev_bvh_nodesのインデックスを値とするMAPを作る
    auto prev_node_map = build_prev_node_map(h_prev_nodes);

    // dirty leafの親を前フレームのBVHからとってくる
    vector<int> h_root_indices = build_unique_prev_parent_roots(dirty_leaves_cpu, prev_node_map);

    printf("root_indices=%zu dirty_leaves=%zu dirty_keys=%zu\n",
           h_root_indices.size(),
           dirty_leaves_cpu.size(),
           h_dirty_keys.size());

    if (h_root_indices.empty())
        return;

    // GPUにroot_indicesとdirty_keysをアップロード
    int *d_root_indices = nullptr;
    ulonglong2 *d_dirty_keys = nullptr;

    upload_root_indices_and_dirty_keys(h_root_indices, h_dirty_keys, d_root_indices, d_dirty_keys);

    // frontierを展開してaffected clusterを集めるためのバッファ確保
    int max_out = num_prev_nodes + num_prev_leaves; // 最悪の場合、全てのノードとリーフがaffected clusterになる。GBVH的には最高？
    CHECK_CUDA(cudaMalloc(&d_affected_clusters, sizeof(GPU_Cluster) * max_out));

    int *d_num_out_clusters = nullptr;
    CHECK_CUDA(cudaMalloc(&d_num_out_clusters, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_num_out_clusters, 0, sizeof(int)));

    int *d_frontier_cur = nullptr;
    int *d_frontier_next = nullptr;
    int *d_frontier_next_size = nullptr;
    int *d_visited_prev_nodes = nullptr;
    int *d_visited_prev_leaves = nullptr;

    CHECK_CUDA(cudaMalloc(&d_frontier_cur, sizeof(int) * num_prev_nodes));
    CHECK_CUDA(cudaMalloc(&d_frontier_next, sizeof(int) * num_prev_nodes));
    CHECK_CUDA(cudaMalloc(&d_frontier_next_size, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_visited_prev_nodes, sizeof(int) * num_prev_nodes));
    CHECK_CUDA(cudaMalloc(&d_visited_prev_leaves, sizeof(int) * num_prev_leaves));

    CHECK_CUDA(cudaMemset(d_visited_prev_leaves, 0, sizeof(int) * num_prev_leaves));
    CHECK_CUDA(cudaMemset(d_visited_prev_nodes, 0, sizeof(int) * num_prev_nodes));

    // frontierの初期化
    int frontier_size = (int)h_root_indices.size();
    CHECK_CUDA(cudaMemcpy(d_frontier_cur, d_root_indices, sizeof(int) * frontier_size, cudaMemcpyDeviceToDevice)); // まずはdirty_leavesの親から探索を始める

    // rootはvisitedにマークする（どうせdirtyなのでやらなくてもいいかもしれない）
    {
        vector<int> h_root_marks(num_prev_nodes, 0);
        for (int idx : h_root_indices)
        {
            if (idx >= 0 && idx < num_prev_nodes)
            {
                h_root_marks[idx] = 1;
            }
        }

        CHECK_CUDA(cudaMemcpy(d_visited_prev_nodes, h_root_marks.data(), sizeof(int) * num_prev_nodes, cudaMemcpyHostToDevice));
    }

    // frontierを展開してaffected clusterを集める
    while (frontier_size > 0)
    {
        CHECK_CUDA(cudaMemset(d_frontier_next_size, 0, sizeof(int)));

        int block = 256;
        int grid = (frontier_size + block - 1) / block;

        kernel_expand_frontier_to_affected_clusters<<<grid, block, 0, stream>>>(
            h_scene.prev_bvh_nodes,
            h_scene.prev_complete_leaves,

            d_frontier_cur,
            frontier_size,

            d_dirty_keys,
            (int)h_dirty_keys.size(),

            d_visited_prev_nodes,
            d_visited_prev_leaves,

            d_frontier_next,
            d_frontier_next_size,

            d_affected_clusters,
            d_num_out_clusters);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpy(&frontier_size, d_frontier_next_size, sizeof(int), cudaMemcpyDeviceToHost));

        std::swap(d_frontier_cur, d_frontier_next);
    }

    CHECK_CUDA(cudaMemcpy(&num_affected_clusters, d_num_out_clusters, sizeof(int), cudaMemcpyDeviceToHost));

    printf("collect_affected: roots=%zu dirty_leaves=%zu dirty_keys=%zu\n",
           h_root_indices.size(),
           dirty_leaves_cpu.size(),
           h_dirty_keys.size());

    for (int i = 0; i < std::min<int>(10, h_root_indices.size()); ++i)
    {
        int idx = h_root_indices[i];
        const auto &n = h_prev_nodes[idx];
        printf("root[%d] node_idx=%d bits=%u code=%llu left_type=%d left_idx=%d right_type=%d right_idx=%d\n",
               i,
               idx,
               (unsigned)n.grid_bits,
               (unsigned long long)n.grid_code,
               n.left_type,
               n.left_idx,
               n.right_type,
               n.right_idx);
    }

    for (int i = 0; i < std::min<int>(10, dirty_leaves_cpu.size()); ++i)
    {
        LeafNode *leaf = dirty_leaves_cpu[i];
        if (!leaf)
            continue;

        printf("dirty leaf[%d] bits=%u code=%llu parent_bits=%u parent_code=%llu\n",
               i,
               (unsigned)leaf->grid_bits,
               (unsigned long long)leaf->grid_code,
               (unsigned)(leaf->grid_bits >= GRID_BITS_PER_LEVEL ? leaf->grid_bits - GRID_BITS_PER_LEVEL : 0),
               (unsigned long long)(leaf->grid_code >> GRID_BITS_PER_LEVEL));
    }

    cudaFree(d_frontier_cur);
    cudaFree(d_frontier_next);
    cudaFree(d_frontier_next_size);
    cudaFree(d_visited_prev_nodes);
    cudaFree(d_visited_prev_leaves);
    cudaFree(d_root_indices);
    cudaFree(d_dirty_keys);
    cudaFree(d_num_out_clusters);
}

__global__ void kernel_concat_clusters(
    const GPU_Cluster *dirty_clusters,
    int num_dirty,
    const GPU_Cluster *affected_clusters,
    int num_affected,
    GPU_Cluster *out_clusters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_dirty + num_affected;
    if (i >= total)
        return;

    if (i < num_dirty)
    {
        out_clusters[i] = dirty_clusters[i];
    }
    else
    {
        out_clusters[i] = affected_clusters[i - num_dirty];
    }
}

__device__ inline const GPU_LeafNode *cluster_to_leaf(
    const GPU_Cluster &c,
    const GPU_LeafNode *leaves)
{
    if (c.leaf_idx < 0)
        return nullptr;
    return &leaves[c.leaf_idx];
}

__device__ inline bool cluster_is_leaf(const GPU_Cluster &c)
{
    return c.leaf_idx >= 0;
}

__device__ inline bool cluster_is_node(const GPU_Cluster &c)
{
    return c.node_idx >= 0;
}

__device__ inline const GPU_LeafNode *get_cluster_leaf(const GPU_Cluster &c, const GPU_LeafNode *leaves)
{
    return cluster_is_leaf(c) ? &leaves[c.leaf_idx] : nullptr;
}

__device__ inline const GPU_BVH_Node *get_cluster_node(const GPU_Cluster &c, const GPU_BVH_Node *nodes)
{
    return cluster_is_node(c) ? &nodes[c.node_idx] : nullptr;
}

__global__ void kernel_mark_leaf_clusters(
    const GPU_Cluster *clusters,
    int n,
    int *flags)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    flags[i] = cluster_is_leaf(clusters[i]) ? 1 : 0;
}

__global__ void kernel_gather_affected_leaves(
    const GPU_Cluster *affected_clusters,
    int n,
    const int *flags,
    const int *offsets,
    const GPU_LeafNode *prev_complete_leaves,
    GPU_LeafNode *out_leaves)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    if (flags[i])
    {
        int out_idx = offsets[i];
        int leaf_idx = affected_clusters[i].leaf_idx;
        out_leaves[out_idx] = prev_complete_leaves[leaf_idx];
    }
}

__global__ void kernel_concat_frame_leaves(
    const GPU_LeafNode *dirty_leaves,
    int num_dirty,
    const GPU_LeafNode *affected_leaves,
    int num_affected,
    GPU_LeafNode *out_leaves)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_dirty + num_affected;
    if (i >= total)
        return;

    if (i < num_dirty)
    {
        out_leaves[i] = dirty_leaves[i];
    }
    else
    {
        out_leaves[i] = affected_leaves[i - num_dirty];
    }
}

__global__ void kernel_remap_affected_leaf_cluster_indices(
    GPU_Cluster *affected_clusters,
    int n,
    const int *leaf_flags,
    const int *leaf_offsets,
    int dirty_leaf_base)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    if (leaf_flags[i])
    {
        affected_clusters[i].leaf_idx = leaf_offsets[i] + dirty_leaf_base;
    }
}

void build_frame_leaves(
    DeviceScene *d_scene,
    DeviceScene &h_scene,
    GPU_Cluster *d_affected_clusters,
    int num_affected_clusters,
    cudaStream_t stream = 0)
{
    // ------------------------------------------------------------
    // 0. 前提チェック
    // ------------------------------------------------------------
    if (h_scene.frame_leaves)
    {
        CHECK_CUDA(cudaFree(h_scene.frame_leaves));
        h_scene.frame_leaves = nullptr;
    }
    h_scene.num_frame_leaves = 0;

    int num_dirty_leaves = h_scene.num_dirty_leaves;

    // affected が無いなら frame_leaves = dirty_leaves のコピーでよい
    if (num_affected_clusters <= 0)
    {
        if (num_dirty_leaves > 0)
        {
            CHECK_CUDA(cudaMalloc(&h_scene.frame_leaves,
                                  sizeof(GPU_LeafNode) * num_dirty_leaves));

            CHECK_CUDA(cudaMemcpyAsync(h_scene.frame_leaves,
                                       h_scene.dirty_leaves,
                                       sizeof(GPU_LeafNode) * num_dirty_leaves,
                                       cudaMemcpyDeviceToDevice,
                                       stream));
        }

        h_scene.num_frame_leaves = num_dirty_leaves;

        CHECK_CUDA(cudaMemcpyAsync(d_scene,
                                   &h_scene,
                                   sizeof(DeviceScene),
                                   cudaMemcpyHostToDevice,
                                   stream));
        return;
    }

    // prev_complete_leaves が必要
    if (h_scene.prev_complete_leaves == nullptr || h_scene.num_prev_complete_leaves <= 0)
    {
        // affected cluster があるのに prev leaves が無いのは不整合
        fprintf(stderr, "ERROR: affected clusters exist but prev_complete_leaves is null.\n");
        return;
    }

    // ------------------------------------------------------------
    // 1. affected_clusters の中で leaf cluster を見つける
    // ------------------------------------------------------------
    thrust::device_vector<int> leaf_flags(num_affected_clusters);
    thrust::device_vector<int> leaf_offsets(num_affected_clusters);

    {
        int block = 256;
        int grid = (num_affected_clusters + block - 1) / block;

        kernel_mark_leaf_clusters<<<grid, block, 0, stream>>>(
            d_affected_clusters,
            num_affected_clusters,
            thrust::raw_pointer_cast(leaf_flags.data()));
        CHECK_CUDA(cudaGetLastError());
    }

    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        leaf_flags.begin(), leaf_flags.end(),
        leaf_offsets.begin());

    int num_affected_leaves = thrust::reduce(
        thrust::cuda::par.on(stream),
        leaf_flags.begin(), leaf_flags.end(),
        0,
        thrust::plus<int>());

    // ------------------------------------------------------------
    // 2. affected leaf だけを gather
    // ------------------------------------------------------------
    GPU_LeafNode *d_affected_leaves = nullptr;
    if (num_affected_leaves > 0)
    {
        CHECK_CUDA(cudaMalloc(&d_affected_leaves,
                              sizeof(GPU_LeafNode) * num_affected_leaves));

        int block = 256;
        int grid = (num_affected_clusters + block - 1) / block;

        kernel_gather_affected_leaves<<<grid, block, 0, stream>>>(
            d_affected_clusters,
            num_affected_clusters,
            thrust::raw_pointer_cast(leaf_flags.data()),
            thrust::raw_pointer_cast(leaf_offsets.data()),
            h_scene.prev_complete_leaves,
            d_affected_leaves);
        CHECK_CUDA(cudaGetLastError());
    }

    // ------------------------------------------------------------
    // 3. frame_leaves = dirty_leaves + affected_leaves
    // ------------------------------------------------------------
    int num_frame_leaves = num_dirty_leaves + num_affected_leaves;

    if (num_frame_leaves > 0)
    {
        CHECK_CUDA(cudaMalloc(&h_scene.frame_leaves,
                              sizeof(GPU_LeafNode) * num_frame_leaves));

        int block = 256;
        int grid = (num_frame_leaves + block - 1) / block;

        kernel_concat_frame_leaves<<<grid, block, 0, stream>>>(
            h_scene.dirty_leaves,
            num_dirty_leaves,
            d_affected_leaves,
            num_affected_leaves,
            h_scene.frame_leaves);
        CHECK_CUDA(cudaGetLastError());
    }

    h_scene.num_frame_leaves = num_frame_leaves;

    // ------------------------------------------------------------
    // 4. affected leaf cluster の leaf_idx を frame_leaves 用に振り直す
    //    dirty leaves は先頭に並ぶので base = num_dirty_leaves
    // ------------------------------------------------------------
    if (num_affected_leaves > 0)
    {
        int block = 256;
        int grid = (num_affected_clusters + block - 1) / block;

        kernel_remap_affected_leaf_cluster_indices<<<grid, block, 0, stream>>>(
            d_affected_clusters,
            num_affected_clusters,
            thrust::raw_pointer_cast(leaf_flags.data()),
            thrust::raw_pointer_cast(leaf_offsets.data()),
            num_dirty_leaves);
        CHECK_CUDA(cudaGetLastError());
    }

    // ------------------------------------------------------------
    // 5. 一時バッファ解放
    // ------------------------------------------------------------
    if (d_affected_leaves)
    {
        CHECK_CUDA(cudaFree(d_affected_leaves));
        d_affected_leaves = nullptr;
    }

    // ------------------------------------------------------------
    // 6. DeviceScene を GPU に書き戻す
    // ------------------------------------------------------------
    CHECK_CUDA(cudaMemcpyAsync(d_scene,
                               &h_scene,
                               sizeof(DeviceScene),
                               cudaMemcpyHostToDevice,
                               stream));
}

// //
// static vector<ulonglong2> build_sorted_unique_dirty_keys(
//     const vector<LeafNode*>& dirty_leaves
// )

__device__ inline void set_child_from_cluster(
    const GPU_Cluster &c,
    int &out_idx,
    int &out_type)
{
    if (cluster_is_leaf(c))
    {
        out_idx = c.leaf_idx;
        out_type = GPU_CHILD_LEAF;
    }
    else
    {
        out_idx = c.node_idx;
        out_type = GPU_CHILD_NODE;
    }
}

__global__ void kernel_assign_group_id(
    int n,
    const int *group_offsets,
    int num_groups,
    int *group_id_of_elem)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    int ub = upper_bound_int(group_offsets, num_groups + 1, i);
    int gid = ub - 1;

    if (gid < 0)
        gid = 0;
    if (gid >= num_groups)
        gid = num_groups - 1;
    group_id_of_elem[i] = gid;
}

__host__ __device__ static float merge_cost_surface_area(const AABB &a, const AABB &b)
{
    float xmin = fminf(a.vmin.x, b.vmin.x);
    float ymin = fminf(a.vmin.y, b.vmin.y);
    float zmin = fminf(a.vmin.z, b.vmin.z);

    float xmax = fmaxf(a.vmax.x, b.vmax.x);
    float ymax = fmaxf(a.vmax.y, b.vmax.y);
    float zmax = fmaxf(a.vmax.z, b.vmax.z);

    float dx = xmax - xmin;
    float dy = ymax - ymin;
    float dz = zmax - zmin;

    return 2.0f * (dx * dy + dy * dz + dz * dx);
}

__global__ void kernel_find_nn(
    const GPU_Cluster *clusteres_sorted,
    int n,
    int target_bits,
    const int *group_id_of_elem,
    const int *group_offsets,
    int *out_neightbor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const GPU_Cluster &self = clusteres_sorted[i];

    if ((int)self.grid_bits != target_bits)
    {
        out_neightbor[i] = -1;
        // out_cost[i] = 1e30f;
        return;
    }

    int gid = group_id_of_elem[i];
    int begin = group_offsets[gid];
    int end = group_offsets[gid + 1];

    // if(end - begin > 1){
    //     printf("Group %d: [%d, %d)\n", gid, begin, end);
    // }

    if (end - begin <= 1)
    {
        out_neightbor[i] = -1;
        // out_cost[i] = 1e30f;
        return;
    }

    const AABB a = clusteres_sorted[i].aabb;

    float best = 1e30f;
    int best_j = -1;

    for (int j = begin; j < end; ++j)
    {
        if (j == i)
            continue;
        float c = merge_cost_surface_area(a, clusteres_sorted[j].aabb);
        if (c < best)
        {
            best = c;
            best_j = j;
        }
        // printf("i=%d j=%d cost=%f\n", i, j, c);
    }

    out_neightbor[i] = best_j;
    // out_cost[i] = best;
}

__host__ __device__ inline void lift_to_parent(uint64_t &code, uint8_t &bits)
{
    if (bits >= GRID_BITS_PER_LEVEL)
    {
        code = code >> GRID_BITS_PER_LEVEL;
        bits -= GRID_BITS_PER_LEVEL;
    }
    else
    {
        code = 0;
        bits = 0;
    }
}

__global__ void kernel_lift_all_to_parent(GPU_Cluster *clusters, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    lift_to_parent(clusters[i].grid_code, clusters[i].grid_bits);
}

__global__ void kernel_lift_only_target_bits(
    GPU_Cluster *clusters, int n, uint8_t target_bits)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    if (clusters[i].grid_bits == target_bits)
    {
        lift_to_parent(clusters[i].grid_code, clusters[i].grid_bits);
    }
}

__global__ void kernel_pair_flags(
    int n,
    const int *neighbor,
    int *pair_flag,
    int *partner)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    int j = neighbor[i];
    if (j < 0 || j >= n)
    {
        pair_flag[i] = 0;
        partner[i] = -1;
        return;
    }

    if (neighbor[j] == i)
    {
        if (i < j)
        {
            pair_flag[i] = 1;
            partner[i] = j;
        }
        else
        {
            pair_flag[i] = 0;
            partner[i] = -1;
        }
    }
    else
    {
        pair_flag[i] = 0;
        partner[i] = -1;
    }
}

__global__ void kernel_mark_used(
    int n,
    const int *pair_flag,
    const int *partner,
    int *used)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    if (pair_flag[i])
    {
        int j = partner[i];
        used[i] = 1;

        if (j >= 0 && j < n)
        {
            used[j] = 1;
        }
    }
}

__global__ void kernel_build_next_clusters_no_lift(
    const GPU_Cluster *clusters_sorted,
    int n,
    const int *pair_flag,
    const int *partner,
    const int *pair_offsets,
    const int *unmerged_flag,
    const int *unmerged_offsets,
    int num_pairs,
    GPU_Cluster *clusters_next)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    if (pair_flag[i])
    {
        int out = pair_offsets[i];
        int j = partner[i];

        GPU_Cluster a = clusters_sorted[i];
        GPU_Cluster b = clusters_sorted[j];

        GPU_Cluster m{};
        m.aabb = AABB::merge(a.aabb, b.aabb);

        m.grid_code = a.grid_code; // どっちでもいいはず
        m.grid_bits = a.grid_bits; // どっちでもいいはず

        m.leaf_idx = -1;
        m.node_idx = -1; // placeholder

        clusters_next[out] = m;
        return;
    }

    if (unmerged_flag[i])
    {
        int out = num_pairs + unmerged_offsets[i];
        clusters_next[out] = clusters_sorted[i];
        return;
    }
}

__global__ void kernel_build_next_clusters_with_nodes(
    const GPU_Cluster *clusters_sorted,
    int n,
    const int *pair_flag,
    const int *partner,
    const int *pair_offsets,
    const int *unmerged_flag,
    const int *unmerged_offsets,
    int num_pairs,
    GPU_Cluster *d_next,
    GPU_BVH_Node *d_bvh_nodes,
    int *d_num_bvh_nodes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    if (pair_flag[i])
    {
        int out_idx = pair_offsets[i];
        int j = partner[i];

        const GPU_Cluster &a = clusters_sorted[i];
        const GPU_Cluster &b = clusters_sorted[j];

        int new_node_idx = atomicAdd(d_num_bvh_nodes, 1);

        GPU_BVH_Node node{};
        node.aabb = AABB::merge(a.aabb, b.aabb);

        set_child_from_cluster(a, node.left_idx, node.left_type);
        set_child_from_cluster(b, node.right_idx, node.right_type);

        d_bvh_nodes[new_node_idx] = node;

        GPU_Cluster merged{};
        merged.aabb = node.aabb;
        merged.grid_code = a.grid_code; // どっちでもいいはず
        merged.grid_bits = a.grid_bits; // どっちでもいいはず
        merged.leaf_idx = -1;
        merged.node_idx = new_node_idx;

        d_next[out_idx] = merged;

        if (new_node_idx < 20)
        {
            printf("create node idx=%d bits a=%u b=%u code a=%llu b=%llu\n",
                   new_node_idx,
                   (unsigned)a.grid_bits,
                   (unsigned)b.grid_bits,
                   (unsigned long long)a.grid_code,
                   (unsigned long long)b.grid_code);
        }
        return;
    }

    if (unmerged_flag[i])
    {
        int out_idx = num_pairs + unmerged_offsets[i];
        d_next[out_idx] = clusters_sorted[i];
        return;
    }
}

struct RoundResult
{
    GPU_Cluster *d_next = nullptr;
    int n_next = 0;
    int num_groups = 0;
    int max_group_size = 0;
    int num_pairs = 0;
};

struct IsTargetBitsMax
{
    int target_bits;
    __host__ __device__ int operator()(const thrust::tuple<ulonglong2, int> &t) const
    {
        ulonglong2 key = thrust::get<0>(t);
        int idx = thrust::get<1>(t);
        return (int)key.y == target_bits;
    }
};

struct InvertFlag
{
    __host__ __device__ int operator()(int u) const
    {
        return u ? 0 : 1;
    }
};

struct CheckNN
{
    __host__ __device__ int operator()(int neighbor_idx) const
    {
        return neighbor_idx >= 0;
    }
};

struct IsTargetBits
{
    int target_bits;
    __host__ __device__
    IsTargetBits(int tb) : target_bits(tb) {}

    __host__ __device__ int operator()(const GPU_Cluster &c) const
    {
        return (int)c.grid_bits == target_bits;
    }
};

inline RoundResult agc_round_one_level_no_lift(
    GPU_Cluster *d_clusters,
    int n,
    int target_bits,
    GPU_BVH_Node *d_bvh_nodes,
    int *d_num_bvh_nodes,
    cudaStream_t stream)
{
    RoundResult rr{};
    if (n <= 1)
    {
        rr.d_next = d_clusters;
        rr.n_next = n;
        rr.num_groups = n;
        rr.max_group_size = n;
        rr.num_pairs = 0;
        return rr;
    }

    thrust::device_vector<GPU_Cluster> clusters_sorted(n);
    cudaMemcpyAsync(thrust::raw_pointer_cast(clusters_sorted.data()),
                    d_clusters,
                    sizeof(GPU_Cluster) * n,
                    cudaMemcpyDeviceToDevice,
                    stream);

    thrust::device_vector<ulonglong2> keys(n);
    thrust::transform(thrust::cuda::par.on(stream),
                      clusters_sorted.begin(), clusters_sorted.end(),
                      keys.begin(),
                      KeyFromCluster{});

    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        keys.begin(), keys.end(),
                        clusters_sorted.begin(),
                        U128Less{});

    thrust::device_vector<ulonglong2> unique_keys(n);
    thrust::device_vector<int> counts(n);

    auto new_end = thrust::reduce_by_key(
        thrust::cuda::par.on(stream),
        keys.begin(), keys.end(),
        thrust::make_constant_iterator(1),
        unique_keys.begin(),
        counts.begin(),
        U128Equal{},
        thrust::plus<int>());

    int G = (int)(new_end.first - unique_keys.begin());
    unique_keys.resize(G);
    counts.resize(G);
    rr.num_groups = G;

    // target_bits の max group size を求める
    int max_count_target = 0;
    if (G > 0)
    {
        auto zipped = thrust::make_zip_iterator(thrust::make_tuple(unique_keys.begin(), counts.begin()));
        auto zipped_end = thrust::make_zip_iterator(thrust::make_tuple(unique_keys.end(), counts.end()));
        max_count_target = thrust::transform_reduce(
            thrust::cuda::par.on(stream),
            zipped, zipped_end,
            IsTargetBitsMax{target_bits},
            0,
            thrust::maximum<int>());
    }

    printf("Group count: %d, max group size for target bits (%d): %d\n", G, target_bits, max_count_target);
    rr.max_group_size = max_count_target;

    // group_offsets
    thrust::device_vector<int> group_offsets(G + 1);
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                           counts.begin(), counts.end(),
                           group_offsets.begin());

    cudaMemcpyAsync(thrust::raw_pointer_cast(group_offsets.data()) + G,
                    &n, sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // group_id_of_elem
    thrust::device_vector<int> group_id_of_elem(n);
    {
        int block = 256, grid = (n + block - 1) / block;
        kernel_assign_group_id<<<grid, block, 0, stream>>>(
            n,
            thrust::raw_pointer_cast(group_offsets.data()),
            G,
            thrust::raw_pointer_cast(group_id_of_elem.data()));
    }

    // int target_count = thrust::count_if(
    //     thrust::cuda::par.on(stream),
    //     clusters_sorted.begin(),
    //     clusters_sorted.end(),
    //     IsTargetBits(target_bits)
    // );
    // printf("target_bits=%d target_count=%d / %d\n", target_bits, target_count, n);

    // NN (target_bits only)
    thrust::device_vector<int> neighbor(n);
    {
        int block = 256, grid = (n + block - 1) / block;
        kernel_find_nn<<<grid, block, 0, stream>>>(
            thrust::raw_pointer_cast(clusters_sorted.data()),
            n,
            target_bits,
            thrust::raw_pointer_cast(group_id_of_elem.data()),
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(neighbor.data()));
    }

    // int nn_count = thrust::count_if(thrust::cuda::par.on(stream),
    //                               neighbor.begin(), neighbor.end(),
    //                               CheckNN());
    // printf("nn_count = %d / %d\n", nn_count, n);

    // pair + used
    thrust::device_vector<int> pair_flag(n), partner(n);
    thrust::device_vector<int> used(n);
    thrust::fill(thrust::cuda::par.on(stream), used.begin(), used.end(), 0);

    {
        int block = 256, grid = (n + block - 1) / block;
        kernel_pair_flags<<<grid, block, 0, stream>>>(
            n,
            thrust::raw_pointer_cast(neighbor.data()),
            thrust::raw_pointer_cast(pair_flag.data()),
            thrust::raw_pointer_cast(partner.data()));
        kernel_mark_used<<<grid, block, 0, stream>>>(
            n,
            thrust::raw_pointer_cast(pair_flag.data()),
            thrust::raw_pointer_cast(partner.data()),
            thrust::raw_pointer_cast(used.data()));
    }

    thrust::device_vector<int> pair_offsets(n);
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                           pair_flag.begin(), pair_flag.end(),
                           pair_offsets.begin());

    int num_pairs = thrust::reduce(thrust::cuda::par.on(stream),
                                   pair_flag.begin(), pair_flag.end(),
                                   0, thrust::plus<int>());
    rr.num_pairs = num_pairs;

    // unmerged = (used==0) → target外も全部 unmerged になって素通し
    thrust::device_vector<int> unmerged_flag(n), unmerged_offsets(n);
    thrust::transform(thrust::cuda::par.on(stream),
                      used.begin(), used.end(),
                      unmerged_flag.begin(),
                      InvertFlag{});

    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                           unmerged_flag.begin(), unmerged_flag.end(),
                           unmerged_offsets.begin());

    int num_unmerged = thrust::reduce(thrust::cuda::par.on(stream),
                                      unmerged_flag.begin(), unmerged_flag.end(),
                                      0, thrust::plus<int>());

    int n_next = num_pairs + num_unmerged;

    GPU_Cluster *d_next = nullptr;
    cudaMalloc(&d_next, sizeof(GPU_Cluster) * n_next);

    {
        int block = 256, grid = (n + block - 1) / block;
        kernel_build_next_clusters_with_nodes<<<grid, block, 0, stream>>>(
            thrust::raw_pointer_cast(clusters_sorted.data()),
            n,
            thrust::raw_pointer_cast(pair_flag.data()),
            thrust::raw_pointer_cast(partner.data()),
            thrust::raw_pointer_cast(pair_offsets.data()),
            thrust::raw_pointer_cast(unmerged_flag.data()),
            thrust::raw_pointer_cast(unmerged_offsets.data()),
            num_pairs,
            d_next,
            d_bvh_nodes,
            d_num_bvh_nodes);
    }

    rr.d_next = d_next;
    rr.n_next = n_next;
    return rr;
}

struct StageAResult
{
    GPU_Cluster *d_out = nullptr;
    int n_out = 0;
    int rounds_taken = 0;
};

struct GetBits
{
    __host__ __device__ int operator()(const GPU_Cluster &c) const { return (int)c.grid_bits; }
};

inline int device_max_bits(GPU_Cluster *d_clusters, int n, cudaStream_t stream)
{
    if (n <= 0)
        return 0;
    thrust::device_ptr<GPU_Cluster> d_ptr(d_clusters);
    return thrust::transform_reduce(thrust::cuda::par.on(stream),
                                    d_ptr, d_ptr + n,
                                    GetBits{},
                                    0, thrust::maximum<int>());
}

__global__ void kernel_set_bvh_root_from_final_cluster(
    DeviceScene *d_scene,
    const GPU_Cluster *d_clusters,
    int n_clusters)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    if (n_clusters != 1)
    {
        // 異常系
        printf("Error: expected exactly 1 cluster at the end, got %d\n", n_clusters);
        d_scene->curr_bvh_root_node_idx = -1;
        return;
    }

    const GPU_Cluster &c = d_clusters[0];

    // 今回は最終的に internal node になる前提
    d_scene->curr_bvh_root_node_idx = c.node_idx;
}

// inline StageAResult reduce_same_key_fully_then_lift(
//     GPU_Cluster* d_in,
//     int n_in,
//     cudaStream_t stream = 0,
//     int max_rounds_guard = 64 // safety
// ){
//     StageAResult res{};
//     GPU_Cluster* cur = d_in;
//     int n = n_in;

//     if (n <= 1) {
//         res.d_out = cur;
//         res.n_out = n;
//         res.rounds_taken = 0;
//         return res;
//     }

//     int rounds = 0;

//     while (rounds < max_rounds_guard) {
//         RoundResult rr = agc_round_same_key_no_lift(cur, n, stream);

//         // rr.d_next is new allocation (unless n<=1 case)
//         // free old cur if it was allocated in previous round
//         if (cur != d_in) {
//             cudaFree(cur);
//         }

//         cur = rr.d_next;
//         n   = rr.n_next;
//         rounds++;

//         // If every group has size <= 1, we are fully reduced per key.
//         // Note: after compaction, keys may mix; but the next round sorts anyway.
//         if (rr.max_group_size <= 1) {
//             break;
//         }

//         // If no merges happened in a round (n doesn't shrink), we'd loop forever.
//         // A cheap guard: stop if n stayed same AND max_group_size>1.
//         // (You can improve by checking num_pairs.)
//         // Here we allow another round unless n == previous n and no progress.
//         // We'll rely on max_rounds_guard for safety.
//     }

//     // Now lift all clusters once to parent (THIS is "next stage")
//     {
//         int block=256, grid=(n+block-1)/block;
//         kernel_lift_all_to_parent<<<grid, block, 0, stream>>>(cur, n);
//     }

//     res.d_out = cur;
//     res.n_out = n;
//     res.rounds_taken = rounds;
//     return res;
// }

inline StageAResult reduce_levelwise_merge_then_lift(
    GPU_Cluster *d_in,
    int n_in,
    GPU_BVH_Node *d_bvh_nodes,
    int *d_num_bvh_nodes,
    cudaStream_t stream = 0,
    int max_rounds_guard = 256)
{
    StageAResult res{};
    GPU_Cluster *cur = d_in;
    int n = n_in;
    int rounds = 0;

    if (n <= 1)
    {
        res.d_out = cur;
        res.n_out = n;
        res.rounds_taken = 0;
        return res;
    }

    while (n > 1 && rounds < max_rounds_guard)
    {

        // ターゲットになるレベル（深さ）
        int target_bits = device_max_bits(cur, n, stream);
        if (target_bits == 0 && n <= 1)
        {
            break;
        }

        // printf("[outer] rounds=%d n=%d target_bits=%d\n", rounds, n, target_bits);

        while (rounds < max_rounds_guard)
        {
            int n_before = n;
            // target bitsの深さのクラスターだけをマージする
            RoundResult rr = agc_round_one_level_no_lift(
                cur, n, target_bits,
                d_bvh_nodes, d_num_bvh_nodes,
                stream);

            // printf("num_pairs = %d\n", rr.num_pairs);
            // printf("max_group_size (global) = %d\n", rr.max_group_size);

            if (cur != d_in)
                cudaFree(cur);
            cur = rr.d_next;
            n = rr.n_next;
            rounds++;

            // printf("[inner] target_bits=%d n_before=%d n_after=%d groups=%d max_group=%d pairs=%d\n",
            //         target_bits, n_before, rr.n_next, rr.num_groups, rr.max_group_size, rr.num_pairs);

            if (rr.max_group_size <= 1)
            {
                // printf("Finished merging at bits=%d after %d rounds (n=%d, pairs=%d)\n", target_bits, rounds, n, rr.num_pairs);
                break;
            }

            if (rr.num_pairs == 0)
            {
                // printf("No pairs merged at bits=%d, stopping to avoid infinite loop (n=%d)\n", target_bits, n);
                break;
            }
        }

        // ここが必要：この level を parent に持ち上げる
        {
            int block = 256;
            int grid = (n + block - 1) / block;
            kernel_lift_only_target_bits<<<grid, block, 0, stream>>>(cur, n, target_bits);
            CHECK_CUDA(cudaGetLastError());
        }
    }

    res.d_out = cur;
    res.n_out = n;
    res.rounds_taken = rounds;
    return res;
}

void build_bvh_on_gpu(
    DeviceScene *d_scene,
    DeviceScene &h_scene,
    const std::vector<LeafNode *> &dirty_leaves_cpu,
    const std::vector<ulonglong2> &h_dirty_keys,
    cudaStream_t stream)
{
    // ------------------------------------------------------------
    // 1. dirty clusters を入力の基本とする
    // ------------------------------------------------------------
    GPU_Cluster *d_input_clusters = h_scene.clusters;
    GPU_Cluster *old_clusters = h_scene.clusters;
    int num_input_clusters = h_scene.num_clusters;

    GPU_Cluster *d_affected_clusters = nullptr;
    int num_affected_clusters = 0;

    GPU_Cluster *d_merged_clusters = nullptr;

    // ------------------------------------------------------------
    // 2. prev completed BVH があるなら affected clusters を集める
    // ------------------------------------------------------------
    int num_prev_nodes = 0;
    if (h_scene.num_prev_bvh_nodes)
    {
        CHECK_CUDA(cudaMemcpy(&num_prev_nodes,
                              h_scene.num_prev_bvh_nodes,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    bool has_prev_bvh =
        (h_scene.prev_bvh_nodes != nullptr) &&
        (h_scene.prev_complete_leaves != nullptr) &&
        (num_prev_nodes > 0) &&
        (h_scene.num_prev_complete_leaves > 0);

    printf("has_prev_bvh=%d prev_nodes=%p prev_leaves=%p num_prev_nodes=%d num_prev_leaves=%d\n",
           (int)has_prev_bvh,
           (void *)h_scene.prev_bvh_nodes,
           (void *)h_scene.prev_complete_leaves,
           num_prev_nodes,
           h_scene.num_prev_complete_leaves);

    auto start_time = std::chrono::high_resolution_clock::now();
    if (has_prev_bvh)
    {
        collect_affected_clusters_from_prev(
            h_scene,
            dirty_leaves_cpu,
            h_dirty_keys,
            d_affected_clusters,
            num_affected_clusters,
            stream);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
    printf("Time for collecting affected clusters: %.3f ms\n", elapsed.count());

    build_frame_leaves(
        d_scene,
        h_scene,
        d_affected_clusters,
        num_affected_clusters,
        stream);

    // ------------------------------------------------------------
    // 3. dirty + affected を連結
    // ------------------------------------------------------------
    if (num_affected_clusters > 0)
    {
        int total_clusters = num_input_clusters + num_affected_clusters;

        CHECK_CUDA(cudaMalloc(&d_merged_clusters,
                              sizeof(GPU_Cluster) * total_clusters));

        int block = 256;
        int grid = (total_clusters + block - 1) / block;

        kernel_concat_clusters<<<grid, block, 0, stream>>>(
            h_scene.clusters,
            num_input_clusters,
            d_affected_clusters,
            num_affected_clusters,
            d_merged_clusters);
        CHECK_CUDA(cudaGetLastError());

        d_input_clusters = d_merged_clusters;
        num_input_clusters = total_clusters;
    }

    printf("Total input clusters for AGC: %d (dirty: %d, affected: %d)\n",
           num_input_clusters, h_scene.num_dirty_leaves, num_affected_clusters);

    // ------------------------------------------------------------
    // 4. curr_bvh_nodes を total cluster 数に合わせて再確保
    //    internal node 最大数は (num_input_clusters - 1)
    // ------------------------------------------------------------
    if (h_scene.curr_bvh_nodes)
    {
        CHECK_CUDA(cudaFree(h_scene.curr_bvh_nodes));
        h_scene.curr_bvh_nodes = nullptr;
    }

    if (h_scene.num_curr_bvh_nodes)
    {
        CHECK_CUDA(cudaFree(h_scene.num_curr_bvh_nodes));
        h_scene.num_curr_bvh_nodes = nullptr;
    }

    if (num_input_clusters >= 2)
    {
        CHECK_CUDA(cudaMalloc(&h_scene.curr_bvh_nodes,
                              sizeof(GPU_BVH_Node) * (num_input_clusters - 1)));
    }
    else
    {
        h_scene.curr_bvh_nodes = nullptr;
    }

    CHECK_CUDA(cudaMalloc(&h_scene.num_curr_bvh_nodes, sizeof(int)));
    CHECK_CUDA(cudaMemsetAsync(h_scene.num_curr_bvh_nodes, 0, sizeof(int), stream));

    // DeviceScene に最新ポインタを書き戻しておく
    CHECK_CUDA(cudaMemcpyAsync(d_scene,
                               &h_scene,
                               sizeof(DeviceScene),
                               cudaMemcpyHostToDevice,
                               stream));

    // ------------------------------------------------------------
    // 5. cluster が 0 / 1 / 2+ の場合分け
    // ------------------------------------------------------------
    if (num_input_clusters <= 0)
    {
        h_scene.curr_bvh_root_node_idx = -1;
        CHECK_CUDA(cudaMemcpyAsync(d_scene,
                                   &h_scene,
                                   sizeof(DeviceScene),
                                   cudaMemcpyHostToDevice,
                                   stream));
    }
    else if (num_input_clusters == 1)
    {
        // root が leaf cluster のみになるケース
        // 今の DeviceScene は node root しか持っていないので暫定的に -1
        // 必要なら将来 root_type / root_leaf_idx を追加する
        h_scene.curr_bvh_root_node_idx = -1;
        CHECK_CUDA(cudaMemcpyAsync(d_scene,
                                   &h_scene,
                                   sizeof(DeviceScene),
                                   cudaMemcpyHostToDevice,
                                   stream));
    }
    else
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        // --------------------------------------------------------
        // 6. AGC 実行
        // --------------------------------------------------------
        StageAResult res = reduce_levelwise_merge_then_lift(
            d_input_clusters,
            num_input_clusters,
            h_scene.curr_bvh_nodes,
            h_scene.num_curr_bvh_nodes,
            stream);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        printf("Time for AGC reduction: %.3f ms (rounds taken: %d)\n", elapsed.count(), res.rounds_taken);

        // root を DeviceScene に書く
        kernel_set_bvh_root_from_final_cluster<<<1, 1, 0, stream>>>(
            d_scene,
            res.d_out,
            res.n_out);
        CHECK_CUDA(cudaGetLastError());

        // host mirror 側にも root を反映
        GPU_Cluster h_final_cluster{};
        CHECK_CUDA(cudaMemcpy(&h_final_cluster,
                              res.d_out,
                              sizeof(GPU_Cluster),
                              cudaMemcpyDeviceToHost));

        if (res.n_out == 1 && h_final_cluster.node_idx >= 0)
        {
            h_scene.curr_bvh_root_node_idx = h_final_cluster.node_idx;
        }
        else
        {
            h_scene.curr_bvh_root_node_idx = -1;
        }

        // clusters / num_clusters も最新に更新
        h_scene.clusters = res.d_out;
        h_scene.num_clusters = res.n_out;

        CHECK_CUDA(cudaMemcpyAsync(d_scene,
                                   &h_scene,
                                   sizeof(DeviceScene),
                                   cudaMemcpyHostToDevice,
                                   stream));

        // old dirty clusters を解放
        if (old_clusters && old_clusters != res.d_out && old_clusters != d_merged_clusters)
        {
            CHECK_CUDA(cudaFree(old_clusters));
        }

        // merged 一時配列を解放
        if (d_merged_clusters && d_merged_clusters != res.d_out)
        {
            CHECK_CUDA(cudaFree(d_merged_clusters));
        }
    }

    // ------------------------------------------------------------
    // 7. 一時バッファ解放
    // ------------------------------------------------------------
    if (d_affected_clusters)
    {
        CHECK_CUDA(cudaFree(d_affected_clusters));
        d_affected_clusters = nullptr;
    }

    if (d_merged_clusters)
    {
        // res.d_out が d_merged_clusters を指している可能性があるなら
        // ここで即 free は危険。
        // 今の reduce_levelwise_merge_then_lift は途中で新しい配列を作るので、
        // 最終的に res.d_out が別配列になる想定なら free 可。
        // 安全側に倒してここでは解放しない方がよい。
        //
        CHECK_CUDA(cudaFree(d_merged_clusters));
    }

    CHECK_CUDA(cudaGetLastError());
}

void promote_curr_to_prev(DeviceScene &h_scene)
{
    // old prev を解放
    if (h_scene.prev_bvh_nodes)
    {
        CHECK_CUDA(cudaFree(h_scene.prev_bvh_nodes));
    }
    if (h_scene.prev_complete_leaves)
    {
        CHECK_CUDA(cudaFree(h_scene.prev_complete_leaves));
    }
    if (h_scene.num_prev_bvh_nodes)
    {
        CHECK_CUDA(cudaFree(h_scene.num_prev_bvh_nodes));
    }

    // curr → prev に移す
    h_scene.prev_bvh_nodes = h_scene.curr_bvh_nodes;
    h_scene.num_prev_bvh_nodes = h_scene.num_curr_bvh_nodes;
    h_scene.prev_bvh_root_node_idx = h_scene.curr_bvh_root_node_idx;

    h_scene.prev_complete_leaves = h_scene.frame_leaves;
    h_scene.num_prev_complete_leaves = h_scene.num_frame_leaves;

    // curr をリセット（重要）
    h_scene.curr_bvh_nodes = nullptr;
    h_scene.num_curr_bvh_nodes = nullptr;
    h_scene.curr_bvh_root_node_idx = -1;

    h_scene.frame_leaves = nullptr;
    h_scene.num_frame_leaves = 0;
}
