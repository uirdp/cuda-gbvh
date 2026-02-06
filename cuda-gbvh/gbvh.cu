#include "includes/bvtree.cuh"
#include "includes/object.cuh"
#include "includes/scene.cuh"
#include "includes/external/glm/vec3.hpp"

#include <vector>
#include <alloca.h>

using glm::vec3;
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

inline static void delete_object_from_leaf(LeafNode *leaf, Object *obj)
{
    for (int i = 0; i < leaf->nobjs; i++)
    {
        Object* current_obj = leaf->get_object(i);
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

inline static void collect_objects(GridNode *grid, LeafNode* dest)
{
    for (int i = 0; i < NDIV*NDIV*NDIV; i++) {
        if (!grid->cells[i]) continue;

        assert(grid->cells[i]->type == NT_LEAF);
        LeafNode *src = (LeafNode*)grid->cells[i];

        for (int j = 0; j < src->nobjs; j++) {
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

void insert_object(TreeNode *&node, Object *obj, int glevel)
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
        leaf->set_grid_code(parent_code, parent_bits);
        *np = leaf;
        // dirty_leaves.push_back(leaf);
    }
    else
    {
        LeafNode *leaf = (LeafNode *)*np;
        leaf->is_dirty = true;
        // leaf->set_grid_code(parent_code, parent_bits);
        if (leaf->nobjs + 1 > MAX_LEAF_SIZE && glevel < MAX_GRID_LEVEL)
        {
            GridNode *grid = new GridNode();
            grid->set_code(path_code, path_bits);
            for (int i = 0; i < leaf->nobjs; i++)
            {
                Object *o = leaf->get_object(i);
                insert_object((TreeNode*&)grid, o, glevel);
            }
            insert_object((TreeNode*&)grid, obj, glevel);
            *np = (TreeNode*)grid;
            if (leaf->bvh_node)
                destroy_tree(leaf->bvh_node);
            delete leaf;
        }
        else
        {
            add_object_to_leaf(leaf, obj);
            // dirty_leaves.push_back(leaf);
        }
    }
}

static void delete_object(TreeNode *&node, Object *obj, int glevel)
{

    if (node->type == NT_GRID)
    {
        GridNode *grid = (GridNode *)node;
        grid->is_dirty = true;
        grid->nobjs--;
        int idx = (obj->code >> ((MAX_GRID_LEVEL - 1 - glevel) *
                                 NDIV_SHIFT * 3)) &
                  IDX_MASK;
        delete_object(grid->cells[idx], obj, glevel + 1);
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
            // dirty_leaves.push_back(leaf);
        }
    }
    else
    {
        assert(node->type == NT_LEAF);
        LeafNode *leaf = (LeafNode *)node;
        delete_object_from_leaf(leaf, obj);

        leaf->is_dirty = true;
        if (leaf->nobjs == 0)
        {
            if (leaf->bvh_node)
                destroy_tree(leaf->bvh_node);

            delete leaf;
            node = nullptr;
        }

        // else
        // {
        //     // dirty_leaves.push_back(leaf);
        // }
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
        printf("objects size=%lu, actions size=%lu\n", objects->size(), actions.size());
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
                      int glevel
                      )
    {

        if (node == nullptr)
        {
           GridNode* root = new GridNode();
           // root->set_code(0, 0); 
           node = (TreeNode*)root;
            // node = new GridNode();
        }

        GridNode *grid = (GridNode *)node;
        ObjectInfo *infos = &obj_infos[0];
        // d_start から end までdeleteする
        printf("Deleting objects from %d to %d\n", d_start, end);
        for (int i = d_start; i < end; i++)
        {
            delete_object(node, (*objects)[infos[i].obj_id], glevel);
        }
        // start から d_start までinsertする

        printf("Inserting objects from %d to %d\n", start, d_start);
        for (int i = start; i < d_start; i++)
        {
            insert_object(node, (*objects)[infos[i].obj_id], glevel);
        }
    }
};

void process_actions(TreeNode *&node,
                     const vector<Object *> &objects,
                     const vector<Action> &action,
                     const AABB &cent_aabb,
                     int frame,
                     vector<LeafNode*>& dirty_leaves)
{
    GridBuilder builder(objects, action, cent_aabb, frame);
    builder.build_serial(node, 0, builder.del_start, action.size(), 0); // これだと（多分）nodeがnullなため動かない
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