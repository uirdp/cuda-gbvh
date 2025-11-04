#pragma once

#include "object.h"
#include "embree/kernels/bvh/bvh.h"
#include "embree/common/math/vec3fa.h"
#include "embree/kernels/geometry/triangle.h"
#include "aabb.h"
#include "glm/vec3.hpp"

#include <limits>

#define BVTREE_MAX_LEVEL 30   // Tree の最大深さ
#define BVTREE_SAH_KB    1.0  // SAHにおけるAABB交差判定コストの定数
#define BVTREE_SAH_KT    1    // SAHにおけるトラバーサルコストの定数
#define BVTREE_SAH_KI    0.2  // SAHにおける交差計算コストの定数

// GBVHでNDIV_SHIFTを3にするときには、MAX_GRID_LEVELを3にするか、CodeTypeを拡張する。
// HLBVHで分割数を増やすときは、NDIV_SHIFTを増やし、NBITS_INT_COORDは10に固定。
#define NDIV_SHIFT   2
//#define NDIV_SHIFT   3
#define NDIV     (1 << NDIV_SHIFT)
#define NDIV_MASK  ((1 << NDIV_SHIFT) - 1)
#define IDX_MASK  ((1 << (NDIV_SHIFT * 3)) - 1)

#define MAX_GRID_LEVEL   5   /* 生成されるBVHの高さは最大で (MAX_GRID_LEVEL *(NDIV - 1) * 3) になる */
#define NBITS_INT_COORD   (MAX_GRID_LEVEL * NDIV_SHIFT)
//#define NBITS_INT_COORD   10
#define CODE_LENGTH   (NBITS_INT_COORD * 3)

#define EXT_GRID_LEVEL  3
#define EXT_NBITS_INT_COORD   (EXT_GRID_LEVEL * NDIV_SHIFT)
#define EXT_CODE_LENGTH  (EXT_NBITS_INT_COORD * 3)

#define MAX_LEAF_SIZE  8    // must be a multiple of 4
#define LEAF_BUF_SIZE (MAX_LEAF_SIZE / 4)

#define NT_LEAF  0
#define NT_BRANCH  1
#define NT_GRID  2

struct TreeNode{
    int type;
};

// 中間ノード
struct BVH_Node : public TreeNode {
    int axis; // 分割軸
    embree::BVH4::AlignedNode aabbs;
    TreeNode *left, *right;
    BVH_Node(int axis, TreeNode *left, TreeNode *right)
        : axis(axis), left(left), right(right) {
        type = NT_BRANCH;
    }
};

struct GridNodeBase : public TreeNode {
    int nobjs;
    int expire;
    GridNodeBase() : nobjs(0), expire(INT_MAX) {}
};

// グリッドノード
struct GridNode : public GridNodeBase {
    TreeNode* cells[NDIV*NDIV*NDIV];
    AABB aabb;
    BVH_Node *bvh_node;
    BVH_Node *node_alloc_buf;
    /* bvh_node と bvh_alloc_buf を１つにすることも考えたが、子が１つしかない
       グリッドノードのときに、２重freeの問題があったのでやめた */
    float cost;
    bool is_dirty;

public:
    GridNode() : node_alloc_buf(nullptr), is_dirty(true){
        type = NT_GRID;
        memset((void*)cells, 0, sizeof(TreeNode*) * NDIV * NDIV * NDIV);
    }

    static int get_index(int x, int y, int z) { return (z* NDIV + y) * NDIV + x; }
    static int get_index(const glm::ivec3& idx) { return (idx.z * NDIV + idx.y) * NDIV + idx.x; }
};

// 葉ノード
struct LeafNode : public GridNodeBase {
    embree::Triangle4 *triangles;
    embree::Triangle4 triangles_buf[LEAF_BUF_SIZE];
    AABB aabb; // GBVHのみで使用
    BVH_Node *bvh_node; // GBVHで葉をBVH木に分解したときにその根が入る
    bool is_dirty;
    int allocated; /* nobjsがMAX_LEAF_SIZEを超える場合、mallocで割りつけら
		       れた Triangle4 の数が入る。 */

    LeafNode() : is_dirty(true), allocated(0), bvh_node(nullptr){
        type = NT_LEAF;
        triangles = this->triangles_buf;
    }

}