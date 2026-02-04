#include "includes/bvtree.cuh"

#define  SIZE_MARGIN  10




int flatten_node(TreeNode* node, FlattenContext& ctx){
    int node_idx = ctx.nodes.size();
    ctx.nodes.emplace_back();

    if(node->type == NT_LEAF){
        LeafNode* leaf = (LeafNode*)node;

        GPU_LeafNode gpu_leaf;
        gpu_leaf.aabb = leaf->aabb;
        gpu_leaf.tri_offset = ctx.triangles.size();
        gpu_leaf.tri_count = leaf->nobjs;

        // printf("Triangle offset: %d, count: %d\n", gpu_leaf.tri_offset, gpu_leaf.tri_count);
        // printf("triangle size before adding: %zu\n", ctx.triangles.size());

        for(int i = 0; i < leaf->nobjs; ++i){
            ctx.triangles.push_back(leaf->triangles[i]);
        }

        int leaf_idx = ctx.leaves.size();
        ctx.leaves.push_back(gpu_leaf);

        ctx.nodes[node_idx] = {
            .aabb = leaf->aabb,
            .left = -1,
            .right = -1,
            .leaf = leaf_idx
        };
    } else if(node->type == NT_BRANCH){
        BVH_Node* bvh = (BVH_Node*)node;

        int left = flatten_node(bvh->left, ctx);
        int right = flatten_node(bvh->right, ctx);

        ctx.nodes[node_idx] = {
            .aabb = AABB::merge(bvh->aabbs[0], bvh->aabbs[1]),
            .left = left,
            .right = right,
            .leaf = -1
        };
    } else {
        GridNode* grid = (GridNode*)node;
        if (grid->bvh_node) {
            return flatten_node(grid->bvh_node, ctx);
        } else {
            return -1;
        }
    }

    return node_idx;
} 

void destroy_tree(TreeNode *node){
     if( !node ) return;
    if( node->type == NT_GRID ) {
	for( int i = 0; i < NDIV*NDIV*NDIV; i++ ) {
	    destroy_tree(((GridNode*)node)->cells[i]);
	}
	if( ((GridNode*)node)->node_alloc_buf )
	    free((void*) ((GridNode*)node)->node_alloc_buf);
	delete node;
    } else if( node->type == NT_BRANCH ) {
	destroy_tree(((BVH_Node*)node)->left);
	destroy_tree(((BVH_Node*)node)->right);
	delete node;
    } else {
	if( ((LeafNode*)node)->bvh_node ) destroy_tree(((LeafNode*)node)->bvh_node);
	delete (LeafNode*) node;
    }
}


