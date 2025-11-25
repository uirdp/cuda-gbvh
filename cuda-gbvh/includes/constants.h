#pragma once

#define RT_BUFLEN 1024
#define RT_ERROR  (-1)

#define EPSILON_AABB 1e-5f

#define BUILD_TREE_GBVH  0
#define BUILD_TREE_GBVH_REBUILD  1
#define BUILD_TREE_BIN   2
#define BUILD_TREE_LBVH  3
#define BUILD_TREE_HLBVH 4
#define BUILD_TREE_AGC   5

#ifndef DO_REFIT
#define DO_REFIT  0   /* DO_REFIT=1 のときは、USE_EXPIRE=0 が必要。また、BUILD_TREE_GBVH, BUILD_TREE_GBVH_REBUILD は使用不可 */
#endif