#pragma once

#include "glm/vec3.hpp"
#include "object.h"
#include "aabb.h"
#include "bvtree.h"
#include <map>
#include <vector>
#include <string>

struct Action {
    int obj_id; // 正のときは insert、負(1's complement)のときは delete
    Action() {}
    Action(int id) : obj_id(id) {}
};

struct ViewAction {
    enum ActionType{ VIEWPOINT, LOOKAT, VIEWUP } type;
    glm::vec3 value;

    ViewAction() {}
    ViewAction(ActionType t, const glm::vec3& v) : type(t), value(v) {}
};

struct Scene {
    std::vector<Object*> objects;
    std::map<std::string, Material*> materials;
    std::map<std::string, Texture*> textures;
    TreeNode *bvtree_root;
    TreeNode *grid_root;
    void *tree_buffer;

    std::vector<std::vector<Action>> scenario;
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
};
