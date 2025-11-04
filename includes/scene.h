#pragma once

#include "glm/vec3.hpp"
#include <map>

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
    
}
