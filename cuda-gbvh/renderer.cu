#include "includes/renderer.cuh"

__global__ void render_image(vec3* framebuffer, int image_width, int image_height, CameraParameter cam_params, DeviceScene* d_scene, int frame){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if(x >= image_width || y >= image_height) return;

    float u = float(x) / float(image_width);
    float v = float(y) / float(image_height);

    vec3 direction = glm::normalize(cam_params.lower_left_corner + u * cam_params.horizontal + v * cam_params.vertical - cam_params.origin);
    Ray ray(cam_params.origin, direction);

    int num_tris = d_scene->num_actions_at_frame[frame];

    vec3 color = raycast(ray, d_scene, frame);

    int pixel_index = frame * image_width * image_height + y * image_width + x;  // framebuffer[frame][x][y]
    framebuffer[pixel_index] = color;
}

void export_to_ppm(string filename, vec3* framebuffers, int image_width, int image_height, int num_frames){

    for(int f = 0; f < num_frames - 1; f++){
        string frame_filename = filename + "_" + std::to_string(f) + ".ppm";
        FILE* fp = fopen(frame_filename.c_str(), "wb");
        if(!fp){
            fprintf(stderr, "Error: Could not open file %s for writing.\n", frame_filename.c_str());
            return;
        }

        fprintf(fp, "P6\n%d %d\n255\n", image_width, image_height);

        for(int j = image_height - 1; j >= 0; j--){
            std::clog << "Writing frame " << f << ", scanline " << (image_height - j) << " / " << image_height << "\r";
            for(int i = 0; i < image_width; i++){
                int pixel_index = f * image_width * image_height + j * image_width + i;
                unsigned char r = static_cast<unsigned char>(255.99f * glm::clamp(framebuffers[pixel_index].r, 0.0f, 1.0f));
                unsigned char g = static_cast<unsigned char>(255.99f * glm::clamp(framebuffers[pixel_index].g, 0.0f, 1.0f));
                unsigned char b = static_cast<unsigned char>(255.99f * glm::clamp(framebuffers[pixel_index].b, 0.0f, 1.0f));
                fwrite(&r, 1, 1, fp);
                fwrite(&g, 1, 1, fp);
                fwrite(&b, 1, 1, fp);
            }
        }
        fclose(fp);
    }

  
}

void print_frame_buffer(vec3* framebuffer, int image_width, int image_height, int num_frames){
    for(int f = 0; f < num_frames; f++){
        printf("Frame %d:\n", f);
        for(int j = image_height - 1; j >= 0; j--){
            for(int i = 0; i < image_width; i++){
                int pixel_index = f * image_width * image_height + j * image_width + i;
                vec3 color = framebuffer[pixel_index];
                printf("Pixel (%d, %d): Color (%.2f, %.2f, %.2f)\n", i, j, color.r, color.g, color.b);
            }
        }
    }
}

void print_one_frame_buffer(vec3* framebuffer, int image_width, int image_height, int frame){
    printf("Frame %d:\n", frame);
    for(int j = image_height - 1; j >= 0; j--){
        for(int i = 0; i < image_width; i++){
            int pixel_index = frame * image_width * image_height + j * image_width + i;
            vec3 color = framebuffer[pixel_index];
            printf("Pixel (%d, %d): Color (%.2f, %.2f, %.2f)\n", i, j, color.r, color.g, color.b);
        }
    }
}

__device__ vec3 closest_hit(Ray ray, Intersection isect){
    vec3 white = vec3(1.0f, 1.0f, 1.0f);
    vec3 light_pos = vec3(5.0f, 5.0f, -5.0f);

    vec3 color = white * glm::max(0.0f, glm::dot(normalize(light_pos), isect.obj->get_normal_at_intersection(ray, isect.t, isect.uv)));
    return isect.obj->get_normal_at_intersection(ray, isect.t, isect.uv) * 0.5f + vec3(0.5f, 0.5f, 0.5f);
}

__device__ bool intersect_leaf(
    const GPU_LeafNode& leaf,
    const Ray& ray,
    Intersection& itsc,
    float t_min,
    float& closest_t
){
    bool hit_any = false;

    for(int i = 0; i < leaf.tri_count; ++i){
        const Triangle* tri = &leaf.triangles[i];

        Intersection cand;
        if(intersect_triangle(ray, tri, cand, t_min, closest_t)){
            closest_t = cand.t;
            itsc = cand;
            hit_any = true;
        }
    }

    return hit_any;
}

__device__ bool find_intersection_bvh_cpu(
    const DeviceScene* d_scene,
    const Ray& ray,
    Intersection& itsc,
    float t_min,
    float t_max
){
    const GPU_BVH_Node* nodes  = d_scene->bvh_nodes;
    const GPU_LeafNode* leaves = d_scene->bvh_leaves;
    const Triangle* triangles = d_scene->triangles;

    int stack[64];   // 深さは log2(N) 程度 → 固定長でOK
    int sp = 0;

    bool hit_any = false;
    float closest_t = t_max;

    int root = d_scene->bvh_root;
    if (root < 0) return false;

    stack[sp++] = root;

    while (sp > 0) {
        if (sp >= 64) return hit_any;  // とりあえず溢れたら打ち切り
        int node_idx = stack[--sp];
        const GPU_BVH_Node& node = nodes[node_idx];

        // AABB カリング
        if (!intersect_aabb(node.aabb, ray, t_min, closest_t))
            continue;

        // =========================
        //          LEAF
        // =========================
        if (node.leaf >= 0) {
            const GPU_LeafNode& leaf = leaves[node.leaf];

            for (int i = 0; i < leaf.tri_count; ++i) {
                // if(d_scene->num_triangles <= leaf.tri_offset + i){
                //     printf("Warning: Triangle index out of bounds: %d (num_triangles: %d)\n", leaf.tri_offset + i, d_scene->num_triangles);
                //     continue;
                // }
                const Triangle* tri =
                    &triangles[leaf.tri_offset + i]; // indexを0にするとエラーが起きないので、ここが原因

                Intersection cand;
                if (intersect_triangle(ray, tri, cand, t_min, closest_t)) {
                    closest_t = cand.t;
                    itsc = cand;
                    hit_any = true;
                }
            }
        }
        // =========================
        //        INTERNAL
        // =========================
        else {
            // left
            if (node.left >= 0)
                stack[sp++] = node.left;

            // right
            if (node.right >= 0)
                stack[sp++] = node.right;
        }
    }

    return hit_any;
}

__device__ bool find_intersection_bvh(
    const DeviceScene* d_scene,
    const Ray& ray,
    Intersection& itsc,
    float t_min,
    float t_max
){
    const GPU_BVH_Node* nodes  = d_scene->bvh_nodes;
    const GPU_LeafNode* leaves = d_scene->dirty_leaves;

    int stack[64];
    int sp = 0;

    bool hit_any = false;
    float closest_t = t_max;

    int root = d_scene->bvh_root_node_idx;
    // printf("Starting BVH traversal from root node %d\n", root);
    if(root < 0) return false;

    stack[sp++] = root;

    while(sp > 0){
        if(sp >= 64) return hit_any;  // 念のため

        int node_idx = stack[--sp];
        const GPU_BVH_Node& node = nodes[node_idx];

        // node AABB で cull
        if(!intersect_aabb(node.aabb, ray, t_min, closest_t))
            continue;

        // -------------------------
        // left child
        // -------------------------
        if(node.left_type == GPU_CHILD_LEAF){
            // printf("Checking leaf node %d\n", node.left_idx);
            if(node.left_idx >= 0){
                const GPU_LeafNode& leaf = leaves[node.left_idx];

                // leaf AABB があるならここでも cull できる
                if(intersect_aabb(leaf.aabb, ray, t_min, closest_t)){
                    if(intersect_leaf(leaf, ray, itsc, t_min, closest_t)){
                        hit_any = true;
                    }
                }
            }
        }
        else { // GPU_CHILD_NODE
            if(node.left_idx >= 0){
                if(sp < 64) stack[sp++] = node.left_idx;
            }
        }

        // -------------------------
        // right child
        // -------------------------
        if(node.right_type == GPU_CHILD_LEAF){
            // printf("Checking leaf node %d\n", node.right_idx);
            if(node.right_idx >= 0){
                const GPU_LeafNode& leaf = leaves[node.right_idx];

                if(intersect_aabb(leaf.aabb, ray, t_min, closest_t)){
                    if(intersect_leaf(leaf, ray, itsc, t_min, closest_t)){
                        hit_any = true;
                    }
                }
            }
        }
        else { // GPU_CHILD_NODE
            if(node.right_idx >= 0){
                if(sp < 64) stack[sp++] = node.right_idx;
            }
        }
    }

    return hit_any;
}

__device__ vec3 raycast(const Ray& ray, const DeviceScene* d_scene, int frame){
    Intersection isect(FLT_MAX);

    bool hit = find_intersection_bvh(
        d_scene,
        ray,
        isect,
        0.001f,
        FLT_MAX
    );

    vec3 color;
    if (hit) {
        color = closest_hit(ray, isect);
    } else {
        color = vec3(0.0f, 0.0f, 0.7f);
    }

    return color;
}

