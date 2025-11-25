#include "includes/renderer.cuh"

__global__ void render_image(vec3* framebuffer, int image_width, int image_height, CameraParameter cam_params, DeviceScene* d_scene, int frame){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if(x >= image_width || y >= image_height) return;

    float u = float(x) / float(image_width);
    float v = float(y) / float(image_height);

    // printf("Debug: frame %d", frame);

    vec3 direction = glm::normalize(cam_params.lower_left_corner + u * cam_params.horizontal + v * cam_params.vertical - cam_params.origin);
    Ray ray(cam_params.origin, direction);


    int num_tris = d_scene->num_actions_at_frame[frame];
    // printf("Debug: Number of triangles at frame %d: %d\n", frame, num_tris);

    vec3 color = raycast(ray, d_scene, num_tris, frame);
    // printf("Debug: Pixel (%d, %d): Color (%.2f, %.2f, %.2f)\n", x, y, color.r, color.g, color.b);  
    int pixel_index = frame * image_width * image_height + y * image_width + x;  // framebuffer[frame][x][y]
    framebuffer[pixel_index] = color;
}

void export_to_ppm(string filename, vec3* framebuffers, int image_width, int image_height, int num_frames){

    for(int f = 0; f < num_frames; f++){
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

__device__ vec3 raycast(const Ray& ray, const DeviceScene* d_scene, int num_tris, int frame) {
    // printf("Debug: raycast called for frame %d with %d triangles\n", frame, num_tris);
    Intersection isect(FLT_MAX);
    bool hit = false;
    float closest_t = FLT_MAX;

    for(int i = 0; i < num_tris; i++){
        Triangle* tri = d_scene->get_triangle_at_frame(frame, i);
        // printf("triangle %d vertices: (%.2f, %.2f, %.2f), (%.2f, %.2f, %.2f), (%.2f, %.2f, %.2f)\n", i,
        //        tri->vertex(0).x, tri->vertex(0).y, tri->vertex(0).z,
        //        tri->vertex(1).x, tri->vertex(1).y, tri->vertex(1).z,
        //        tri->vertex(2).x, tri->vertex(2).y, tri->vertex(2).z);
        if(intersect_triangle(ray, tri, isect, 0.001f, closest_t)){
            hit = true;
            closest_t = isect.t;
        }
    }

    vec3 color(0.0f, 0.0f, 0.0f);
    color = hit ? closest_hit(ray, isect) : vec3(0.0f, 1.0f, 0.0f);
    return color;
}
