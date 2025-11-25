// #include "includes/trace.cuh"

// __device__ vec3 raycast(const Ray& ray, const DeviceScene* d_scene, int num_tris, int frame) {
//     // printf("Debug: raycast called for frame %d with %d triangles\n", frame, num_tris);
//     Intersection isect(FLT_MAX);
//     bool hit = false;
//     float closest_t = FLT_MAX;

//     for(int i = 0; i < num_tris; i++){
//         Triangle* tri = d_scene->get_triangle_at_frame(frame, i);
//         if(intersect_triangle(ray, tri, isect, 0.001f, closest_t)){
//             hit = true;
//             closest_t = isect.t;
//         }
//     }

//     vec3 color(0.0f, 0.0f, 0.0f);
//     color = hit ? vec3(1.0f, 0.0f, 0.0f) : vec3(0.0f, 0.0f, 0.0f);
//     return color;
// }