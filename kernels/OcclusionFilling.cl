__kernel void OcclusionFilling(__global unsigned char* map, __global int* width, __global unsigned char* result) {

    int y = get_global_id(0);
    int w = *width;

    unsigned char nn_color = 0;

    for (int x = 0; x < w; x++) {
        if (map[y * w + x] > 0) {
            nn_color = map[y * w + x];
            break;
        }
    }

    for (int x = 0; x < w; x++) {
        if (map[y * w + x] == 0) { //nearest neighbour color to 0 disps
            result[y * w + x] = nn_color;
        } else {
            nn_color = map[y * w + x];
            result[y * w + x] = map[y * w + x];
        }
    }

}