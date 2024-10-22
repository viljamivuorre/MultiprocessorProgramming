__kernel void PostProcessing(__global unsigned char* left_disp_map, __global unsigned char* right_disp_map, 
  __global int* width, __global int* height, __global int* treshold, __global unsigned char* min,
   __global unsigned char* max, __global unsigned char* result) {

    int y = get_global_id(0);
    
    int w = *width;
    int h = *height;
    int tresh = *treshold;

    /* cross checking */

    for (int x = 0; x < w; x++) {
        if (abs(left_disp_map[y * w + x] - right_disp_map[y * w + x]) > tresh) { //assign zero to pixels where difference is greater than treshold
            result[y * w + x] = 0;
        }
        else {
            result[y * w + x] = left_disp_map[y * w + x];
        }
    }

    /* occlusion filling */

    unsigned char nn_color = 0;
    for (int x = 0; x < w; x++) {
        if (result[y * width + x] > 0) {
            nn_color = result[y * w + x]
            break;
        }
    }

    for (int x = 0; x < w; x++) {
        unsigned char nn_color = 0;
        if (result[y * width + x] == 0) { //nearest neighbour color to 0 disps
            result[y * width + x] = nn_color;
        } else {
            nn_color = result[y * width + x];
        }
    }

    //normalization
    
    int ma = *max;
    int mi = *mi;

    for (int x=0; x<w; x++) {
        result[y * w + x] = (unsigned char)(255 * (left_disp_map[y * w + x] - mi) / (ma - mi));
    }

}