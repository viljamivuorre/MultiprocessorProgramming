__kernel void CrossChecking(__global unsigned char* left_disp_map, __global unsigned char* right_disp_map,
 __global int* treshold, __global unsigned char* result) {

    int i = get_global_id(0);
    int tresh = *treshold;

    if (abs(left_disp_map[i] - right_disp_map[i]) > tresh) { //assign zero to pixels where difference is greater than treshold
        result[i] = 0;
    }
    else {
        result[i] = left_disp_map[i];
    }

}