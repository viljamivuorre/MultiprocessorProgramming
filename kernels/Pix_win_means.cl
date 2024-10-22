__kernel void Pix_win_means(
    __global unsigned char* left_img,
    __global unsigned char* right_img,
    __global unsigned int* width,
    __global unsigned int* height,
    __global int* win_size,
    __global float* means) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    int w = *width;
    int h = *height;

    int win = *win_size;

    int min_x = x - (win - 1) / 2; //window bounds
    int max_x = x + (win - 1) / 2;

    int min_y = y - (win - 1) / 2;
    int max_y = y + (win - 1) / 2;

    float mean_l = 0;
    float mean_r = 0;

    for (int yw = min_y; yw <= max_y; yw++) { //window means with mirror padding. both left and right
        int y_px = (yw < 0) ? -(yw + 1) : (yw >= h) ? h - (yw % h + 1) : yw;
        for (int xw = min_x; xw <= max_x; xw++) {
            int x_px = (xw < 0) ? -(xw + 1) : (xw >= w) ? w - (xw % w + 1) : xw;
            mean_l += left_img[y_px * w + x_px];
            mean_r += right_img[y_px * w + x_px];
        }
    }

    means[y * w + x] = mean_l / (win * win); //first left and next height*width = right image means.
    means[y * w + x + w * h] = mean_r / (win * win);

}