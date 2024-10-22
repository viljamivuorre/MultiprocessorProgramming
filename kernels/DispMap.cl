__kernel void DispMap(__global unsigned char* left_img,
    __global unsigned char* right_img,
    __global unsigned int* width,
    __global unsigned int* height,
    __global int* min_disp,
    __global int* max_disp,
    __global int* win_size,
    __global unsigned char* disp_map) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    int w = *width;
    int h = *height;

    int min_d = *min_disp;
    int max_d = *max_disp;

    int win = *win_size;

    int min_x = x - (win - 1) / 2; //window bounds.
    int max_x = x + (win - 1) / 2;

    int min_y = y - (win - 1) / 2;
    int max_y = y + (win - 1) / 2;

    float max_zncc = 0;
    int best_disp = 0;

    for (int d = min_d; d < max_d; d++) {

        float mean_l = 0;
        float mean_r = 0;

        for (int yw = min_y; yw < max_y; yw++) { //window means. mirror padding
            for (int xw = min_x; xw < max_x; xw++) {
                int y_px = (yw < 0) ? -(yw + 1) : ((yw >= h) ? h - (yw % h + 1) : yw);
                int x_px = (xw < 0) ? -(xw + 1) : ((xw >= w) ? w - (xw % w + 1) : xw);
                int rx_px = (x_px - d < 0) ? -(x_px - d + 1) : ((x_px - d >= w) ? w - ((x_px - d) % w + 1) : x_px - d);
                mean_l += left_img[y_px * w + x_px];
                mean_r += right_img[y_px * w + rx_px];
            }
        }
        mean_l /= win * win;
        mean_r /= win * win;

        float zncc_l = 0;
        float zncc_r = 0;
        float zncc_lr = 0;

        float zncc_val = 0;

        for (int yw = min_y; yw < max_y; yw++) { //actual zncc algorithm. mirror padding
            for (int xw = min_x; xw < max_x; xw++) {
                int y_px = (yw < 0) ? -(yw + 1) : ((yw >= h) ? h - (yw % h + 1) : yw);
                int x_px = (xw < 0) ? -(xw + 1) : ((xw >= w) ? w - (xw % w + 1) : xw);
                int rx_px = (x_px - d < 0) ? -(x_px - d + 1) : ((x_px - d >= w) ? w - ((x_px - d) % w + 1) : x_px - d);
                zncc_l += (left_img[y_px * w + x_px] - mean_l) * (left_img[y_px * w + x_px] - mean_l);
                zncc_r += (right_img[y_px * w + rx_px] - mean_r) * (right_img[y_px * w + rx_px] - mean_r);
                zncc_lr += (left_img[y_px * w + x_px] - mean_l) * (right_img[y_px * w + rx_px] - mean_r);
            }
        }

        if (zncc_l == 0 || zncc_r == 0) { //very rare if winsize is enough. prevents dividing by zero 
            zncc_val = 1000000000000;
        }
        else {
            zncc_val = zncc_lr / (sqrt(zncc_l) * sqrt(zncc_r));
        }

        if (zncc_val > max_zncc) { // biggest ZNCC = best disparity
            max_zncc = zncc_val;
            best_disp = d;
        }

    }

    disp_map[y * w + x] = (char)abs(best_disp);

}