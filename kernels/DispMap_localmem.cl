__kernel void DispMap_localmem(__global unsigned char* left_img,
                               __global unsigned char* right_img,
                               __global float* means,
                               __global int* width,
                               __global int* height,
                               __global int* min_disp,
                               __global int* max_disp,
                               __global int* win_size,
                               __global int* lr,
                               __global unsigned char* disp_map) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    int w = *width;
    int h = *height;

    int win = *win_size;

    int min_d = *min_disp;
    int max_d = *max_disp;

    int half_win = (win - 1) / 2;

    int min_x = x - half_win; // window bounds
    int min_y = y - half_win;

    int lx = get_local_id(0); // local index within the work-group
    int ly = get_local_id(1);

    local unsigned char left_win_pixels[9][9];
    local unsigned char right_win_pixels[9][9];

    int yy = ly + min_y;
    int xx = lx + min_x;

    // Handle boundary conditions for left window pixels
    int y_px = (yy < 0) ? -yy - 1 : ((yy >= h) ? h - (yy % h + 1) : yy);
    int x_px = (xx < 0) ? -xx - 1 : ((xx >= w) ? w - (xx % w + 1) : xx);

    left_win_pixels[ly][lx] = left_img[y_px * w + x_px];

    barrier(CLK_LOCAL_MEM_FENCE);

    float max_zncc = -FLT_MAX;
    int best_disp = 0;

    for (int d = min_d; d <= max_d; d++) {
        // Handle boundary conditions for right window pixels
        int rx_px = (xx - d < 0) ? -xx - d - 1 : ((xx - d >= w) ? w - ((xx - d) % w + 1) : xx - d);
        right_win_pixels[ly][lx] = right_img[y_px * w + rx_px];

        barrier(CLK_LOCAL_MEM_FENCE);

        float zncc_l = 0;
        float zncc_r = 0;
        float zncc_lr = 0;

        float mean_l;
        float mean_r;

        int rx = (x - d < 0) ? -x - d - 1 : ((x - d >= w) ? w - ((x - d) % w + 1) : x - d);

        if ((*lr) == 0) {  //first w * h = left image means and then right image means
            mean_l = means[y * w + x];
            mean_r = means[y * w + rx + w * h];
        } else {
            mean_r = means[y * w + rx];
            mean_l = means[y * w + x + w * h];
        }

        float l_val = (float)left_win_pixels[ly][lx];
        float r_val = (float)right_win_pixels[ly][lx];

        local float zncc_l_list[9][9];
        local float zncc_r_list[9][9];
        local float zncc_lr_list[9][9];

        zncc_l_list[ly][lx] = (l_val - mean_l) * (l_val - mean_l);
        zncc_r_list[ly][lx] = (r_val - mean_r) * (r_val - mean_r);
        zncc_lr_list[ly][lx] = (l_val - mean_l) * (r_val - mean_r);

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i=0; i<win; i++) {
            for (int j=0; j<win; j++) {
                zncc_l += zncc_l_list[i][j];
                zncc_r += zncc_r_list[i][j];
                zncc_lr += zncc_lr_list[i][j];
            }
        }

        float zncc_val = 0;
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

    disp_map[y * w + x] = (unsigned char)abs(best_disp);
}