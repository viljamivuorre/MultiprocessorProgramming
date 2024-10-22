__kernel void ZNCC_vectorized(__global unsigned char* left_img,
                   __global unsigned char* right_img,
                   __global float* means,
                   __global unsigned int* width,
                   __global unsigned int* height,
                   __global int* min_disp,
                   __global int* max_disp,
                   __global int* win_size,
                   __global int* lr,
                   __global unsigned char* disp_map) {

    int x = get_global_id(0) * 4;
    int y = get_global_id(1);

    int w = *width;
    int h = *height;

    int min_d = *min_disp;
    int max_d = *max_disp;

    int win = *win_size;

    int min_x = x - (win - 1) / 2; //window bounds
    int max_x = x + (win - 1) / 2;

    int min_y = y - (win - 1) / 2;
    int max_y = y + (win - 1) / 2;

    float max_zncc = 0;
    int best_disp = 0;

    float4 zncc_l = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 zncc_r = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 zncc_lr = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int d = min_d; d < max_d; d++) {

        zncc_l = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        zncc_r = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        zncc_lr = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

        float zncc_val = 0;

        float4 mean_l;
        float4 mean_r;

        int rx[4] = {(x - d < 0) ? -(x - d + 1) : (x - d >= w) ? w - ((x - d) % w + 1) : x - d,
                    (x+1 - d < 0) ? -(x+1 - d + 1) : (x+1 - d >= w) ? w - ((x+1 - d) % w + 1) : x+1 - d,
                    (x+2 - d < 0) ? -(x+2 - d + 1) : (x+2 - d >= w) ? w - ((x+2 - d) % w + 1) : x+2 - d,
                    (x+3 - d < 0) ? -(x+3 - d + 1) : (x+3 - d >= w) ? w - ((x+3 - d) % w + 1) : x+3 - d};
        
        if ((*lr) == 0) {
            mean_l = (float4)(means[y * w + x], means[y * w + x+1], means[y * w + x+2], means[y * w + x+3]);
            mean_r = (float4)(means[y * w + rx[0] + w * h], means[y * w + rx[1] + w * h], means[y * w + rx[2] + w * h], means[y * w + rx[3] + w * h]);
        } else {
            mean_r = (float4)(means[y * w + rx[0]], means[y * w + rx[1]], means[y * w + rx[2]], means[y * w + rx[3]]);
            mean_l = (float4)(means[y * w + x + w * h], means[y * w + x+1 + w * h], means[y * w + x+2 + w * h], means[y * w + x+3 + w * h]);
        }

        for (int yw = min_y; yw < max_y; yw++) { //mirror padding
            for (int xw = min_x; xw < max_x; xw += 4) {
                int y_px = (yw < 0) ? -(yw + 1) : (yw >= h) ? h - (yw % h + 1) : yw;
                int x_px[4] = {(xw < 0) ? -(xw + 1) : (xw >= w) ? w - (xw % w + 1) : xw,
                                 (xw+1 < 0) ? -(xw+1 + 1) : (xw+1 >= w) ? w - ((xw+1) % w + 1) : xw+1,
                                 (xw+2 < 0) ? -(xw+2 + 1) : (xw+2 >= w) ? w - ((xw+2) % w + 1) : xw+2,
                                 (xw+3 < 0) ? -(xw+3 + 1) : (xw+3 >= w) ? w - ((xw+3) % w + 1) : xw+3};
                int rx_px[4] = {(x_px[0] - d < 0) ? -(x_px[0] - d + 1) : (x_px[0] - d >= w) ? w - ((x_px[0] - d) % w + 1) : x_px[0] - d,
                                (x_px[1] - d < 0) ? -(x_px[1] - d + 1) : (x_px[1] - d >= w) ? w - (((x_px[1]) - d) % w + 1) : x_px[1] - d,
                                (x_px[2] - d < 0) ? -(x_px[2] - d + 1) : (x_px[2] - d >= w) ? w - (((x_px[2]) - d) % w + 1) : x_px[2] - d,
                                (x_px[3] - d < 0) ? -(x_px[3] - d + 1) : (x_px[3] - d >= w) ? w - (((x_px[3]) - d) % w + 1) : x_px[3] - d};

                float4 left_val = vload4(0, left_img + y_px * w + x_px[0]);
                float4 right_val = vload4(0, right_img + y_px * w + rx_px[0]);

                zncc_l += (left_val - mean_l) * (left_val - mean_l);
                zncc_r += (right_val - mean_r) * (right_val - mean_r);
                zncc_lr += (left_val - mean_l) * (right_val - mean_r);
            }
        }

        float4 zncc_l_sqrt = sqrt(zncc_l);
        float4 zncc_r_sqrt = sqrt(zncc_r);
        float4 zncc_lr_div = zncc_lr / (zncc_l_sqrt * zncc_r_sqrt);

        float max_zncc_local = fmax(fmax(fmax(zncc_lr_div.s0, zncc_lr_div.s1), fmax(zncc_lr_div.s2, zncc_lr_div.s3)), max_zncc);
        if (max_zncc_local > max_zncc) {
            max_zncc = max_zncc_local;
            best_disp = d;
        }
    }
    disp_map[y * w + x] = abs(best_disp);
}