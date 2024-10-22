__kernel void ResizeImage(__global unsigned char* img_in, __global unsigned char* img_out, __global unsigned int* width, __global unsigned int* height, __global int* color) {

    int w = *width;
    int h = *height;
    int re_width = w / 4;

    int i = get_global_id(0);

    if (*color == 0) {
        int x = i % re_width;
        int y = (i - x) / re_width;
        int in_x = x * 4;
        int in_y = y * 4;
        int in_px = (in_y * w + in_x) * 4;
        int out_px = (y * re_width + x) * 4;
        img_out[out_px] = img_in[in_px];
        img_out[out_px + 1] = img_in[in_px + 1];
        img_out[out_px + 2] = img_in[in_px + 2];
        img_out[out_px + 3] = img_in[in_px + 3];
    }
    else if (*color == 1) {
        int x = i % re_width;
        int y = (i - x) / re_width;
        int in_x = x * 4;
        int in_y = y * 4;
        int in_px = in_y * w + in_x;
        int out_px = y * re_width + x;
        img_out[out_px] = img_in[in_px];
    }
}