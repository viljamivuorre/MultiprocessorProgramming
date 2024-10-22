__kernel void GrayScaleImage(__global unsigned char* img_in, __global unsigned char* img_out, __global unsigned int* width, __global unsigned int* height, __global int* color) {

    int i = get_global_id(0);

    if (*color == 1) {
        img_out[i] = img_in[i];
    }
    else if (*color == 0) {
        img_out[i] = 0.2126 * img_in[i * 4] + 0.7152 * img_in[i * 4 + 1] + 0.0722 * img_in[i * 4 + 2];
    }
}