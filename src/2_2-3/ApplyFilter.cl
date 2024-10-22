__kernel void ApplyFilter(__global unsigned char* img_in,
                          __global unsigned char* img_out,
                          __global unsigned int* width,
                          __global unsigned int* height,
                          __global int* color) {

    int w = *width;
    int h = *height;

    int i = get_global_id(0);

    if (*color == 0) {
        img_out[i] = img_in[i];
    }
    else if (*color == 1) {
        int hits = 0;
        float val = 0.0;
        for (int y = -2; y <= 2; y++) {
            for (int x = -2; x <= 2; x++) {
                int nx = (i % w) + x;
                int ny = (i - (i % w)) / w + y;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    hits++;
                    val += (float)(img_in[ny * w + nx]);
                }
            }
        }
        img_out[i] = (unsigned char)(val / hits);
    }
}