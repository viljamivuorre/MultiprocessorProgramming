__kernel void Normalization(__global unsigned char* map, __global unsigned char* max, __global unsigned char* min,
__global unsigned char* result) {

    int i = get_global_id(0);

    unsigned char ma = *max;
    unsigned char mi = *min;

    result[i] = (unsigned char)(255 * (map[i] - mi) / (ma - mi));

}