__kernel void add_Matrix(__global float* matrix_1, __global float* matrix_2, __global float* result) {
    {
        int i = get_global_id(0);

        result[i] = matrix_1[i] + matrix_2[i];
    }
}