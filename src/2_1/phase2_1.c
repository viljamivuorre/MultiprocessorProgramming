

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>
#pragma comment(lib, "OpenCL.lib")

void platform_info(cl_platform_id platform, cl_device_id device);
void add_Matrix(float* matrix_1, float* matrix_2, int rows, int cols, float* result);
void generate_Matrix(float* matrix, int rows, int cols);
int compare_Matrix(float* matrix_1, float* matrix_2, int rows, int cols);
void print_Matrix(float* matrix, int rows, int cols);

int main() {

    srand(time(NULL));

    const int cols = 100;
    const int rows = 100;

    cl_mem m1_cl;
    cl_mem m2_cl;
    cl_mem mr_cl;
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    clock_t start, end; // clock

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    platform_info(platform, device);

    FILE* fp = fopen("./add_Matrix.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    char* source_str = (char*)malloc(0x100000);
    size_t source_size = fread(source_str, 1, 0x100000, fp);
    fclose(fp);

    float* matrix1 = (float*)malloc(rows * cols * sizeof(float*));
    float* matrix2 = (float*)malloc(rows * cols * sizeof(float*));
    float* matrix_cl = (float*)malloc(rows * cols * sizeof(float*));
    float* matrix_c = (float*)malloc(rows * cols * sizeof(float*));

    generate_Matrix(matrix1, rows, cols);
    generate_Matrix(matrix2, rows, cols);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    queue = clCreateCommandQueue(context, device, 0, NULL);

    m1_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * rows * cols, matrix1, &err);
    m2_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * rows * cols, matrix2, &err);
    mr_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * rows * cols, NULL, &err);

    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    kernel = clCreateKernel(program, "add_Matrix", &err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&m1_cl);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&m2_cl);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mr_cl);

    size_t result;
    size_t size_ret;
    err = clGetKernelWorkGroupInfo(kernel, NULL, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void*)&result, &size_ret);

    start = clock();

    size_t global_work_size = rows * cols;
    size_t local_work_size = 1;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

    err = clEnqueueReadBuffer(queue, mr_cl, CL_TRUE, 0, cols * rows * sizeof(float), matrix_cl, 0, NULL, NULL); //read result

    end = clock();

    printf("Execution time for OpenCL: %d ms\n", (int)(end - start));

    start = clock();

    add_Matrix(matrix1, matrix2, rows, cols, matrix_c); // run add in host

    end = clock();

    printf("Execution time for host: %d ms\n", (int)(end - start));

    printf("\n\n");

    if (compare_Matrix(matrix_cl, matrix_c, rows, cols))
    {
        printf("Matrices are not equal\n");
    }
    else
    {
        printf("Same result in host and openCL!\n");
    }

    free(matrix1);
    free(matrix2);
    free(matrix_cl);
    free(matrix_c);

    clFlush(queue);
    clFinish(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseMemObject(m1_cl);
    clReleaseMemObject(m2_cl);
    clReleaseMemObject(mr_cl);
    
	return 0;
}

void platform_info(cl_platform_id platform, cl_device_id device) { //print platform and device info
    char name[128];
    size_t name_size;
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), name, &name_size);
    printf("\n\nOpenCL Name: %s\n", name);

    char profile[128];
    size_t profile_size;
    clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, sizeof(profile), profile, &profile_size);
    printf("OpenCL Profile: %s\n", profile);

    char version[128];
    size_t version_size;
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(version), version, &version_size);
    printf("OpenCL Version: %s\n", version);

    char vendor[128];
    size_t vendor_size;
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, &vendor_size);
    printf("OpenCL Vendor: %s\n\n", vendor);

    char* value;
    size_t valueSize;
    cl_uint maxComputeUnits;

    // print device name
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*)malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Device Name: %s\n", value);
    free(value);

    // print hardware device version
    clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &valueSize);
    value = (char*)malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, valueSize, value, NULL);
    printf("Device Hardware version: %s\n", value);
    free(value);

    // print software driver version
    clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &valueSize);
    value = (char*)malloc(valueSize);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, valueSize, value, NULL);
    printf("Cevice driver version: %s\n", value);
    free(value);

    // print c version supported by compiler for device
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
    value = (char*)malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
    printf("Device OpenCL C version: %s\n", value);
    free(value);

    // print parallel compute units
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf("Device Parallel compute units: %d\n\n\n", maxComputeUnits);
}

void add_Matrix(float* matrix_1, float* matrix_2, int rows, int cols, float* result) { //add matrix host
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i * rows + j] = matrix_1[i * rows + j] + matrix_2[i * rows + j];
        }
    }
}

void generate_Matrix(float* matrix, const int rows, const int cols) { //generate matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * rows + j] = (float)rand() / RAND_MAX;
        }
    }
}

int compare_Matrix(float* matrix_1, float* matrix_2, int rows, int cols) { //compare
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matrix_1[i * rows + j] != matrix_2[i * rows + j]) {
                return 1;
            }
        }
    }
    return 0;
}

void print_Matrix(float* matrix, int rows, int cols) { //print
    printf("\n\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * rows + j]);
            if (j == cols-1) {
                printf("\n");
            }
        }
    }
    printf("\n\n");
}