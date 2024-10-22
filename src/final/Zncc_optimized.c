#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../../LodePNG/lodepng.h"
#include "../../LodePNG/lodepng.c"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#pragma comment(lib, "OpenCL.lib")

#define MAX_DISP 65

#define WIN_SIZE 35

#define TRESHOLD 10

#define resize 1
#define grayscale 2
#define filter 3

#define rgba 0
#define gray 1

#define check_err(_ret_value) if (_ret_value != 0) printf("\nError %d\n\n", _ret_value)

void means_OpenCL(cl_device_id * device, unsigned char* left_img, unsigned char* right_img, int* width, int* height,
    int* win_size, float* means);

void ZNCC_OpenCL(cl_device_id * device, unsigned char* left_img, unsigned char* right_img, float* means, int* width, int* height,
    int* win_size, int* min_disp, int* max_disp, int* lr, unsigned char* disp_map);

void dispMap(cl_device_id* device, unsigned char* left_img, unsigned char* right_img, int* width, int* height,
    int* win_size, int* min_disp, int* max_disp, unsigned char* disp_img);

void crossCheckingOpenCL(cl_device_id* device, unsigned char* left_img, unsigned char* right_img, int* treshold, unsigned char* result, int width, int height);
void occlusionFillingOpenCL(cl_device_id* device, unsigned char* map, int* width, unsigned char* result, int height);
void normalizationOpenCL(cl_device_id* device, unsigned char* map, int* max, int* min, unsigned char* result, int width, int height);

void postProcessingOpenCL(cl_device_id* device, unsigned char* left_img, unsigned char* right_img, int* width, int* height,
    int* treshold, unsigned char* min, unsigned char* max, unsigned char* result, unsigned char* map1, unsigned char* map2);

void cross_checking(unsigned char* left_disp_map, unsigned char* right_disp_map, unsigned char* result, int size, int treshold);
void occlusion_filling(unsigned char* result, int width, int height);
void normalization(unsigned char* disp_map, int size);
void resizeImage(unsigned char* img_in, unsigned char* img_out, unsigned int width, unsigned int height, int color);
void grayScaleImage(unsigned char* img_in, unsigned char* img_out, unsigned int width, unsigned int height, int color);
void applyFilter(unsigned char* img_in, unsigned char* img_out, unsigned int width, unsigned int height, int color);
int processImageCL(cl_device_id* device, unsigned char* img_in, unsigned char** img_out,
    unsigned int* width, unsigned int* height, int type, int* color);
void ReadImage(char* filename, unsigned char** image, unsigned int* width, unsigned int* height, int type);
void WriteImage(char* filename, unsigned char* image, unsigned int width, unsigned int height, int type);
void device_info(cl_platform_id platform, cl_device_id device);

int main() {

    //clock_t start, end;

    cl_int           err;
    cl_platform_id platform;
    cl_device_id     device;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err) {
        printf("No platform detected, exit\n");
        exit(1);
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err) {
        printf("No device detected, exit\n");
        exit(1);
    }

    device_info(platform, device);

    /* Read Images */

    unsigned char* img0 = NULL;
    unsigned int width = NULL;
    unsigned int height = NULL;

    unsigned char* img1 = NULL;
    unsigned int width1 = NULL;
    unsigned int height1 = NULL;

    ReadImage("../../im0.png", &img0, &width, &height, rgba);

    unsigned char* resized_img0 = (unsigned char*)malloc(sizeof(unsigned char) * (int)(width / 4) * (int)(height / 4) * 4);

    int color = 0;

    ReadImage("../../im1.png", &img1, &width1, &height1, rgba);

    unsigned char* resized_img1 = (unsigned char*)malloc(sizeof(unsigned char) * (int)(width1 / 4) * (int)(height1 / 4) * 4);

    color = 0;

    processImageCL(&device, img0, &resized_img0, &width, &height, resize, &color);
    processImageCL(&device, img1, &resized_img1, &width, &height, resize, &color);

    WriteImage("resized0.png", resized_img0, (unsigned int)(width / 4), (unsigned int)(height / 4), rgba);
    WriteImage("resized1.png", resized_img1, (unsigned int)(width1 / 4), (unsigned int)(height1 / 4), rgba);

    free(resized_img0);
    free(resized_img1);

    ReadImage("./resized0.png", &img0, &width, &height, 0);
    ReadImage("./resized1.png", &img1, &width1, &height1, 0);

    unsigned char* grayscale_img0 = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
    unsigned char* grayscale_img1 = (unsigned char*)malloc(sizeof(unsigned char) * width1 * height1);

    color = 0;
    processImageCL(&device, img0, &grayscale_img0, &width, &height, grayscale, &color);
    processImageCL(&device, img1, &grayscale_img1, &width, &height, grayscale, &color);


    WriteImage("grayscaleImage0.png", grayscale_img0, width, height, 1);
    WriteImage("grayscaleImage1.png", grayscale_img1, width1, height1, 1);

    free(grayscale_img0);
    free(grayscale_img1);

    ReadImage("./grayscaleImage0.png", &img0, &width, &height, 1);
    ReadImage("./grayscaleImage1.png", &img1, &width1, &height1, 1);

    unsigned char* dispMap0_img = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
    unsigned char* dispMap1_img = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
    float* means = (float*)malloc(sizeof(float) * width * height * 2);

    int win_size = WIN_SIZE;
    int min_disp = 0;
    int max_disp = MAX_DISP;

    means_OpenCL(&device, img0, img1, &width, &height, &win_size, means); //calculate mean map

    int lr = 0;

    ZNCC_OpenCL(&device, img0, img1, means, &width, &height, &win_size, &min_disp, &max_disp, &lr, dispMap0_img); //zncc values.

    min_disp = -MAX_DISP + 1;
    max_disp = 1;
    lr = 1;

    ZNCC_OpenCL(&device, img1, img0, means, &width, &height, &win_size, &min_disp, &max_disp, &lr, dispMap1_img);

    /*dispMap(&device, img0, img1, &width, &height, &win_size, &min_disp, &max_disp, dispMap0_img);

    min_disp = -MAX_DISP + 1;
    max_disp = 1;

    dispMap(&device, img1, img0, &width, &height, &win_size, &min_disp, &max_disp, dispMap1_img);
    */

    WriteImage("dispMapImage0.png", dispMap0_img, width, height, 1);
    WriteImage("dispMapImage1.png", dispMap1_img, width, height, 1);

    clock_t start = clock();

    unsigned char* result_img = (unsigned char*)malloc(sizeof(unsigned char) * width * height);

    unsigned char max = 0;
    unsigned char min = 500;

    int treshold = TRESHOLD;

    /* post processing */
    unsigned char* map1 = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
    unsigned char* map2 = (unsigned char*)malloc(sizeof(unsigned char) * width * height);

    postProcessingOpenCL(&device, dispMap0_img, dispMap1_img, &width, &height, &treshold, &min, &max, result_img, map1, map2);
    //crossCheckingOpenCL(&device, dispMap0_img, dispMap1_img, &treshold, map1, width, height);
    //occlusionFillingOpenCL(&device, map1, &width, map2, height);

    /*for (int i = 0; i < (width) * (height); i++) {
        if (map2[i] < min) {
            min = map2[i];
        }
        if (map2[i] > max) {
            max = map2[i];
        }
    }
    */
    //normalizationOpenCL(&device, map2, &max, &min, result_img, width, height);

    clock_t end = clock();
    printf("execution time for post processing OpenCL: %d ms\n\n", (int)(end - start));

    WriteImage("final.png", result_img, width, height, 1);

    free(map1);
    free(map2);
    free(dispMap0_img);
    free(dispMap1_img);
    free(result_img);
    free(img0);
    free(img1);

    return 0;
}

void means_OpenCL(cl_device_id* device, unsigned char* left_img, unsigned char* right_img, int* width, int* height,
    int* win_size, float* means) {  //win means kernel execution

    cl_int           err;

    FILE* fp;

    size_t buf_size = (size_t)(sizeof(unsigned char) * (*width) * (*height));
    size_t buf_size2 = (size_t)(sizeof(float) * (*width) * (*height) * 2);

    fp = fopen("../../kernels/Pix_win_means.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    char* source_str = (char*)malloc(0x100000);
    size_t source_size = fread(source_str, 1, 0x100000, fp);
    fclose(fp);

    cl_context context = clCreateContext(NULL, 1, device, NULL, NULL, &err);

    check_err(err);

    cl_command_queue queue = clCreateCommandQueue(context, *device, 0, &err);

    check_err(err);

    cl_mem left_img_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, left_img, &err);
    check_err(err);
    cl_mem right_img_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, right_img, &err);
    check_err(err);
    cl_mem width_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), width, &err);
    check_err(err);
    cl_mem height_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), height, &err);
    check_err(err);
    cl_mem win_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), win_size, &err);
    check_err(err);
    cl_mem means_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buf_size2, NULL, &err);
    check_err(err);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
    check_err(err);

    err = clBuildProgram(program, 1, device, NULL, NULL, NULL);
    check_err(err);

    cl_kernel kernel = clCreateKernel(program, "Pix_win_means", &err);
    check_err(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&left_img_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&right_img_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&width_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&height_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&win_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&means_cl);
    check_err(err);

    size_t global_work_size[2] = { (*width), (*height) };

    clock_t start = clock();

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    check_err(err);
    err = clEnqueueReadBuffer(queue, means_cl, CL_TRUE, 0, buf_size2, means, 0, NULL, NULL);
    check_err(err);

    clock_t end = clock();

    clReleaseMemObject(left_img_cl);  //release mem
    clReleaseMemObject(right_img_cl);
    clReleaseMemObject(width_cl);
    clReleaseMemObject(height_cl);;
    clReleaseMemObject(win_cl);
    clReleaseMemObject(means_cl);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(source_str);

    printf("\nExecution time of zncc algorithm for calculating means OpenCL: %d ms\n", (int)(end - start));

}

void ZNCC_OpenCL(cl_device_id* device, unsigned char* left_img, unsigned char* right_img, float* means, int* width, int* height,
    int* win_size, int* min_disp, int* max_disp, int* lr, unsigned char* disp_map) { //zncc kernel execution with already calculated win means

    cl_int           err;

    FILE* fp;

    size_t buf_size = (size_t)(sizeof(unsigned char) * (*width) * (*height));
    size_t buf_size2 = (size_t)(sizeof(float) * (*width) * (*height) * 2);

    fp = fopen("../../kernels/ZNCC.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    char* source_str = (char*)malloc(0x100000);
    size_t source_size = fread(source_str, 1, 0x100000, fp);
    fclose(fp);

    cl_context context = clCreateContext(NULL, 1, device, NULL, NULL, &err);

    check_err(err);

    cl_command_queue queue = clCreateCommandQueue(context, *device, 0, &err);

    check_err(err);

    cl_mem left_img_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, left_img, &err);
    check_err(err);
    cl_mem right_img_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, right_img, &err);
    check_err(err);
    cl_mem means_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size2, means, &err);
    check_err(err);
    cl_mem width_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), width, &err);
    check_err(err);
    cl_mem height_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), height, &err);
    check_err(err);
    cl_mem min_disp_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), min_disp, &err);
    check_err(err);
    cl_mem max_disp_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), max_disp, &err);
    check_err(err);
    cl_mem win_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), win_size, &err);
    check_err(err);
    cl_mem lr_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), lr, &err);
    check_err(err);
    cl_mem disp_map_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buf_size, NULL, &err);
    check_err(err);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
    check_err(err);

    err = clBuildProgram(program, 1, device, NULL, NULL, NULL);
    check_err(err);

    cl_kernel kernel = clCreateKernel(program, "ZNCC", &err);
    check_err(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&left_img_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&right_img_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&means_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&width_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&height_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&min_disp_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&max_disp_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&win_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&lr_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&disp_map_cl);
    check_err(err);

    //cl_event event;

    size_t global_work_size[2] = { (*width), (*height) };
    //size_t local_work_size[2] = { 32, 32 };

    clock_t start = clock();

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    check_err(err);
    err = clEnqueueReadBuffer(queue, disp_map_cl, CL_TRUE, 0, buf_size, disp_map, 0, NULL, NULL);
    check_err(err);

    // Suoritusaika
    //cl_ulong start, end;
    //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    //double elapsed_time = (end - start) / 1000000.0; // muunnetaan mikrosekunneiksi

    //printf("\nExecution time of zncc algorithm for OpenCL: %f ms\n", elapsed_time);

    clock_t end = clock();

    clReleaseMemObject(left_img_cl);
    clReleaseMemObject(right_img_cl);
    clReleaseMemObject(means_cl);
    clReleaseMemObject(width_cl);
    clReleaseMemObject(height_cl);
    clReleaseMemObject(min_disp_cl);
    clReleaseMemObject(max_disp_cl);
    clReleaseMemObject(win_cl);
    clReleaseMemObject(lr_cl);
    clReleaseMemObject(disp_map_cl);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(source_str);

    printf("\nExecution time of zncc algorithm for OpenCL: %d ms\n", (int)(end - start));

}

void dispMap(cl_device_id* device, unsigned char* left_img, unsigned char* right_img, int* width, int* height,
    int* win_size, int* min_disp, int* max_disp, unsigned char* disp_map) {

    cl_int           err;

    FILE* fp;

    size_t buf_size = (size_t)(sizeof(unsigned char) * (*width) * (*height));

    fp = fopen("../../kernels/DispMap.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    char* source_str = (char*)malloc(0x100000);
    size_t source_size = fread(source_str, 1, 0x100000, fp);
    fclose(fp);

    cl_context context = clCreateContext(NULL, 1, device, NULL, NULL, &err);

    check_err(err);

    cl_command_queue queue = clCreateCommandQueue(context, *device, 0, &err);

    check_err(err);

    cl_mem left_img_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, left_img, &err);
    check_err(err);
    cl_mem right_img_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, right_img, &err);
    check_err(err);
    cl_mem width_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), width, &err);
    check_err(err);
    cl_mem height_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), height, &err);
    check_err(err);
    cl_mem min_disp_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), min_disp, &err);
    check_err(err);
    cl_mem max_disp_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), max_disp, &err);
    check_err(err);
    cl_mem win_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), win_size, &err);
    check_err(err);
    cl_mem disp_map_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buf_size, NULL, &err);
    check_err(err);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
    check_err(err);

    err = clBuildProgram(program, 1, device, NULL, NULL, NULL);
    check_err(err);

    cl_kernel kernel = clCreateKernel(program, "DispMap", &err);
    check_err(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&left_img_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&right_img_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&width_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&height_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&min_disp_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&max_disp_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&win_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&disp_map_cl);
    check_err(err);

    //cl_event event;

    printf("%d, %d", *width, *height);

    size_t global_work_size[2] = { (*width), (*height)};
    size_t local_work_size[2] = {32, 32};

    clock_t start = clock();

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    check_err(err);
    err = clEnqueueReadBuffer(queue, disp_map_cl, CL_TRUE, 0, buf_size, disp_map, 0, NULL, NULL);
    check_err(err);

    // Suoritusaika
    //cl_ulong start, end;
    //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    //double elapsed_time = (end - start) / 1000000.0; // muunnetaan mikrosekunneiksi

    //printf("\nExecution time of zncc algorithm for OpenCL: %f ms\n", elapsed_time);

    clock_t end = clock();

    clReleaseMemObject(left_img_cl);
    clReleaseMemObject(right_img_cl);
    clReleaseMemObject(width_cl);
    clReleaseMemObject(height_cl);
    clReleaseMemObject(min_disp_cl);
    clReleaseMemObject(max_disp_cl);
    clReleaseMemObject(win_cl);
    clReleaseMemObject(disp_map_cl);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(source_str);

    printf("\nExecution time of zncc algorithm for OpenCL: %d ms\n", (int)(end - start));

}

void crossCheckingOpenCL(cl_device_id* device, unsigned char* left_img, unsigned char* right_img, int* treshold, unsigned char* result, int width, int height) {

    cl_int           err;

    FILE* fp;

    size_t buf_size = (size_t)(sizeof(unsigned char) * width * height);

    fp = fopen("../../kernels/CrossChecking.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    clock_t start = clock();

    char* source_str = (char*)malloc(0x100000);
    size_t source_size = fread(source_str, 1, 0x100000, fp);
    fclose(fp);

    cl_context context = clCreateContext(NULL, 1, device, NULL, NULL, &err);

    check_err(err);

    cl_command_queue queue = clCreateCommandQueue(context, *device, 0, &err);

    check_err(err);

    cl_mem left_img_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, left_img, &err);
    check_err(err);
    cl_mem right_img_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, right_img, &err);
    check_err(err);
    cl_mem treshold_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), treshold, &err);
    check_err(err);
    cl_mem result_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buf_size, NULL, &err);
    check_err(err);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
    check_err(err);

    err = clBuildProgram(program, 1, device, NULL, NULL, NULL);
    check_err(err);

    cl_kernel kernel = clCreateKernel(program, "CrossChecking", &err);
    check_err(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&left_img_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&right_img_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&treshold_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&result_cl);
    check_err(err);

    size_t global_work_size = (size_t)(width * height);
    size_t local_work_size = 1;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    check_err(err);
    err = clEnqueueReadBuffer(queue, result_cl, CL_TRUE, 0, buf_size, result, 0, NULL, NULL);
    check_err(err);

    free(source_str);
    clReleaseMemObject(left_img_cl);
    clReleaseMemObject(right_img_cl);
    clReleaseMemObject(treshold_cl);
    clReleaseMemObject(result_cl);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

}

void occlusionFillingOpenCL(cl_device_id* device, unsigned char* map, int* width, unsigned char* result, int height) {

    cl_int           err;

    FILE* fp;

    size_t buf_size = (size_t)(sizeof(unsigned char) * (*width) * height);

    fp = fopen("../../kernels/OcclusionFilling.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    clock_t start = clock();

    char* source_str = (char*)malloc(0x100000);
    size_t source_size = fread(source_str, 1, 0x100000, fp);
    fclose(fp);

    cl_context context = clCreateContext(NULL, 1, device, NULL, NULL, &err);

    check_err(err);

    cl_command_queue queue = clCreateCommandQueue(context, *device, 0, &err);

    check_err(err);

    cl_mem map_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, map, &err);
    check_err(err);
    cl_mem width_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), width, &err);
    check_err(err);
    cl_mem result_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buf_size, NULL, &err);
    check_err(err);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
    check_err(err);

    err = clBuildProgram(program, 1, device, NULL, NULL, NULL);
    check_err(err);

    cl_kernel kernel = clCreateKernel(program, "OcclusionFilling", &err);
    check_err(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&map_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&width_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&result_cl);
    check_err(err);

    size_t global_work_size = (size_t)(height);
    size_t local_work_size = 1;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    check_err(err);
    err = clEnqueueReadBuffer(queue, result_cl, CL_TRUE, 0, buf_size, result, 0, NULL, NULL);
    check_err(err);

    free(source_str);
    clReleaseMemObject(map_cl);
    clReleaseMemObject(width_cl);
    clReleaseMemObject(result_cl);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

}

void normalizationOpenCL(cl_device_id* device, unsigned char* map, int* max, int* min, unsigned char* result, int width, int height) {

    cl_int           err;

    FILE* fp;

    size_t buf_size = (size_t)(sizeof(unsigned char) * width * height);

    fp = fopen("../../kernels/Normalization.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    clock_t start = clock();

    char* source_str = (char*)malloc(0x100000);
    size_t source_size = fread(source_str, 1, 0x100000, fp);
    fclose(fp);

    cl_context context = clCreateContext(NULL, 1, device, NULL, NULL, &err);

    check_err(err);

    cl_command_queue queue = clCreateCommandQueue(context, *device, 0, &err);

    check_err(err);

    cl_mem map_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, map, &err);
    check_err(err);
    cl_mem max_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), max, &err);
    check_err(err);
    cl_mem min_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), min, &err);
    check_err(err);
    cl_mem result_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buf_size, NULL, &err);
    check_err(err);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
    check_err(err);

    err = clBuildProgram(program, 1, device, NULL, NULL, NULL);
    check_err(err);

    cl_kernel kernel = clCreateKernel(program, "Normalization", &err);
    check_err(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&map_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&max_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&min_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&result_cl);
    check_err(err);

    size_t global_work_size = (size_t)(height * width);
    size_t local_work_size = 1;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    check_err(err);
    err = clEnqueueReadBuffer(queue, result_cl, CL_TRUE, 0, buf_size, result, 0, NULL, NULL);
    check_err(err);

    free(source_str);
    clReleaseMemObject(map_cl);
    clReleaseMemObject(max_cl);
    clReleaseMemObject(min_cl);
    clReleaseMemObject(result_cl);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

}

void postProcessingOpenCL(cl_device_id* device, unsigned char* left_img, unsigned char* right_img, int* width, int* height,
    int* treshold, unsigned char* min, unsigned char* max, unsigned char* result, unsigned char* map1, unsigned char* map2) {

    cl_int           err;

    FILE* fp;

    size_t buf_size = (size_t)(sizeof(unsigned char) * (*width) * (*height));

    fp = fopen("../../kernels/CrossChecking.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    clock_t start = clock();

    char* source_str = (char*)malloc(0x100000);
    size_t source_size = fread(source_str, 1, 0x100000, fp);
    fclose(fp);

    cl_context context = clCreateContext(NULL, 1, device, NULL, NULL, &err);

    check_err(err);

    cl_command_queue queue = clCreateCommandQueue(context, *device, 0, &err);

    check_err(err);

    cl_mem left_img_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, left_img, &err);
    check_err(err);
    cl_mem right_img_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, right_img, &err);
    check_err(err);
    cl_mem treshold_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), treshold, &err);
    check_err(err);
    cl_mem map1_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buf_size, NULL, &err);
    check_err(err);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
    check_err(err);

    err = clBuildProgram(program, 1, device, NULL, NULL, NULL);
    check_err(err);

    cl_kernel kernel = clCreateKernel(program, "CrossChecking", &err);
    check_err(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&left_img_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&right_img_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&treshold_cl);
    check_err(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&map1_cl);
    check_err(err);

    size_t global_work_size = (size_t)((*height) * (*width));
    size_t local_work_size = 1;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    check_err(err);
    err = clEnqueueReadBuffer(queue, map1_cl, CL_TRUE, 0, buf_size, map1, 0, NULL, NULL);
    check_err(err);

    fp = fopen("../../kernels/OcclusionFilling.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(0x100000);
    source_size = fread(source_str, 1, 0x100000, fp);
    fclose(fp);


    map1_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, map1, &err);
    check_err(err);
    cl_mem width_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), width, &err);
    check_err(err);
    cl_mem map2_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buf_size, NULL, &err);
    check_err(err);

    cl_program program2 = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
    check_err(err);

    err = clBuildProgram(program2, 1, device, NULL, NULL, NULL);
    check_err(err);

    cl_kernel kernel2 = clCreateKernel(program2, "OcclusionFilling", &err);
    check_err(err);

    err = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void*)&map1_cl);
    check_err(err);
    err = clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void*)&width_cl);
    check_err(err);
    err = clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void*)&map2_cl);
    check_err(err);

    global_work_size = (size_t)(*height);
    local_work_size = 1;

    err = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    check_err(err);
    err = clEnqueueReadBuffer(queue, map2_cl, CL_TRUE, 0, buf_size, map2, 0, NULL, NULL);
    check_err(err);

    // min and max values

    int mi = 500;
    int ma = 0;

    for (int i = 0; i < (*width) * (*height); i++) {
        if (map2[i] < mi) {
            mi = map2[i];
        }
        if (map2[i] > ma) {
            ma = map2[i];
        }
    }

    *min = (unsigned char*)(mi);
    *max = (unsigned char*)(ma);

    fp = fopen("../../kernels/Normalization.cl", "r");

    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(0x100000);
    source_size = fread(source_str, 1, 0x100000, fp);
    fclose(fp);

    map2_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, map2, &err);
    check_err(err);
    cl_mem max_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(unsigned char)), max, &err);
    check_err(err);
    cl_mem min_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(unsigned char)), min, &err);
    check_err(err);
    cl_mem result_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buf_size, NULL, &err);
    check_err(err);

    cl_program program3 = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
    check_err(err);

    err = clBuildProgram(program3, 1, device, NULL, NULL, NULL);
    check_err(err);

    cl_kernel kernel3 = clCreateKernel(program3, "Normalization", &err);
    check_err(err);

    err = clSetKernelArg(kernel3, 0, sizeof(cl_mem), (void*)&map2_cl);
    check_err(err);
    err = clSetKernelArg(kernel3, 1, sizeof(cl_mem), (void*)&max_cl);
    check_err(err);
    err = clSetKernelArg(kernel3, 2, sizeof(cl_mem), (void*)&min_cl);
    check_err(err);
    err = clSetKernelArg(kernel3, 3, sizeof(cl_mem), (void*)&result_cl);
    check_err(err);

    global_work_size = (*width) * (*height);
    local_work_size = 1;

    err = clEnqueueNDRangeKernel(queue, kernel3, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    check_err(err);
    err = clEnqueueReadBuffer(queue, result_cl, CL_TRUE, 0, buf_size, result, 0, NULL, NULL);
    check_err(err);

    clock_t end = clock();

    clReleaseMemObject(map2_cl);
    clReleaseMemObject(result_cl);
    clReleaseMemObject(min_cl);
    clReleaseMemObject(max_cl);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(source_str);

    printf("\nExecution time of post processing algorithm for OpenCL: %d ms\n", (int)(end - start));

}

void cross_checking(unsigned char* left_disp_map, unsigned char* right_disp_map, unsigned char* result, int size, int treshold) {

    for (int i = 0; i < size; i++) {
        if (abs(left_disp_map[i] - right_disp_map[i]) > treshold) { //assign zero to pixels where difference is greater than treshold
            result[i] = 0;
        }
        else {
            result[i] = left_disp_map[i];
        }
    }

}

void occlusion_filling(unsigned char* result, int width, int height) {

    int size = width * height;

    /*unsigned char* tmp = (unsigned char*)malloc(size * sizeof(char)); //using fill with window average of > 0 pixels

    for (int i = 0; i < size; i++) {
        tmp[i] = result[i];
    }

    for (int i = 0; i < size; i++) {
        if (result[i] == 0) {
            int px_height = ((i - (i % width)) / width);
            int px_width = i % width;
            int hits = 0;
            unsigned char total = 0;
            for (int y = px_height - 15; y <= px_height + 15; y++) {
                int y_px;
                int x_px;
                if (y < 0 || y >= height) {
                    continue;
                }
                for (int x = px_width - 15; x <= px_width + 15; x++) {
                    if ((y == px_height && x == px_width) || x < 0 || x >= width || tmp[y * width + x] == 0) {
                        continue;
                    }

                    hits++;
                    total += tmp[y * width + x];

                }
            }
            if (hits == 0) {
                printf("a\n");
                for (int j = i; j < size; j++) {
                    if (result[j] > 0) {
                        result[i] = tmp[j];
                        break;
                    }
                }
            }
            else {
                result[i] = (unsigned char)(floor(total / hits));
                printf("b\n");
            }
        }
    }*/

    unsigned char nn_color = 0;  //nearest left pixel algorithm

    for (int i = 0; i < size; i++) {
        if (result[i] > 0) {
            nn_color = result[i];
            break;
        }
    }

    for (int i = 0; i < size; i++) { //nearest neighbour color to 0 disps
        if (result[i] == 0) {
            result[i] = nn_color;
        }
        else {
            nn_color = result[i];
        }
    }
}

void normalization(unsigned char* disp_map, int size) {

    unsigned char min = (unsigned char)(500);
    unsigned char max = (unsigned char)(0);

    for (int i = 0; i < size; i++) { //min and max values
        if (disp_map[i] > max) {
            max = disp_map[i];
        }
        if (disp_map[i] < min) {
            min = disp_map[i];
        }
    }
    for (int i = 0; i < size; i++) { //normalization
        disp_map[i] = (unsigned char)(255 * (disp_map[i] - min) / (max - min));
    }

}




void resizeImage(unsigned char* img_in, unsigned char* img_out, unsigned int width, unsigned int height, int color) {

    clock_t start = clock();

    int re_width = width / 4;
    int re_height = height / 4;

    if (color == rgba) {
        for (int y = 0; y < re_height; y++) {
            for (int x = 0; x < re_width; x++) {
                int x_in = x * 4;
                int y_in = y * 4;

                int out_px = (y * re_width + x) * 4;
                int in_px = (y_in * width + x_in) * 4;

                for (int i = 0; i < 4; i++) {
                    img_out[out_px + i] = img_in[in_px + i];
                }
            }
        }
    }
    else if (color == gray) {
        for (int i = 0; i < re_height; i++) {
            for (int j = 0; j < re_width; j++) {

                img_out[i * re_width + j] = img_in[i * 4 * width + 4 * j];

            }
        }
    }

    clock_t end = clock();

    printf("\n\nresize execution time in host: %d us\n\n", (int)(end - start));

}

void grayScaleImage(unsigned char* img_in, unsigned char* img_out, unsigned int width, unsigned int height, int color) {

    clock_t start = clock();

    if (color == gray) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                img_out[i * width + j] = img_in[i * width + j];
            }
        }
    }
    else if (color == rgba) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                int out_px = y * width + x;
                int in_px = out_px * 4;

                img_out[out_px] = 0.2126 * img_in[in_px] + 0.7152 * img_in[in_px + 1] + 0.0722 * img_in[in_px + 2];
            }
        }
    }

    clock_t end = clock();

    printf("\n\ngrayscale execution time in host: %d us\n\n", (int)(end - start));

}

void applyFilter(unsigned char* img_in, unsigned char* img_out, unsigned int width, unsigned int height, int color) {

    clock_t start = clock();

    if (color == rgba) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int x = 0; x < 4; x++) {
                    img_out[4 * (i * width + j) + x] = img_in[4 * (i * width + j) + x];
                }
            }
        }
    }
    else if (color == gray) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int hits = 0;
                float val = 0.0;
                for (int ky = -2; ky <= 2; ky++) {
                    for (int kx = -2; kx <= 2; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            hits++;
                            val += (float)(img_in[ny * width + nx]);
                        }
                    }
                }
                img_out[y * width + x] = (unsigned char)(val / hits);
            }
        }
    }

    clock_t end = clock();

    printf("\n\nfilter execution time in host: %d us\n\n", (int)(end - start));

}

int processImageCL(cl_device_id* device, unsigned char* img_in, unsigned char** img_out, unsigned int* width, unsigned int* height, int type, int* color) {

    FILE* fp;

    size_t buf_size = 0;
    size_t buf_size_out = 0;

    size_t global_work_size = 0;

    switch (type) {
    case 1:
        fp = fopen("../../kernels/ResizeImage.cl", "r");
        buf_size = (size_t)(sizeof(unsigned char) * (*width) * (*height));
        global_work_size = (size_t)((*width) * (*height) / 16);
        if (*color == 0) {
            buf_size = (size_t)(sizeof(unsigned char) * (*width) * (*height) * 4);
        }
        else if (*color == gray) {
            buf_size = (size_t)(sizeof(unsigned char) * (*width) * (*height));
        }
        else {
            return 1;
        }
        buf_size_out = (buf_size / 16);
        break;
    case 2:
        fp = fopen("../../kernels/GrayScaleImage.cl", "r");
        global_work_size = (size_t)((*width) * (*height));
        if (*color == gray) {
            buf_size = (size_t)(sizeof(unsigned char) * (*width) * (*height));
        }
        else if (*color == rgba) {
            buf_size = (size_t)(sizeof(unsigned char) * (*width) * (*height) * 4);
        }
        else {
            return 1;
        }
        buf_size_out = (size_t)(sizeof(unsigned char) * (*width) * (*height));
        break;
    case 3:
        fp = fopen("../../kernels/ApplyFilter.cl", "r");
        global_work_size = (size_t)((*width) * (*height));
        if (*color == rgba) {
            buf_size = (size_t)(sizeof(unsigned char) * (*width) * (*height) * 4);
        }
        else if (*color == gray) {
            buf_size = (size_t)(sizeof(unsigned char) * (*width) * (*height));
        }
        else {
            return 1;
        }
        buf_size_out = buf_size;
        break;
    default:
        return 1;
    }

    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    char* source_str = (char*)malloc(0x100000);
    size_t source_size = fread(source_str, 1, 0x100000, fp);
    fclose(fp);

    cl_int err;

    cl_context context = clCreateContext(NULL, 1, device, NULL, NULL, &err);

    cl_command_queue queue = clCreateCommandQueue(context, *device, 0, NULL);

    cl_mem img_in_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size, img_in, &err);
    cl_mem img_out_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buf_size_out, NULL, &err);
    cl_mem width_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(unsigned int)), width, &err);
    cl_mem height_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(unsigned int)), height, &err);
    cl_mem color_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (sizeof(int)), color, &err);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);

    err = clBuildProgram(program, 1, device, NULL, NULL, NULL);

    cl_kernel kernel;
    switch (type) {
    case 1:
        kernel = clCreateKernel(program, "ResizeImage", &err);
        break;
    case 2:
        kernel = clCreateKernel(program, "GrayScaleImage", &err);
        break;
    case 3:
        kernel = clCreateKernel(program, "ApplyFilter", &err);
        break;
    default:
        return 1;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&img_in_cl);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&img_out_cl);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&width_cl);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&height_cl);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&color_cl);

    size_t local_work_size = 20;

    clock_t start = clock();

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

    err = clEnqueueReadBuffer(queue, img_out_cl, CL_TRUE, 0, buf_size_out, *img_out, 0, NULL, NULL);

    clock_t end = clock();

    // Free resources
    clReleaseMemObject(img_in_cl);
    clReleaseMemObject(img_out_cl);
    clReleaseMemObject(width_cl);
    clReleaseMemObject(height_cl);
    clReleaseMemObject(color_cl);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(source_str);

    if (type == 1) {
        printf("\n\nOpenCL execution time for resize: %d ms\n\n", (int)(end - start));
    }
    else if (type == 2) {
        printf("\n\nOpenCL execution time for grayscale: %d ms\n\n", (int)(end - start));
    }
    else if (type == 3) {
        printf("\n\nOpenCL execution time for filter: %d ms\n\n", (int)(end - start));
    }
    return (err != CL_SUCCESS) ? 1 : 0;
}

void ReadImage(char* filename, unsigned char** image, unsigned int* width, unsigned int* height, int type) {

    unsigned img_err;
    if (type == 0) {
        img_err = lodepng_decode32_file(image, width, height, filename);
    }
    else {
        img_err = lodepng_decode_file(image, width, height, filename, LCT_GREY, 8);
    }
    if (img_err) {
        printf("\n\nimage not found\n\n");
    }
}

void WriteImage(char* filename, unsigned char* image, unsigned int width, unsigned int height, int type) {
    const char* output_file = filename != NULL ? filename : "output.png";
    if (type == 0) {
        lodepng_encode32_file(output_file, image, width, height);
    }
    else {
        lodepng_encode_file(output_file, image, width, height, LCT_GREY, 8);
    }
    //free(image);
}

void device_info(cl_platform_id platform, cl_device_id device) { //print platform and device info
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
    printf("Device driver version: %s\n", value);
    free(value);

    // print c version supported by compiler for device
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
    value = (char*)malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
    printf("Device OpenCL C version: %s\n", value);
    free(value);

    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE,
        sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf("Device local mem type: %d\n", maxComputeUnits);

    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
        sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf("Device local mem size: %d\n", maxComputeUnits);

    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf("Device Parallel compute units: %d\n", maxComputeUnits);

    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
        sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf("Device max clock frequency: %d\n", maxComputeUnits);

    clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
        sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf("Device max constant buffer size: %d\n", maxComputeUnits);

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf("Device max work group size: %d\n", maxComputeUnits);

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
        sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf("Device max work item sizes: %d\n\n\n", maxComputeUnits);
}
