#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include "../../LodePNG/lodepng.c"
#include "../../LodePNG/lodepng.h"
#pragma comment(lib, "OpenCL.lib")

void resizeImage(unsigned char* img_in, unsigned char* img_out, unsigned int width, unsigned int height, int color);
void grayScaleImage(unsigned char* img_in, unsigned char* img_out, unsigned int width, unsigned int height, int color);
void applyFilter(unsigned char* img_in, unsigned char* img_out, unsigned int width, unsigned int height, int color);
int processImageCL(cl_device_id* device, unsigned char* img_in, unsigned char** img_out, unsigned int* width, unsigned int* height, int type, int* color);
void ReadImage(char* filename, unsigned char** image, unsigned int* width, unsigned int* height, int type);
void WriteImage(char* filename, unsigned char* image, unsigned int width, unsigned int height, int type);

#define resize 1
#define grayscale 2
#define filter 3

#define rgba 0
#define gray 1

int main() {

    cl_int           err;
    cl_platform_id platform;
    cl_device_id     device;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err) {
        printf("No platform detected, exit\n");
        exit(1);
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    if (err) {
        printf("No device detected, exit\n");
        exit(1);
    }

    /* Read Images */

    unsigned char* img = NULL;
    unsigned int width = NULL;
    unsigned int height = NULL;

    unsigned char* img1 = NULL;
    unsigned int width1 = NULL;
    unsigned int height1 = NULL;
    
    ReadImage("../../im0.png", &img, &width, &height, 0);
    
    unsigned char* resized_img = (unsigned char*)malloc(sizeof(unsigned char) * (int)(width/4) * (int)(height/4) * 4);

    int color = 0;

    ReadImage("../../im1.png", &img1, &width1, &height1, 0);

    unsigned char* resized_img1 = (unsigned char*)malloc(sizeof(unsigned char) * (int)(width1 / 4) * (int)(height1 / 4) * 4);

    color = 0;

    processImageCL(&device, img, &resized_img, &width, &height, resize, &color);

    resizeImage(img1, resized_img1, width1, height1, color);

    WriteImage("resized0.png", resized_img, (unsigned int)(width/4), (unsigned int)(height/4), rgba);
    WriteImage("resized1.png", resized_img1, (unsigned int)(width1 / 4), (unsigned int)(height1 / 4), rgba);

    ReadImage("./resized0.png", &img, &width, &height, 0);
    ReadImage("./resized1.png", &img1, &width1, &height1, 0);

    unsigned char* grayscale_img = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
    unsigned char* grayscale_img1 = (unsigned char*)malloc(sizeof(unsigned char) * width1 * height1);

    color = 0;
    processImageCL(&device, img, &grayscale_img, &width, &height, grayscale, &color);

    grayScaleImage(img1, grayscale_img1, width1, height1, color);

    WriteImage("grayscaleImage0.png", grayscale_img, width, height, 1);
    WriteImage("grayscaleImage1.png", grayscale_img1, width1, height1, 1);


    ReadImage("./grayscaleImage0.png", &img, &width, &height, 1);
    ReadImage("./grayscaleImage1.png", &img1, &width1, &height1, 1);

    unsigned char* filtered_img = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
    unsigned char* filtered_img1 = (unsigned char*)malloc(sizeof(unsigned char) * width1 * height1);

    color = 1;
    processImageCL(&device, img, &filtered_img, &width, &height, filter, &color);

    applyFilter(img1, filtered_img1, width1, height1, color);

    WriteImage("filteredimage0.png", filtered_img, width, height, 1);
    WriteImage("filteredimage1.png", filtered_img1, width1, height1, 1);

    free(img);
    free(img1);

	return 0;
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

                img_out[i*re_width+j] = img_in[i*4*width+4*j];
                
            }
        }
    }

    clock_t end = clock();

    printf("\n\nresize execution time in host: %d ms\n\n", (int)(end - start));

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

    printf("\n\ngrayscale execution time in host: %d ms\n\n", (int)(end - start));

}

void applyFilter(unsigned char* img_in, unsigned char* img_out, unsigned int width, unsigned int height, int color) {

    clock_t start = clock();

    if (color == rgba) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int x = 0; x < 4; x++) {
                    img_out[4*(i * width + j) + x] = img_in[4*(i * width + j) + x];
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

    printf("\n\nfilter execution time in host: %d ms\n\n", (int)(end - start));

}

int processImageCL(cl_device_id* device, unsigned char* img_in, unsigned char** img_out, unsigned int* width, unsigned int* height, int type, int* color) {

    FILE* fp;

    size_t buf_size = 0;
    size_t buf_size_out = 0;

    size_t global_work_size = 0;

    switch (type) {
        case 1:
            fp = fopen("./ResizeImage.cl", "r");
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
            fp = fopen("./GrayScaleImage.cl", "r");
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
            fp = fopen("./ApplyFilter.cl", "r");
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
    free(image);
}