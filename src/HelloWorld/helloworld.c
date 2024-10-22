
#include <stdio.h>
#include <stdlib.h>

//Depending of your installation more includes should be uses, check your particular SDK installation
//
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#pragma comment(lib, "OpenCL.lib")

//The rest of the code is intended to be executed in the host
int main()
{
    cl_int           err;
    cl_uint          num_platforms;
    cl_platform_id* platforms;
    cl_device_id     device;
    cl_context       context;
    cl_command_queue queue;
    cl_program       program;
    cl_kernel        kernel;
    cl_mem           output;

    char* result = (char*)malloc(sizeof(char) * 13);

    // PLATFORM
    // In this example we will only consider one platform
    //
    int num_max_platforms = 1;
    err = clGetPlatformIDs(num_max_platforms, NULL, &num_platforms);
    printf("Num platforms detected: %d\n", num_platforms);

    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_max_platforms, platforms, &num_platforms);

    if (num_platforms < 1)
    {
        printf("No platform detected, exit\n");
        exit(1);
    }

    //DEVICE (could be CL_DEVICE_TYPE_GPU)
    //
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    //CONTEXT
    //
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    //QUEUE
    //
    queue = clCreateCommandQueue(context, device, 0, &err);

    //READ KERNEL AND COMPILE IT
    //

    FILE* fp = fopen("./helloworld.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    char* source_str = (char*)malloc(0x100000);
    size_t source_size = fread(source_str, 1, 0x100000, fp);
    fclose(fp);

    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);


    //CREATE KERNEL AND KERNEL PARAMETERS
    //
    kernel = clCreateKernel(program, "hello", &err);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 14 * sizeof(char), NULL, &err);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&output);

    //EXECUTE KERNEL!
    //
    err = clEnqueueTask(queue, kernel, 0, NULL, NULL);

    //READ KERNEL OUTPUT
    //
    err = clEnqueueReadBuffer(queue, output, CL_TRUE, 0, 14 * sizeof(char), result, 0, NULL, NULL);
    printf("%s\n", result);


    //Free your memory please....
    clReleaseMemObject(output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(platforms);


    return 0;
}