#include "kernel.h"
#include <chrono>

#include <cstdio>

#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/swap.h>
#include "helper_cuda.h"

constexpr int size = 10;


struct Structure 
{
    float *data;
};

__global__ void kernel(Structure *str, float *buf) 
{
    int x = threadIdx.x;
    buf[x] = str->data[x];
}

void cuda_main()
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("device %d: %s \n", dev, deviceProp.name);
    cudaSetDevice(dev);

    Structure h_struct;
    thrust::device_ptr<Structure> d_struct;
    thrust::device_ptr<Structure> d_struct0;

    // Structure *d_struct;
    
    float *d_tmp_data;
    float *d_tmp_data0;

    float *h_buffer = new float[size];
    float *d_buffer;

    h_struct.data = new float[size];
    for(int i = 0; i < size; ++i)
    {
        h_struct.data[i] = i;
        printf("init: %f\n", h_struct.data[i]);
    }

    // checkCudaErrors(cudaMalloc(&d_struct, sizeof(Structure)));
    d_struct = thrust::device_malloc<Structure>(1);
    d_struct0 = thrust::device_malloc<Structure>(1);

    checkCudaErrors(cudaMalloc(&d_tmp_data, sizeof(float) * size));
    checkCudaErrors(cudaMalloc(&d_tmp_data0, sizeof(float) * size));
    checkCudaErrors(cudaMalloc(&d_buffer, sizeof(float) * size));

    printf("end allocation\n");

    checkCudaErrors(cudaMemcpy(d_tmp_data, h_struct.data, sizeof(float) * size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_tmp_data0, 0, sizeof(float) * size));

    printf("%d\n", sizeof(h_struct.data));
    checkCudaErrors(cudaMemcpy(&(d_struct.get()->data), &d_tmp_data, sizeof(d_struct.get()->data), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(d_struct0.get()->data), &d_tmp_data0, sizeof(d_struct0.get()->data), cudaMemcpyHostToDevice));

    printf("end copy\n");

    // thrust::swap(d_struct, d_struct0);
    kernel<<<1, size>>>(d_struct.get(), d_buffer);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("end kernel\n");

    checkCudaErrors(cudaMemcpy(h_buffer, d_buffer, sizeof(float) * size, cudaMemcpyDeviceToHost));

    printf("end copy\n");

    for(int i = 0; i < size; ++i)
    {
        printf("host: %f\n", h_buffer[i]);
    }

    delete[] h_struct.data;
    delete[] h_buffer;
    cudaFree(d_tmp_data);

    // cudaFree(d_struct);
    thrust::device_free(d_struct);
    cudaFree(d_buffer);

    // reset device
    cudaDeviceReset();
}