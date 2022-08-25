#include "kernel.h"
#include <chrono>

#include <thrust/device_malloc.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>

#include <cuda_runtime.h>
#include <cstdio>

#include "helper_cuda.h"

constexpr int DIM = 256;
constexpr int THREAD = 32;

__global__ void test(const float *d_idata, float *d_odata)
{
    int size = blockDim.x * blockDim.y;
    int block_offset = blockIdx.x + blockIdx.y * gridDim.x;
    d_odata[block_offset] = thrust::reduce(thrust::device, d_idata + size * block_offset, d_idata + size * (block_offset + 1));
}

void cuda_main()
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("device %d: %s \n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // initialization
    constexpr int size = DIM * DIM;
    constexpr int osize = size / (THREAD * THREAD);

    // allocate device memory
    thrust::device_ptr<float> d_thrust_idata;
    thrust::device_ptr<float> d_thrust_odata;

    d_thrust_idata = thrust::device_malloc<float>(size);
    d_thrust_odata = thrust::device_malloc<float>(osize);

    // initialize the array
    for(int b = 0; b < osize; ++b)
        for (int i = 0; i < THREAD * THREAD; ++i)
        {
            d_thrust_idata[i + b * THREAD * THREAD] = (float)(b + 1);
        }

    /////////////////////////
    //  device thrust sum  //
    /////////////////////////

    // run
    auto start = std::chrono::system_clock::now();
    dim3 blocks(DIM / THREAD, DIM / THREAD);
    dim3 threads(THREAD, THREAD);
    test<<<blocks, threads>>>(d_thrust_idata.get(), d_thrust_odata.get());
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::system_clock::now();

    float *h_odata = new float[osize];
    thrust::copy(d_thrust_odata, d_thrust_odata + osize, h_odata);

    // result
    printf("*** Device Thrust Reduce ***\n");
    for(int b = 0; b < osize; ++b)
    {
        printf("%f ", h_odata[b]);
    }
    printf("\n");
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time         : %lld ms\n", msec);

    delete[] h_odata;

    // reset device
    cudaDeviceReset();
}