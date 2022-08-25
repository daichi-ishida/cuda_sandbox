#include "kernel.h"
#include <chrono>

#include <thrust/device_malloc.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <random>

#include "helper_cuda.h"

constexpr int DIM = 256;

void cuda_main()
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("device %d: %s \n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // initialization
    constexpr int size = DIM;

    // allocate host memory
    int *h_idata = new int[size];
    int *h_odata = new int[size];
    int pos = 0;

    // allocate device memory
    thrust::device_ptr<int> d_thrust_idata;
    thrust::device_ptr<int> d_thrust_odata;
    d_thrust_idata = thrust::device_malloc<int>(size);
    d_thrust_odata = thrust::device_malloc<int>(size);

    // initialize the array
    for (int i = 0; i < size; ++i)
    {
        h_idata[i] = 2*i + 1;
        printf("%d ", h_idata[i]);
    }
    printf("\n");

    thrust::copy(h_idata, h_idata + size, d_thrust_idata);

    /////////////////
    //  host scan  //
    /////////////////

    // run
    auto start = std::chrono::system_clock::now();
    cudaMemcpy(h_idata, d_thrust_idata.get(), sizeof(float) * size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
    {
        h_odata[i] = pos;
        pos += h_idata[i];
    }
    cudaMemcpy(d_thrust_odata.get(), h_odata, sizeof(float) * size, cudaMemcpyHostToDevice);
    auto end = std::chrono::system_clock::now();

    // result
    printf("*** Host scan ***\n");
    for (int i = 0; i < size; ++i)
    {
        printf("%d ", h_odata[i]);
    }
    printf("\n");
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time         : %lld ms\n", msec);

    //////////////////////////
    //  device thrust scan  //
    //////////////////////////

    // run
    start = std::chrono::system_clock::now();
    thrust::exclusive_scan(d_thrust_idata, d_thrust_idata+size, d_thrust_odata);
    checkCudaErrors(cudaDeviceSynchronize());
    end = std::chrono::system_clock::now();

    thrust::copy(d_thrust_odata, d_thrust_odata + size, h_odata);

    // result
    printf("*** Device Thrust Scan ***\n");
    for (int i = 0; i < size; ++i)
    {
        printf("%d ", h_odata[i]);
    }
    printf("\n");
    msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time         : %lld ms\n", msec);

    delete[] h_idata;
    delete[] h_odata;

    // reset device
    cudaDeviceReset();
}