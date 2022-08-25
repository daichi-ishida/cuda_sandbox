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

    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_int_distribution<> rand(0, size); 

    // allocate host memory
    int *h_key = new int[size];
    int *h_value = new int[size];

    // allocate device memory
    thrust::device_ptr<int> d_thrust_key;
    thrust::device_ptr<int> d_thrust_value;
    d_thrust_key = thrust::device_malloc<int>(size);
    d_thrust_value = thrust::device_malloc<int>(size);

    // initialize the array
    for (int i = 0; i < size; ++i)
    {
        h_key[i] = rand(mt);
        printf("%d ", h_key[i]);
    }
    printf("\n");


    thrust::copy(h_key, h_key + size, d_thrust_key);
    thrust::sequence(d_thrust_value, d_thrust_value + size);

    //////////////////////////
    //  device thrust sort  //
    //////////////////////////

    // run
    auto start = std::chrono::system_clock::now();
    thrust::sort_by_key(d_thrust_key, d_thrust_key+size, d_thrust_value);
    thrust::reverse(d_thrust_value, d_thrust_value + size);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::system_clock::now();

    thrust::copy(d_thrust_value, d_thrust_value + size, h_value);

    // result
    printf("*** Device Thrust Scan ***\n");
    for (int i = 0; i < size; ++i)
    {
        printf("%d ", h_key[h_value[i]]);
    }
    printf("\n");
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time         : %lld ms\n", msec);

    delete[] h_key;
    delete[] h_value;

    // reset device
    cudaDeviceReset();
}