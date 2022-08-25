#include "kernel.h"
#include <chrono>

#include <thrust/device_malloc.h>
#include <thrust/transform.h>
#include <thrust/copy.h>

#include <cuda_runtime.h>
#include <cstdio>

#include "helper_cuda.h"

constexpr int DIM = 10;

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
    float *h_x = new float[size];
    float *h_y = new float[size];
    float *h_z = new float[size];

    // allocate device memory
    thrust::device_ptr<float> d_thrust_x;
    thrust::device_ptr<float> d_thrust_y;
    thrust::device_ptr<float> d_thrust_z;
    d_thrust_x = thrust::device_malloc<float>(size);
    d_thrust_y = thrust::device_malloc<float>(size);
    d_thrust_z = thrust::device_malloc<float>(size);

    // initialize the array
    for (int i = 0; i < size; ++i)
    {
        h_x[i] = 2.0f*(float)i + 1.0f;
        h_y[i] = 2.0f;
        printf("%f ", h_x[i]);
    }
    printf("\n");

    thrust::copy(h_x, h_x + size, d_thrust_x);
    thrust::copy(h_y, h_y + size, d_thrust_y);

    ////////////////////////////////////
    //  device thrust multiplication  //
    ////////////////////////////////////

    thrust::divides<float> op;

    // run
    auto start = std::chrono::system_clock::now();
    thrust::transform(d_thrust_x, d_thrust_x + size, d_thrust_y, d_thrust_z, op);
    auto end = std::chrono::system_clock::now();

    thrust::copy(d_thrust_z, d_thrust_z + size, h_z);

    // result
    printf("*** Device Thrust Multiplication ***\n");
    for (int i = 0; i < size; ++i)
    {
        printf("%f ", h_z[i]);
    }
    printf("\n");
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time         : %lld ms\n", msec);

    delete[] h_x;
    delete[] h_y;
    delete[] h_z;

    // reset device
    cudaDeviceReset();
}