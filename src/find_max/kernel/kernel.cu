#include "kernel.h"

#include <random>
#include <chrono>
#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

#define DIM     (256 * 256 * 100)
#define SMEMDIM (((DIM % 1024) + 31) / 32)     // 256/32 = 8 
#define FULL_MASK 0xffffffff

#define BLOCK 1024
#define GRID ((DIM + BLOCK - 1) / BLOCK)

#define RANGE (float)(256 * 256 * 512)


__global__ void reduceShfl(float *g_idata, float *g_odata)
{
    // shared memory for each warp sum
    __shared__ float smem[32];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory size to be written
    unsigned int smem_size = (blockIdx.x < (int)(DIM / 1024)) ? 32 : SMEMDIM;

    // calculate lane index and warp index
    int laneIdx = threadIdx.x % warpSize;
    int warpIdx = threadIdx.x / warpSize;

    // calculate which thread will participate shfl operation
    unsigned mask = __ballot_sync(FULL_MASK, idx < DIM);
    float val;
    if (idx < DIM) 
    { 
        val = fabsf(g_idata[idx]); 
        for (int offset = 16; offset > 0; offset /= 2)
        {
            float temp = val;
            val = fmaxf(__shfl_xor_sync(mask, val, offset), temp);
        }
    }

    // save warp sum to shared memory
    if (laneIdx == 0) 
    {
        smem[warpIdx] = val;
    }

    // block synchronization
    __syncthreads();

    // last warp reduce
    mask = __ballot_sync(FULL_MASK, threadIdx.x < smem_size);
    if (threadIdx.x < smem_size) 
    { 
        val = smem[laneIdx];
        for (int offset = 16; offset > 0; offset /= 2)
        {
            float temp = val;
            val = fmaxf(__shfl_xor_sync(mask, val, offset), temp);
        }
    }

    // write result for this block to global mem
    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = val;
    }
}


void cuda_main()
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("device %d: %s ", dev, deviceProp.name);
    cudaSetDevice(dev);

    std::random_device rnd;
    std::mt19937 engine(rnd()); 
    std::uniform_real_distribution<float> dist(-RANGE, RANGE);

    bool bResult = false;

    // initialization
    constexpr int size = DIM;
    printf("  with array size %d  grid %d block %d\n", size, GRID, BLOCK);

    // allocate host memory
    float *h_idata = new float[size];
    thrust::host_vector<float> h_thrust_idata(size);

    // allocate device memory
    float *d_idata;
    cudaMalloc(&d_idata, size * sizeof(float));
    thrust::device_vector<float> d_thrust_idata(size);

    // init max
    float h_normal_max = 0.0f;
    float d_thrust_max = 0.0f;
    float d_cublas_max = 0.0f;
    float d_reduce_max = 0.0f;

    // initialize the array
    for (int i = 0; i < size; ++i)
    {
        h_idata[i] = dist(engine);
        h_thrust_idata[i] = h_idata[i];
    }

    // copy data
    d_thrust_idata = h_thrust_idata;
    cudaMemcpy(d_idata, h_idata, size * sizeof(float), cudaMemcpyHostToDevice);

    ///////////////////////////
    //  host normal abs max  //
    ///////////////////////////

    //run
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < size; ++i)
    {
        float abs = (h_idata[i] < 0) ? -h_idata[i] : h_idata[i];
        if(h_normal_max < abs) h_normal_max = abs;
    } 
    auto end = std::chrono::system_clock::now();

    // result
    printf("*** Host Normal Absolute Max ***\n");
    printf("max          : %lf\n", h_normal_max);
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time         : %lld ms\n", msec);
    
    /////////////////////////////
    //  device thrust abs max  //
    /////////////////////////////

    // run
    start = std::chrono::system_clock::now();
    thrust::pair<thrust::device_vector<float>::iterator,thrust::device_vector<float>::iterator> result = thrust::minmax_element(d_thrust_idata.begin(), d_thrust_idata.end());
    float t_thrust_min = *result.first;
    float t_thrust_max = *result.second;
    d_thrust_max = (-t_thrust_min > t_thrust_max) ? -t_thrust_min : t_thrust_max;
    end = std::chrono::system_clock::now();

    // result
    printf("*** Device Thrust Absolute Max ***\n");
    printf("max          : %lf\n", d_thrust_max);
    msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time         : %lld ms\n", msec);

    d_thrust_idata.clear();
    d_thrust_idata.shrink_to_fit();

    /////////////////////////////
    //  device cublas abs max  //
    /////////////////////////////

    // preparation
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    int maxIdx;

    // run
    start = std::chrono::system_clock::now();
    cublasIsamax(cublasHandle, size, d_idata, 1, &maxIdx);
    cudaMemcpy(&d_cublas_max, d_idata + (maxIdx - 1), sizeof(float), cudaMemcpyDeviceToHost);
    if(d_cublas_max < 0) d_cublas_max *= -1.0f;
    end = std::chrono::system_clock::now();

    // result
    printf("*** Device cuBLAS Absolute Max ***\n");
    printf("max          : %lf\n", d_cublas_max);
    msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time         : %lld ms\n", msec);

    // free memory
    cublasDestroy(cublasHandle);

    /////////////////////////////
    //  device reduce abs max  //
    /////////////////////////////

    // preparation
    float *h_odata = new float[GRID];
    float *d_odata;
    cudaMalloc(&d_odata, GRID * sizeof(float));

    // run
    start = std::chrono::system_clock::now();
    reduceShfl<<<GRID, BLOCK>>>(d_idata, d_odata);
    cudaMemcpy(h_odata, d_odata, GRID * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < GRID; ++i) d_reduce_max = (d_reduce_max < h_odata[i]) ? h_odata[i] : d_reduce_max;
    end = std::chrono::system_clock::now();

    // result
    printf("*** Device Reduced Absolute Max ***\n");
    printf("max          : %lf\n", d_reduce_max);
    msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time         : %lld ms\n", msec);
    printf("grids        : <<<%d, %d>>>\n", GRID, BLOCK);

    // free memory
    delete[] h_odata;
    cudaFree(d_odata);

    //////////////////////////////

    // free host memory
    delete[] h_idata;

    // free device memory
    cudaFree(d_idata);

    // reset device
    cudaDeviceReset();

    // check the results
    bResult = (d_reduce_max == h_normal_max);

    if(!bResult)
    {
        printf("!!!!!!!!!!!!!!!!!!!\n");
        printf("!!  Test failed  !!\n");
        printf("!!!!!!!!!!!!!!!!!!!\n");
    }
}
