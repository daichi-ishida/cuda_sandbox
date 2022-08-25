#include "kernel.h"

#include <random>
#include <chrono>
#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cuda_runtime.h>
#include <cstdio>

#define DIM     (256*256*100)
#define BLOCK 1024
#define GRID ((DIM + BLOCK - 1) / BLOCK)

#define SMEMDIM (((DIM % BLOCK) + 31) / 32)     // 256/32 = 8 
#define FULL_MASK 0xffffffff


__global__ void reduceShfl (float *g_idata, float *g_odata)
{
    // shared memory for each warp sum
    __shared__ float smem[32];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory size to be written
    unsigned int smem_size = (blockIdx.x < (int)DIM / BLOCK) ? 32 : SMEMDIM;

    // calculate lane index and warp index
    int laneIdx = threadIdx.x % warpSize;
    int warpIdx = threadIdx.x / warpSize;

    // calculate which thread will participate shfl operation
    unsigned mask = __ballot_sync(FULL_MASK, idx < DIM);
    float val;
    if (idx < DIM) 
    { 
        val = g_idata[idx]; 
        for (int offset = 16; offset > 0; offset /= 2)
        {
            val += __shfl_xor_sync(mask, val, offset);
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
            val += __shfl_xor_sync(mask, val, offset);
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

    bool bResult = false;

    // initialization
    constexpr int size = DIM;
    printf("  with array size %d  grid %d block %d\n", size, GRID, BLOCK);

    // allocate host memory
    float *h_idata = new float[size];
    double *h_double_idata = new double[size];
    thrust::host_vector<float> h_thrust_idata(size);

    // allocate device memory
    float *d_idata;
    cudaMalloc(&d_idata, size * sizeof(float));
    thrust::device_vector<float> d_thrust_idata(size);

    // init sum
    double h_normal_double_sum = 0.0;
    float h_normal_sum = 0.0f;
    float d_thrust_sum = 0.0f;
    float d_cublas_sum = 0.0f;
    float d_reduce_sum = 0.0f;

    // initialize the array
    for (int i = 0; i < size; ++i)
    {
        h_idata[i] = 0.1f;
        h_double_idata[i] = 0.1;
        h_thrust_idata[i] = h_idata[i];
    }

    // copy data
    d_thrust_idata = h_thrust_idata;
    cudaMemcpy(d_idata, h_idata, size * sizeof(float), cudaMemcpyHostToDevice);

    ///////////////////////
    //  host normal sum  //
    ///////////////////////

    //run
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < size; ++i)
    {
        h_normal_sum += h_idata[i];
    } 
    auto end = std::chrono::system_clock::now();

    // result
    printf("*** Host Normal Sum ***\n");
    printf("sum          : %f\n", h_normal_sum);
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time         : %lld ms\n", msec);
    
    start = std::chrono::system_clock::now();
    for (int i = 0; i < size; ++i)
    {
        h_normal_double_sum += h_double_idata[i];
    } 
    end = std::chrono::system_clock::now();

    // result
    printf("*** Host Normal Double Sum ***\n");
    printf("sum          : %lf\n", h_normal_double_sum);
    msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time         : %lld ms\n", msec);

    /////////////////////////
    //  device thrust sum  //
    /////////////////////////

    // run
    start = std::chrono::system_clock::now();
    d_thrust_sum = thrust::reduce(d_thrust_idata.begin(), d_thrust_idata.end());
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();

    // result
    printf("*** Device Thrust Sum ***\n");
    printf("sum          : %f\n", d_thrust_sum);
    msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time         : %lld ms\n", msec);

    d_thrust_idata.clear();
    d_thrust_idata.shrink_to_fit();

    /////////////////////////
    //  device cublas sum  //
    /////////////////////////

    // preparation
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    // run
    start = std::chrono::system_clock::now();
    cublasSasum(cublasHandle, size, d_idata, 1, &d_cublas_sum);
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();

    // result
    printf("*** Device cuBLAS Absolute Sum ***\n");
    printf("abs sum      : %f\n", d_cublas_sum);
    msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("time         : %lld ms\n", msec);

    // free memory
    cublasDestroy(cublasHandle);

    /////////////////////////
    //  device reduce sum  //
    /////////////////////////

    // preparation
    float *h_odata = new float[GRID];
    float *d_odata;
    cudaMalloc(&d_odata, GRID * sizeof(float));

    // run
    start = std::chrono::system_clock::now();
    reduceShfl<<<GRID, BLOCK>>>(d_idata, d_odata);
    cudaMemcpy(h_odata, d_odata, GRID * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < GRID; ++i) d_reduce_sum += h_odata[i];
    end = std::chrono::system_clock::now();

    // result
    printf("*** Device Reduced Max ***\n");
    printf("max          : %f\n", d_reduce_sum);
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
    float ans = (float)size / 10.0f;
    bResult = (h_normal_sum == ans && d_thrust_sum == ans && d_cublas_sum == ans && d_reduce_sum == ans);

    if(!bResult)
    {
        printf("!!!!!!!!!!!!!!!!!!!\n");
        printf("!!  Test failed  !!\n");
        printf("!!!!!!!!!!!!!!!!!!!\n");
    }
}
