#include "kernel.h"

#include <cuda_runtime.h>

#include <iostream>


void cuda_main()
{
    cudaError_t cudaStatus = cudaSetDevice(0);

    size_t free, total, use;

    // Before memory allocation
    cudaMemGetInfo(&free, &total);
    use = total - free;
    use /= 1024 * 1024;
    std::cout << "[before allocation] use:  " << use << " [MB]\n";

    int N = 52428800; // 200MB
    float *x;
    size_t memsize = sizeof(float) * N;

    cudaMalloc(&x, memsize);

    // After memory allocation.
    cudaMemGetInfo(&free, &total);
    use = total - free;
    use /= 1024 * 1024;
    std::cout << "[after allocation] use:   " << use << " [MB]\n";

    cudaFree(x);

    // After release memory.
    cudaMemGetInfo(&free, &total);
    use = total - free;
    use /= 1024 * 1024;
    std::cout << "[after release] use:      " << use << " [MB]\n";
}