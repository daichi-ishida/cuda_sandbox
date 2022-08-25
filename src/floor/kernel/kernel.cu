#include "kernel.h"

#include <cstdio>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

__global__ void mark_k(float *out)
{
    int x = threadIdx.x;
    out[x] = floorf(-0.1f);
    int ret = __float2int_rd(-0.1f);
    if(ret == -1)
    {
        printf("ok\n");
    }
}

void cuda_main()
{
    float *h_data = new float[1];
    thrust::device_ptr<float> d_thrust_data = thrust::device_malloc<float>(1);

    mark_k<<<1, 1024>>>(d_thrust_data.get());

    cudaMemcpy(h_data, d_thrust_data.get(), sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", h_data[0]);

    thrust::device_free(d_thrust_data);
    delete[] h_data;
}
