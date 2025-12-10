//#include "reduce.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define N_BLOCK_SIZE    512
#define N_BLOCK_NUM     128
#define WARP_MASK 0xffffffff

__global__ void cudaReduceGeneral(const float* d_data, const size_t size,
                                 double* devResult)
{
    __shared__ double buffer[N_BLOCK_SIZE / 32];
    double      sdata = 0.0f;
    size_t      warpId      = threadIdx.x  / 32;
    size_t      laneId      = threadIdx.x % 32;
    size_t      threadId    = blockDim.x * blockIdx.x + threadIdx.x;
    size_t      gridStride  = gridDim.x * blockDim.x;

    for(; threadId < size; threadId += gridStride)
        sdata += d_data[threadId];
    
    for ( int offset = 16; offset > 0; offset /= 2)
        sdata += __shfl_down_sync(WARP_MASK, sdata, offset);

    if( laneId == 0)
        buffer[warpId] = sdata;

    __syncthreads();

    if( warpId == 0 && laneId < N_BLOCK_SIZE / 32 ){

        double final_sum = buffer[laneId];


        for ( int offset = N_BLOCK_SIZE / 32 / 2; offset > 0; offset /= 2)
            final_sum += __shfl_down_sync(WARP_MASK, final_sum, offset);

        if(laneId == 0)
            devResult[blockIdx.x] = final_sum;
    }

}

float gpuReduce_shfl(const float* d_data, const int size)
{

    double  *devResult;
    double  *hostResult;
    double  result = 0.0;
    hostResult = (double*)malloc(N_BLOCK_NUM * sizeof(double));
    if (hostResult == NULL) {
        fprintf(stderr, "Unable to allocate host memory");
        exit(EXIT_FAILURE);
    }

    cudaMalloc(&devResult, N_BLOCK_SIZE * sizeof(double));

    memset(hostResult, 0, N_BLOCK_NUM * sizeof(double));
    cudaMemset(devResult, 0, N_BLOCK_NUM * sizeof(double));


    cudaReduceGeneral<<<N_BLOCK_NUM, N_BLOCK_SIZE>>>(d_data, size, devResult);

    cudaError_t cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess){
        fprintf(stderr, "Kernel failed to execute");
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(hostResult, devResult, N_BLOCK_NUM * sizeof(double), cudaMemcpyDeviceToHost);

    for (size_t n = 0; n < N_BLOCK_NUM; n++) {
        result += hostResult[n];
    }

    free(hostResult);
    cudaFree(devResult);
    printf("%f\n",result);
    return (float)result;
}
